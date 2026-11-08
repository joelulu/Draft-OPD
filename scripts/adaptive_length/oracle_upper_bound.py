"""B=1 local-oracle discovery followed by independent schedule replay.

No learned accuracy claim: predictor outputs are discarded, only their real
native layer-0 full/window-64 reader + 128-wide classifier cost is charged.
Discovery's 8-way search is NEVER included in replay throughput. A local
one-round oracle is not a globally optimal sequence policy.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

KS = [4, 8, 12, 16, 20, 24, 28, 32]


def emitted_count(block_ids, accepted, remaining, eos):
    ids = block_ids[:min(accepted + 1, remaining)]
    if eos is not None and eos in ids:
        return ids.index(eos) + 1
    return len(ids)


def select_trial(trials, objective):
    """Nominal-K ties choose the smallest K, including clipped tail aliases."""
    if objective not in ('accept', 'speed'):
        raise ValueError(objective)
    def value(t):
        if objective == 'accept':
            return t['emitted'] - 1
        return t['emitted'] / statistics.median(t['total_ms'])
    return max(trials, key=lambda t: (value(t), -t['requested_k']))


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def signature(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


class EventProfiler:
    """Separate attribution run, never used for primary tok/s."""
    def __init__(self, torch, device):
        self.torch, self.device, self.events = torch, device, []

    def measure(self, name):
        from contextlib import contextmanager
        @contextmanager
        def scope():
            if self.device.type == 'cuda':
                begin = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                begin.record()
                yield
                end.record()
                self.events.append((name, begin, end))
            else:
                begin = time.perf_counter()
                yield
                self.events.append((name, (time.perf_counter()-begin)*1000, None))
        return scope()

    def resolve(self):
        totals = Counter()
        for name, begin, end in self.events:
            totals[name] += begin.elapsed_time(end) if end is not None else begin
        return dict(totals)


def native_predictor_probe(runtime, mode, head, torch):
    """Exact frozen native layer-0 aggregation, with full or last-64 history.

    Uses the real draft cache. Context prep is reused by propose, rather than
    charged twice. Query sees anchor self-KV and no future MASK positions.
    """
    from adaptive_length.two_stage.dflash_features import (
        materialize_context, read_anchor, _ReadOnlyPrefix, check_draft,
    )
    from adaptive_length.two_stage.classifier import cache_pairs
    start = runtime.prefix.shape[1]
    cached = runtime.draft_cache.get_seq_length()
    if not runtime.context_start <= cached <= start:
        raise RuntimeError('Native selector context misaligned')
    materialize_context(runtime.draft, runtime.draft_cache,
        runtime.context[:, cached-runtime.context_start:], cached)
    embedding = runtime.target.get_input_embeddings()(runtime.anchor)
    if mode == 'full_kv':
        feature = read_anchor(runtime.draft, runtime.draft_cache, embedding, start, 'full_kv')
    elif mode == 'window_64':
        draft = runtime.draft
        check_draft(draft)
        keys, values = cache_pairs(runtime.draft_cache)[0]
        keys, values = keys[..., -64:, :], values[..., -64:, :]
        positions = torch.arange(start-keys.shape[-2], start+1, device=keys.device)
        allowed = torch.ones_like(positions, dtype=torch.bool)
        window = draft.layers[0].self_attn.sliding_window
        if window is not None:
            allowed = positions > start-int(window)
        anchor_pos = positions.new_tensor([[start]])
        feature = draft.layers[0](hidden_states=embedding,
            target_hidden=embedding.new_empty((1, 0, draft.config.hidden_size)),
            position_ids=anchor_pos, position_embeddings=draft.rotary_emb(embedding, anchor_pos),
            attention_mask=allowed[None, None, None, :],
            past_key_value=_ReadOnlyPrefix(keys, values), use_cache=True)[:, 0]
    else:
        raise ValueError(mode)
    # Include readback/synchronization of a real class decision, then discard it.
    return int(head(feature.float()).argmax(-1).item())


def discover(target, draft, ids, objective, args, eos, Runtime, clock, torch):
    state = Runtime(target, draft, ids)
    prompt_len = ids.shape[1]
    rng = random.Random(args.seed)
    trace, all_trials = [], []
    begin = time.monotonic()
    while state.prefix.shape[1]-prompt_len < args.max_new_tokens:
        remaining = args.max_new_tokens-(state.prefix.shape[1]-prompt_len)
        if remaining == 1:
            count = state.ar_step()
            trace.append(dict(requested_k=0, actual_k=0, accepted_raw=0, committed=count))
        else:
            boundary = state.draft_cache.get_seq_length()
            prefix_len = state.prefix.shape[1]
            # Warm each actual shape outside all trial timing, restoring pending KV.
            for k in sorted({min(k, remaining-1) for k in KS}):
                block = state.propose(k)
                state.verify(block)
                state.draft_cache.crop(boundary)
            trials = {k: dict(requested_k=k, actual_k=min(k, remaining-1),
                             total_ms=[], draft_ms=[], verify_ms=[]) for k in KS}
            for _ in range(args.search_repeats):
                order = KS.copy()
                rng.shuffle(order)
                for k in order:
                    t0 = clock()
                    block = state.propose(trials[k]['actual_k'])
                    t1 = clock()
                    result = state.verify(block)
                    t2 = clock()
                    block_ids = block[0].tolist()
                    emitted = emitted_count(block_ids, result[0], remaining, eos)
                    trial = trials[k]
                    if 'block_ids' in trial and (trial['block_ids'] != block_ids or
                                                trial['accepted_raw'] != result[0]):
                        raise RuntimeError('Repeated proposals/acceptance changed at identical state')
                    trial.update(block_ids=block_ids, accepted_raw=result[0], emitted=emitted)
                    trial['total_ms'].append((t2-t0)*1000)
                    trial['draft_ms'].append((t1-t0)*1000)
                    trial['verify_ms'].append((t2-t1)*1000)
                    state.draft_cache.crop(boundary)
                    if (state.target_cache.get_seq_length() != prefix_len or
                            state.draft_cache.get_seq_length() != boundary):
                        raise RuntimeError('Oracle trial polluted cache lengths')
            winner = select_trial(list(trials.values()), objective)
            block = state.propose(winner['actual_k'])
            result = state.verify(block, keep_cache=True)
            if block[0].tolist() != winner['block_ids'] or result[0] != winner['accepted_raw']:
                raise RuntimeError('Selected trial differs on execution')
            count = state.advance(block, result, cache_kept=True)
            trace.append(dict(requested_k=winner['requested_k'], actual_k=winner['actual_k'],
                              accepted_raw=result[0], committed=count))
            all_trials.append(dict(round=len(trace)-1, context_length=prefix_len,
                                   trials=list(trials.values())))
        generated = state.prefix[0, prompt_len:].tolist()
        if eos is not None and eos in generated:
            break
        if len(trace) % 5 == 0:
            print(json.dumps(dict(event='oracle_search', objective=objective, rounds=len(trace),
                tokens=len(generated), elapsed_s=round(time.monotonic()-begin, 1))), flush=True)
    generated = state.prefix[0, prompt_len:].tolist()[:args.max_new_tokens]
    if eos is not None and eos in generated:
        generated = generated[:generated.index(eos)+1]
    return dict(objective=objective, trace=trace, generated=generated,
                trials=all_trials, search_wall_s=time.monotonic()-begin)


def replay(target, draft, ids, schedule, fixed_k, selector, head, args, eos,
           Runtime, clock, torch, profile=False):
    t0 = clock()
    state = Runtime(target, draft, ids)
    prefill_s = clock()-t0
    prompt_len = ids.shape[1]
    profiler = EventProfiler(torch, ids.device) if profile else None
    measure = profiler.measure if profiler else lambda name: nullcontext()
    records = []
    selector_wall_ms, selector_calls = 0., 0
    begin = clock()
    while state.prefix.shape[1]-prompt_len < args.max_new_tokens:
        remaining = args.max_new_tokens-(state.prefix.shape[1]-prompt_len)
        if schedule is not None:
            if len(records) >= len(schedule['trace']):
                raise RuntimeError('Replay exceeded oracle schedule')
            requested = schedule['trace'][len(records)]['requested_k']
        else:
            requested = fixed_k if remaining > 1 else 0
        k = min(requested, remaining-1) if requested else 0
        if k:
            if selector:
                t0 = time.perf_counter()
                with measure('selector_gpu_ms'):
                    native_predictor_probe(state, selector, head, torch)
                selector_wall_ms += (time.perf_counter()-t0)*1000
                selector_calls += 1
            with measure('draft_gpu_ms'):
                block = state.propose(k)
            with measure('verify_gpu_ms'):
                result = state.verify(block, keep_cache=True)
            with measure('advance_gpu_ms'):
                count = state.advance(block, result, cache_kept=True)
            accepted = result[0]
        else:
            with measure('ar_gpu_ms'):
                count = state.ar_step()
            accepted = 0
        records.append(dict(requested_k=requested, actual_k=k,
                            accepted_raw=accepted, committed=count))
        # Same stop check for all methods, included in wall time.
        generated = state.prefix[0, prompt_len:]
        if eos is not None and bool((generated == eos).any()):
            break
    decode_s = clock()-begin
    generated = state.prefix[0, prompt_len:prompt_len+args.max_new_tokens].tolist()
    if eos is not None and eos in generated:
        generated = generated[:generated.index(eos)+1]
    # All replay validation happens AFTER timed decode.
    if schedule is not None:
        if generated != schedule['generated'] or records != schedule['trace']:
            raise RuntimeError('Replay differs from oracle: outputs or per-round acceptance changed')
    effective = [dict(r) for r in records]
    excess = sum(r['committed'] for r in records)-len(generated)
    if excess:
        effective[-1]['committed'] -= excess
    spec = [r for r in effective if r['actual_k']]
    round_details = [dict(r, round=i+1,
        accepted=max(0, r['committed']-1) if r['actual_k'] else 0,
        tail=0 < r['actual_k'] < r['requested_k']) for i, r in enumerate(effective)]
    return dict(tokens=len(generated), decode_s=decode_s, prefill_s=prefill_s,
        decode_tok_s=len(generated)/decode_s, rounds=len(records), spec_rounds=len(spec),
        ar_steps=len(records)-len(spec),
        accepted_tokens=sum(max(0, r['committed']-1) for r in spec),
        committed_tokens=sum(r['committed'] for r in spec),
        candidate_tokens=sum(r['actual_k'] for r in spec),
        requested_k_counts=dict(Counter(r['requested_k'] for r in spec)),
        selector_calls=selector_calls, selector_wall_ms=selector_wall_ms,
        output_sha256=signature(generated), replay_verified=schedule is not None,
        round_details=round_details,
        profile=profiler.resolve() if profiler else None)


def format_rounds(rounds):
    """accepted(actual K), not committed length or number of selected classes."""
    return ', '.join(f"{r['accepted']}({r['actual_k']})" if r['actual_k'] else 'AR'
                     for r in rounds)


def report_rounds(result, output, print_trace=False):
    fields = ['arm', 'prompt_index', 'prompt_key', 'mode', 'repeat', 'round',
              'requested_k', 'actual_k', 'accepted', 'accepted_raw', 'committed', 'tail']
    exists = output.exists()
    with output.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        identity = {k:result[k] for k in fields[:5]}
        for r in result['round_details']:
            writer.writerow(dict(identity, **r))
    if print_trace:
        print(f"ROUNDS [{result['arm']}] prompt={result['prompt_index']} "
              f"{result['mode']} accepted(actual_K):\n"
              f"  {format_rounds(result['round_details'])}", flush=True)
        hist = result['requested_k_counts']
        print('  requested K counts:', json.dumps(hist), flush=True)


def summarize_simple(summary, output):
    baseline = next(r['tok_s'] for r in summary if r['arm']=='fixed' and r['mode']=='fixed_16')
    rows = []
    print('\nGSM8K: each draft compared with its own fixed K16')
    print('draft  mode              tok/s   accepted   committed   rounds/1k  gain vs own K16')
    for arm in ('fixed', 'mixed'):
        own = next(r['tok_s'] for r in summary if r['arm']==arm and r['mode']=='fixed_16')
        for mode in ('fixed_16', 'oracle_accept'):
            r = next(r for r in summary if r['arm']==arm and r['mode']==mode)
            row = dict(arm=arm, mode=mode, tok_s=r['tok_s'], accepted=r['accepted'],
                committed=r['committed'], spec_rounds_per_1k_tokens=r['spec_rounds_per_1k_tokens'],
                gain_vs_own_k16_pct=100*(r['tok_s']/own-1),
                gain_vs_fixed_train_k16_pct=100*(r['tok_s']/baseline-1))
            rows.append(row)
            print(f"{arm:5s}  {mode:16s} {r['tok_s']:8.2f} {r['accepted']:10.3f} "
                  f"{r['committed']:11.3f} {r['spec_rounds_per_1k_tokens']:11.1f} "
                  f"{row['gain_vs_own_k16_pct']:+.2f}%")
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_round_distribution(records, output):
    groups = defaultdict(list)
    for r in records:
        if r['repeat'] == 0:  # No triple-counting repeated identical trajectories.
            for detail in r['round_details']:
                groups[r['arm'], r['mode'], detail['actual_k']].append(detail)
    rows = []
    for (arm, mode, k), rounds in sorted(groups.items()):
        total = sum(len(v) for (a, m, _), v in groups.items() if a == arm and m == mode)
        rows.append(dict(arm=arm, mode=mode, actual_k=k, rounds=len(rounds),
            fraction=len(rounds)/total, mean_accepted=statistics.mean(r['accepted'] for r in rounds),
            accepted_histogram=json.dumps(dict(Counter(r['accepted'] for r in rounds)))))
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, path):
    groups = defaultdict(list)
    for r in records:
        groups[r['arm'], r['mode']].append(r)
    summary = []
    for (arm, mode), rows in groups.items():
        tokens = sum(r['tokens'] for r in rows)
        seconds = sum(r['decode_s'] for r in rows)
        rounds = sum(r['spec_rounds'] for r in rows)
        calls = sum(r['selector_calls'] for r in rows)
        summary.append(dict(arm=arm, mode=mode,
            prompts=len({r['prompt_key'] for r in rows}), measurements=len(rows),
            tokens=tokens, decode_s=seconds, tok_s=tokens/seconds,
            accepted=sum(r['accepted_tokens'] for r in rows)/max(rounds, 1),
            committed=sum(r['committed_tokens'] for r in rows)/max(rounds, 1),
            spec_rounds_per_1k_tokens=rounds*1000/max(tokens, 1),
            candidate_tokens_per_output=sum(r['candidate_tokens'] for r in rows)/max(tokens, 1),
            selector_wall_ms_per_call=sum(r['selector_wall_ms'] for r in rows)/max(calls, 1)))
    baseline = next(r['tok_s'] for r in summary if r['arm']=='fixed' and r['mode']=='fixed_16')
    for arm in ('fixed', 'mixed'):
        own = [r for r in summary if r['arm']==arm]
        best = max((r for r in own if r['mode'].startswith('fixed_')), key=lambda r:r['tok_s'])
        for r in own:
            r.update(vs_fixed_train_fixed16=r['tok_s']/baseline,
                     vs_own_best_fixed=r['tok_s']/best['tok_s'], own_best_fixed=best['mode'])
            print(f"{arm:5s} {r['mode']:28s} {r['tok_s']:8.2f} tok/s  "
                  f"accept={r['accepted']:.3f} rounds/1k={r['spec_rounds_per_1k_tokens']:.1f} "
                  f"vs baseline={r['vs_fixed_train_fixed16']:.3f}x "
                  f"vs own best={r['vs_own_best_fixed']:.3f}x")
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    return summary


def summarize_profiles(path, output):
    groups = defaultdict(list)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        r = json.loads(line)
        groups[r['arm'], r['mode']].append(r)
    results = []
    for (arm, mode), rows in groups.items():
        tokens = sum(r['tokens'] for r in rows)
        rounds = sum(r['spec_rounds'] for r in rows)
        row = dict(arm=arm, mode=mode, tokens=tokens,
            spec_rounds_per_1k_tokens=rounds*1000/max(tokens, 1))
        for stage in ('draft', 'verify', 'selector', 'advance', 'ar'):
            total = sum(r['profile'].get(stage+'_gpu_ms', 0.) for r in rows)
            row[stage+'_gpu_ms_per_1k_tokens'] = total*1000/max(tokens, 1)
            row[stage+'_gpu_ms_per_spec_round'] = total/max(rounds, 1)
        results.append(row)
    with output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('target', 'fixed-draft', 'mixed-draft', 'data', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-prompt', type=int, default=512)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--search-repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-profile', action='store_true')
    p.add_argument('--simple', action='store_true', help='Only own fixed K16 and acceptance oracle; no selector')
    p.add_argument('--print-rounds', action='store_true', help='Print accepted(actual K) after first timed repeat')
    args = p.parse_args()
    if min(args.samples, args.max_new_tokens, args.max_prompt, args.repeats, args.search_repeats)<1:
        p.error('All limits must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists():
        raise ValueError('Use a fresh output directory; existing run is preserved')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model.dflash import DFlashDraftModel
    from adaptive_length.pipeline import read_prompts
    from adaptive_length.runtime import RoundRuntime, clock
    from adaptive_length.two_stage.cli import model_identity
    if not torch.cuda.is_available():
        raise RuntimeError('Benchmark requires CUDA; pure logic tests run on CPU')
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    selected, seen = [], set()
    for key, prompt in read_prompts(args.data):
        if key in seen:
            continue
        seen.add(key)
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
        if ids.shape[1] <= args.max_prompt:
            selected.append((key, ids.cuda()))
        if len(selected) == args.samples:
            break
    if len(selected) != args.samples:
        raise ValueError(f'Only {len(selected)} usable unique prompts; requested {args.samples}')
    metadata = dict(vars(args), k_values=KS, prompt_keys=[k for k, _ in selected],
        target_fingerprint=model_identity(args.target),
        draft_fingerprints={arm:model_identity(path) for arm,path in
                            [('fixed',args.fixed_draft),('mixed',args.mixed_draft)]},
        oracle_scope='greedy local one-round; speed uses synchronized draft+verify median',
        selector_scope='native frozen layer0 anchor-only + fp32 LayerNorm/MLP128/8 + argmax readback; '
                       'random head outputs discarded, oracle supplies perfect class',
        timing_scope='primary wall decode includes selector/advance/stop; excludes discovery and prefill',
        profile_scope='separate instrumented pass, excluded from primary tok/s',
        caveat='outputs and token counts may differ across policies; not a lossless/quality or global-optimum claim')
    json_write(out/'metadata.json', metadata)
    records = []
    with torch.inference_mode():
        for arm, draft_path in [('fixed', args.fixed_draft), ('mixed', args.mixed_draft)]:
            draft = DFlashDraftModel.from_pretrained(draft_path, torch_dtype=torch.bfloat16,
                attn_implementation='sdpa').cuda().eval().requires_grad_(False)
            torch.manual_seed(args.seed)
            head = torch.nn.Sequential(torch.nn.LayerNorm(draft.config.hidden_size),
                torch.nn.Linear(draft.config.hidden_size, 128), torch.nn.SiLU(),
                torch.nn.Linear(128, 8)).cuda().eval().requires_grad_(False) if not args.simple else None
            for index, (key, ids) in enumerate(selected, 1):
                print(f'[{arm}] prompt {index}/{len(selected)} {key[:12]}: warming shapes', flush=True)
                warm = RoundRuntime(target, draft, ids)
                boundary = warm.draft_cache.get_seq_length()
                for k in KS:
                    block = warm.propose(k)
                    warm.verify(block)
                    warm.draft_cache.crop(boundary)
                for mode in (() if args.simple else ('full_kv', 'window_64')):
                    native_predictor_probe(warm, mode, head, torch)
                del warm
                schedules = {}
                for objective in (('accept',) if args.simple else ('accept', 'speed')):
                    print(f'[{arm}] prompt {index}: discover {objective}', flush=True)
                    schedules[objective] = discover(target, draft, ids, objective, args,
                        tokenizer.eos_token_id, RoundRuntime, clock, torch)
                    json_write(out/'schedules'/arm/f'{key}_{objective}.json', schedules[objective])
                jobs = [(f'fixed_{k}', None, k, None) for k in ([16] if args.simple else KS)]
                jobs += [(f'oracle_{obj}', schedules[obj], None, None) for obj in schedules]
                if not args.simple:
                    jobs += [(f'perfect_accept_{mode}', schedules['accept'], None, mode)
                             for mode in ('full_kv', 'window_64')]
                for repeat in range(args.repeats):
                    order = jobs.copy()
                    random.Random(args.seed+repeat+index*100).shuffle(order)
                    for mode, schedule, k, selector in order:
                        r = replay(target, draft, ids, schedule, k, selector, head, args,
                            tokenizer.eos_token_id, RoundRuntime, clock, torch)
                        r.update(arm=arm, mode=mode, repeat=repeat, prompt_key=key, prompt_index=index)
                        records.append(r)
                        with (out/'results.jsonl').open('a') as f:
                            f.write(json.dumps(r)+'\n')
                        report_rounds(r, out/'rounds.csv', args.print_rounds and repeat == 0)
                        print(f"[{arm}] {index}/{len(selected)} repeat={repeat+1}/{args.repeats} "
                              f"{mode}: {r['decode_tok_s']:.2f} tok/s", flush=True)
                if not args.no_profile:
                    for mode, schedule, k, selector in jobs:
                        r = replay(target, draft, ids, schedule, k, selector, head, args,
                            tokenizer.eos_token_id, RoundRuntime, clock, torch, profile=True)
                        r.update(arm=arm, mode=mode, prompt_key=key)
                        with (out/'profiles.jsonl').open('a') as f:
                            f.write(json.dumps(r)+'\n')
            del head, draft
            torch.cuda.empty_cache()
    summary = summarize(records, out/'comparison.csv')
    summarize_profiles(out/'profiles.jsonl', out/'profile_comparison.csv')
    if args.simple:
        summarize_simple(summary, out/'simple_comparison.csv')
        summarize_round_distribution(records, out/'round_distribution.csv')
        print(f'Results: {out}/simple_comparison.csv; rounds: {out}/rounds.csv', flush=True)
        return
    # Token-normalized slack is the maximum extra ms/call under additive-cost assumptions.
    budgets = []
    for arm in ('fixed', 'mixed'):
        own = [r for r in summary if r['arm']==arm]
        oracle = next(r for r in own if r['mode']=='oracle_accept')
        calls = sum(r['selector_calls'] for r in records
                    if r['arm']==arm and r['mode']=='perfect_accept_full_kv')
        for baseline in [r for r in own if r['mode'] in ('fixed_16', oracle['own_best_fixed'])]:
            slack_ms = 1000*oracle['tokens']*(1/baseline['tok_s']-1/oracle['tok_s'])
            budgets.append(dict(arm=arm, baseline=baseline['mode'],
                max_extra_ms_per_selector_call=slack_ms/max(calls, 1),
                assumption='token-normalized additive cost; negative means no positive budget'))
    json_write(out/'selector_budget.json', budgets)
    print('Selector budgets:', json.dumps(budgets), flush=True)
    print(f'Results: {out}/comparison.csv', flush=True)


if __name__ == '__main__':
    main()
