"""Interleaved fixed/mixed K sweep, plus common-AR-state acceptance probes.

Both frozen drafts are resident with one target. Timed methods are randomized
across both drafts and Ks per prompt/repeat. Probes are untimed, use identical
AR contexts/anchors across drafts, and never contribute to decode throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from oracle_upper_bound import KS, json_write, replay, report_rounds, signature


def collect_paired(target, drafts, ids, args, eos, Runtime, paired_measure, torch):
    # Keep one untouched target-AR walker; probes reconstruct fresh caches so
    # an old draft cache cannot fall behind context_start after many AR steps.
    walker = Runtime(target, drafts['fixed'], ids)
    result = []
    prompt_len = ids.shape[1]
    next_position = 0
    begin = time.monotonic()
    while True:
        first = walker
        generated = first.prefix[0, prompt_len:]
        count = generated.numel()
        if count >= args.max_new_tokens or (eos is not None and bool((generated == eos).any())):
            break
        if count >= next_position and count+33 <= args.max_new_tokens:
            order = ['fixed', 'mixed']
            random.Random(args.seed+count).shuffle(order)
            row = dict(generated_position=count, context_length=first.prefix.shape[1],
                prefix_sha256=signature(first.prefix[0].tolist()), anchor_id=int(first.anchor.item()))
            for arm in order:
                probe = Runtime(target, drafts[arm], first.prefix)
                # Identical reference anchor from the untouched AR trajectory.
                probe.anchor = first.anchor.clone()
                row[arm] = paired_measure(probe, args.probe_repeats,
                    random.Random(args.seed+count), eos=eos)
                del probe
            result.append(row)
            print(f'paired state {len(result)}/{args.probe_states} generated={count} '
                  f'elapsed={time.monotonic()-begin:.1f}s', flush=True)
            if len(result) >= args.probe_states:
                break
            next_position = count+args.probe_stride
        walker.ar_step()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('target','fixed-draft','mixed-draft','data','output','dataset'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-prompt', type=int, default=512)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--probe-states', type=int, default=4)
    p.add_argument('--probe-stride', type=int, default=64)
    p.add_argument('--probe-repeats', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.samples,args.max_prompt,args.max_new_tokens,args.repeats,args.probe_stride,args.probe_repeats)<1 or args.probe_states<0:
        p.error('Positive limits required; probe-states may be zero to disable probes')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from model.dflash import DFlashDraftModel
    from adaptive_length.pipeline import read_prompts
    from adaptive_length.runtime import RoundRuntime, clock
    from adaptive_length.two_stage.cli import model_identity, paired_measure
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists():
        raise ValueError('Use a fresh output directory')
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    prompts, seen = [], set()
    for key, prompt in read_prompts(args.data):
        if key in seen:
            continue
        seen.add(key)
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
        if ids.shape[1] > args.max_prompt:
            raise ValueError(f'Prompt {key} exceeds max-prompt; raise cap rather than changing chosen first prompts')
        prompts.append((key,prompt,ids.cuda()))
        if len(prompts) == args.samples:
            break
    if len(prompts) != args.samples:
        raise ValueError(f'Only {len(prompts)} unique prompts, expected {args.samples}')
    metadata = dict(vars(args), k_values=KS, prompt_keys=[key for key,_,_ in prompts],
        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        model_fingerprints={name:model_identity(path) for name,path in
            [('target',args.target),('fixed',args.fixed_draft),('mixed',args.mixed_draft)]},
        scope='B1 greedy, thinking off, EOS or token cap; no predictor/oracle; method order interleaved',
        probes='same confirmed target-AR prefixes and anchors, outside throughput; no output-quality/lossless claim')
    json_write(out/'metadata.json', metadata)
    (out/'selected_prompts.jsonl').write_text(''.join(json.dumps(dict(prompt_key=k,prompt=p),ensure_ascii=False)+'\n' for k,p,_ in prompts))
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    drafts = {arm:DFlashDraftModel.from_pretrained(path, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False)
        for arm,path in [('fixed',args.fixed_draft),('mixed',args.mixed_draft)]}
    with torch.inference_mode():
        for index,(key,prompt,ids) in enumerate(prompts,1):
            print(f'[{args.dataset}] prompt {index}/{len(prompts)} warmup', flush=True)
            for arm, draft in drafts.items():
                state = RoundRuntime(target,draft,ids)
                boundary = state.draft_cache.get_seq_length()
                for k in KS:
                    block = state.propose(k)
                    state.verify(block)
                    state.draft_cache.crop(boundary)
                del state
            for repeat in range(args.repeats):
                jobs = [(arm,k) for arm in drafts for k in KS]
                random.Random(args.seed+index*100+repeat).shuffle(jobs)
                for arm,k in jobs:
                    r = replay(target,drafts[arm],ids,None,k,None,None,args,
                        tokenizer.eos_token_id,RoundRuntime,clock,torch)
                    r.update(dataset=args.dataset,arm=arm,mode=f'fixed_{k}',k=k,
                             prompt_index=index,prompt_key=key,repeat=repeat)
                    with (out/'results.jsonl').open('a') as f:
                        f.write(json.dumps(r)+'\n')
                    report_rounds(r,out/'rounds.csv')
                    print(f"[{args.dataset}] {index}/{len(prompts)} repeat={repeat+1}/{args.repeats} "
                          f"{arm} K{k} {r['decode_tok_s']:.2f} tok/s", flush=True)
            if args.probe_states:
                probes = collect_paired(target,drafts,ids,args,tokenizer.eos_token_id,
                    RoundRuntime,paired_measure,torch)
                with (out/'paired_states.jsonl').open('a') as f:
                    for r in probes:
                        f.write(json.dumps(dict(dataset=args.dataset,prompt_index=index,prompt_key=key,**r))+'\n')
    json_write(out/'complete.json',dict(prompts=len(prompts),repeats=args.repeats))
    print(f'Complete: {out}',flush=True)


if __name__ == '__main__':
    main()
