"""collect -> label -> train: actual target KV, frozen target/drafter, dual labels."""
import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model.dflash import DFlashDraftModel
from ..pipeline import read_prompts
from ..runtime import RoundRuntime, clock
from . import K_VALUES
from .classifier import KVClassifier, cache_pairs
from .labels import calibrate, costs_for, make_labels


def model_identity(path):
    path = Path(path).resolve()
    # Fingerprint all local weight bytes, not only a path/config. Called once per command.
    digest = hashlib.sha256()
    files = sorted(p for p in path.iterdir() if p.name == 'config.json' or p.suffix in ('.safetensors', '.bin'))
    if len(files) < 2:
        raise ValueError(f'Expected local HF config and weights: {path}')
    for file in files:
        digest.update(file.name.encode())
        with file.open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def load_models(args, draft=True):
    if not torch.cuda.is_available():
        raise RuntimeError('Collection/training need CUDA; CPU tests do not')
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    drafter = DFlashDraftModel.from_pretrained(args.draft, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False) if draft else None
    return target, drafter


def select_prompts(args):
    rows = read_prompts(args.data)
    seen = set()
    unique = []
    for key, prompt in rows:
        if key not in seen:
            unique.append((key, prompt))
            seen.add(key)
    random.Random(args.seed).shuffle(unique)
    selected = {'train': [], 'val': [], 'test': []}
    external = read_prompts(args.test_data) if args.test_data else []
    external_keys = {key for key, _ in external}
    for key, prompt in unique:
        if key in external_keys:
            continue
        bucket = int(key[:8], 16) % 10
        split = 'test' if bucket == 0 else 'val' if bucket == 1 else 'train'
        if args.test_data and split == 'test':
            continue
        if len(selected[split]) < getattr(args, split+'_prompts'):
            selected[split].append((key, prompt))
    if args.test_data:
        random.Random(args.seed).shuffle(external)
        used = set()
        for key, prompt in external:
            if key not in used and len(selected['test']) < args.test_prompts:
                selected['test'].append((key, prompt))
                used.add(key)
    return selected


@torch.no_grad()
def paired_measure(runtime, repeats, rng, eos=None):
    start = runtime.prefix.shape[1]
    # Warm kernels, then restore the incoming draft-cache boundary. Every K
    # includes the same pending confirmed-context materialization in its timing.
    draft_start = runtime.draft_cache.get_seq_length()
    runtime.propose(K_VALUES[0])
    runtime.draft_cache.crop(draft_start)
    accepted = {k: [] for k in K_VALUES}
    raw_accepted = {k: [] for k in K_VALUES}
    draft_ms, verify_ms = {k: [] for k in K_VALUES}, {k: [] for k in K_VALUES}
    for _ in range(repeats):
        order = list(K_VALUES)
        rng.shuffle(order)
        for k in order:
            t0 = clock()
            block = runtime.propose(k)
            t1 = clock()
            result = runtime.verify(block)
            t2 = clock()
            raw_accepted[k].append(result[0])
            committed = result[0]+1
            if eos is not None:
                hits = (block[0, :committed] == eos).nonzero(as_tuple=True)[0]
                if hits.numel():
                    committed = int(hits[0])+1
            accepted[k].append(committed-1)
            draft_ms[k].append((t1-t0)*1000)
            verify_ms[k].append((t2-t1)*1000)
            runtime.draft_cache.crop(draft_start)
            if runtime.target_cache.get_seq_length() != start or runtime.draft_cache.get_seq_length() != draft_start:
                raise RuntimeError('Paired trials polluted a cache')
    if any(len(set(a)) != 1 for a in accepted.values()):
        raise RuntimeError('Repeated acceptance changed at a fixed state')
    return dict(accepted_lengths=[accepted[k][0] for k in K_VALUES],
                raw_accepted_lengths=[raw_accepted[k][0] for k in K_VALUES],
                committed_lengths=[accepted[k][0]+1 for k in K_VALUES],
                draft_times_ms=[draft_ms[k] for k in K_VALUES],
                verify_times_ms=[verify_ms[k] for k in K_VALUES],
                total_times_ms=[[a+b for a, b in zip(draft_ms[k], verify_ms[k])] for k in K_VALUES])


def collect(args):
    rank, world = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=1, world_size=world, target=model_identity(args.target), draft=model_identity(args.draft),
        target_path=str(Path(args.target).resolve()), draft_path=str(Path(args.draft).resolve()),
        k_values=list(K_VALUES), seed=args.seed, max_new_tokens=args.max_new_tokens,
        max_prompt=args.max_prompt, timing_repeats=args.timing_repeats,
        test_scope='external' if args.test_data else 'prompt-hash holdout for classifier; draft may have seen these prompts',
        collector=dict(train_prompts=args.train_prompts, val_prompts=args.val_prompts, test_prompts=args.test_prompts,
            states_per_prompt=args.states_per_prompt, state_stride=args.state_stride))
    path = out/f'manifest_rank{rank:03d}.json'
    if path.exists() or any(out.glob(f'*.raw.rank{rank:03d}.jsonl')):
        raise ValueError('Use a fresh collection output directory')
    path.write_text(json.dumps(manifest, indent=2))
    target, draft = load_models(args)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    for split, rows in select_prompts(args).items():
        with (out/f'{split}.raw.rank{rank:03d}.jsonl').open('w') as stream:
            usable = states = 0
            for key, prompt in rows[rank::world]:
                text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(target.device)
                if ids.shape[1] > args.max_prompt:
                    continue
                usable += 1
                rng = random.Random(args.seed + int(key[:16], 16))
                with torch.inference_mode():
                    runtime = RoundRuntime(target, draft, ids)
                    sampled, next_position = 0, ids.shape[1]
                    while runtime.prefix.shape[1]-ids.shape[1] < args.max_new_tokens:
                        remaining = args.max_new_tokens-(runtime.prefix.shape[1]-ids.shape[1])
                        if runtime.prefix.shape[1] >= next_position and remaining >= max(K_VALUES)+1 and sampled < args.states_per_prompt:
                            row = dict(state_id=f'{key}:{runtime.prefix.shape[1]}', prompt_key=key, split=split,
                                context_length=runtime.prefix.shape[1], prefix_ids=runtime.prefix[0].tolist(),
                                anchor_id=int(runtime.anchor.item()), k_values=list(K_VALUES), prompt=prompt)
                            row.update(paired_measure(runtime, args.timing_repeats, rng, eos=tokenizer.eos_token_id))
                            stream.write(json.dumps(row)+'\n')
                            stream.flush()
                            states += 1
                            sampled += 1
                            next_position = runtime.prefix.shape[1] + args.state_stride
                        k = min(rng.choice(K_VALUES), remaining-1)
                        if k > 0:
                            runtime.step(k)
                        else:
                            runtime.ar_step()
                        generated = runtime.prefix[0, ids.shape[1]:]
                        if tokenizer.eos_token_id is not None and bool((generated == tokenizer.eos_token_id).any()):
                            break
                        if sampled >= args.states_per_prompt:
                            break
                print(json.dumps(dict(rank=rank, split=split, prompts=usable, states=states)), flush=True)
    (out/f'complete_rank{rank:03d}.json').write_text(json.dumps(dict(rank=rank)))


def label(args):
    root = Path(args.data)
    manifests = [json.loads(p.read_text()) for p in sorted(root.glob('manifest_rank*.json'))]
    if not manifests or any(m != manifests[0] for m in manifests):
        raise ValueError('Collection manifests differ or are missing')
    world = manifests[0].get('world_size', 1)
    if len(manifests) != world or any(not (root/f'complete_rank{rank:03d}.json').is_file() for rank in range(world)):
        raise ValueError('Collection is incomplete: all rank completion markers are required')
    rows = {s: [] for s in ('train', 'val', 'test')}
    seen, prompt_splits = set(), {}
    for split in rows:
        for path in sorted(root.glob(f'{split}.raw.rank*.jsonl')):
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r['state_id'] in seen or r['split'] != split or r['k_values'] != list(K_VALUES):
                    raise ValueError('Duplicate state or inconsistent split/K')
                if prompt_splits.setdefault(r['prompt_key'], split) != split:
                    raise ValueError('Prompt leakage between classifier splits')
                seen.add(r['state_id'])
                rows[split].append(r)
        if not rows[split]:
            raise ValueError(f'No usable {split} states: increase prompts/response budget')
    table = calibrate(rows['train'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Use a fresh labeled directory')
    (out/'costs.json').write_text(json.dumps(table, indent=2))
    (out/'manifest.json').write_text(json.dumps(manifests[0], indent=2))
    for split, records in rows.items():
        unique_prompts = {r['prompt_key']: r['prompt'] for r in records}
        with (out/f'{split}_prompts.jsonl').open('w') as f:
            for key, prompt in unique_prompts.items():
                f.write(json.dumps(dict(prompt_key=key, prompt=prompt), ensure_ascii=False)+'\n')
        with (out/f'{split}.jsonl').open('w') as stream:
            for r in records:
                r.update(make_labels(r['accepted_lengths'], costs_for(table, r['context_length'])))
                stream.write(json.dumps(r)+'\n')
        print(f'{split}: {len(records)} states, {len({r["prompt_key"] for r in records})} prompts')


def iter_states(target, rows, seed=None):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['prompt_key'], []).append(row)
    groups = list(grouped.values())
    if seed is not None:
        random.Random(seed).shuffle(groups)
    for group in groups:
        cache, prefix = DynamicCache(), []
        for row in sorted(group, key=lambda r: r['context_length']):
            ids = row['prefix_ids']
            if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
                raise ValueError('States are not one monotonic confirmed-prefix trajectory')
            with torch.no_grad():
                target(torch.tensor([ids[len(prefix):]], device=target.device), past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
                anchor = target.get_input_embeddings()(torch.tensor([row['anchor_id']], device=target.device))
            prefix = ids
            yield row, cache_pairs(cache), anchor


def train(args):
    if int(os.getenv('WORLD_SIZE', '1')) != 1:
        raise ValueError('Classifier trainer is single-GPU; collection supports multiple GPUs')
    root = Path(args.data)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['target'] != model_identity(args.target) or manifest['draft'] != model_identity(args.draft):
        raise ValueError('Frozen model weights differ from label collection')
    rows = {s: [json.loads(v) for v in (root/f'{s}.jsonl').read_text().splitlines()] for s in ('train', 'val')}
    if {r['prompt_key'] for r in rows['train']} & {r['prompt_key'] for r in rows['val']}:
        raise ValueError('Train/validation prompt overlap')
    target, _ = load_models(args, draft=False)
    config = target.config
    torch.manual_seed(args.seed)
    models = {}
    for context in ('full_kv', 'window_16'):
        for objective in ('accept', 'speed'):
            torch.manual_seed(args.seed)  # Same initialization across label/context ablations.
            model = KVClassifier(anchor_dim=config.hidden_size,
                head_dim=getattr(config, 'head_dim', config.hidden_size//config.num_attention_heads),
                layers=config.num_hidden_layers, kv_heads=config.num_key_value_heads,
                context_mode=context).to(target.device)
            models[f'{context}_{objective}'] = (model, objective)
    optim = {name: torch.optim.AdamW(m.parameters(), lr=args.lr) for name, (m, _) in models.items()}
    best = {name: float('inf') for name in models}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('*.pt')):
        raise ValueError('Use a fresh classifier output directory')
    for epoch in range(args.epochs):
        for m, _ in models.values():
            m.train()
        count = 0
        for row, pairs, anchor in iter_states(target, rows['train'], args.seed+epoch):
            count += 1
            for name, (model, objective) in models.items():
                y = torch.tensor([K_VALUES.index(row[f'label_{objective}_k'])], device=target.device)
                loss = torch.nn.functional.cross_entropy(model(pairs, anchor), y)/args.accumulate
                loss.backward()
            if count % args.accumulate == 0:
                for name, (model, _) in models.items():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim[name].step()
                    optim[name].zero_grad(set_to_none=True)
        if count % args.accumulate:
            correction = args.accumulate/(count % args.accumulate)
            for name, (model, _) in models.items():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim[name].step()
                optim[name].zero_grad(set_to_none=True)
        totals = {name: dict(loss=0., correct=0, speed_ratio=0.) for name in models}
        for m, _ in models.values():
            m.eval()
        with torch.no_grad():
            for row, pairs, anchor in iter_states(target, rows['val']):
                for name, (model, objective) in models.items():
                    logits = model(pairs, anchor)
                    y = K_VALUES.index(row[f'label_{objective}_k'])
                    pred = int(logits.argmax(-1))
                    totals[name]['loss'] += float(torch.nn.functional.cross_entropy(logits, torch.tensor([y], device=target.device)))
                    totals[name]['correct'] += int(pred == y)
                    totals[name]['speed_ratio'] += row['speed_scores'][pred]/max(row['speed_scores'])
        for name, (model, objective) in models.items():
            metrics = {key: value/len(rows['val']) for key, value in totals[name].items()}
            print(json.dumps(dict(epoch=epoch+1, classifier=name, **metrics)), flush=True)
            if metrics['loss'] < best[name]:
                best[name] = metrics['loss']
                torch.save(dict(settings=model.settings, state_dict=model.state_dict(), metadata=manifest,
                    objective=objective, epoch=epoch+1, validation=metrics,
                    used_prompt_keys=sorted({r['prompt_key'] for rs in rows.values() for r in rs})), out/f'{name}.pt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('collect', 'train', 'label'):
        q = sub.add_parser(name)
        q.add_argument('--data', required=True)
        q.add_argument('--output', required=True)
        if name != 'label':
            q.add_argument('--target', required=True)
            q.add_argument('--draft', required=True)
            q.add_argument('--seed', type=int, default=42)
        if name == 'collect':
            q.add_argument('--test-data')
            q.add_argument('--train-prompts', type=int, default=2000)
            q.add_argument('--val-prompts', type=int, default=256)
            q.add_argument('--test-prompts', type=int, default=256)
            q.add_argument('--states-per-prompt', type=int, default=16)
            q.add_argument('--state-stride', type=int, default=16)
            q.add_argument('--max-prompt', type=int, default=512)
            q.add_argument('--max-new-tokens', type=int, default=512)
            q.add_argument('--timing-repeats', type=int, default=3)
        if name == 'train':
            q.add_argument('--epochs', type=int, default=10)
            q.add_argument('--accumulate', type=int, default=16)
            q.add_argument('--lr', type=float, default=1e-4)
    args = p.parse_args()
    if args.command == 'collect' and min(args.train_prompts, args.val_prompts, args.test_prompts,
        args.states_per_prompt, args.state_stride, args.timing_repeats) < 1:
        p.error('Collection limits must be positive')
    if args.command == 'collect' and args.max_new_tokens < max(K_VALUES)+1:
        p.error('Need max-new-tokens >= 33 for paired labels')
    if args.command == 'train' and (args.epochs < 1 or args.accumulate < 1):
        p.error('epochs and accumulate must be positive')
    {'collect': collect, 'label': label, 'train': train}[args.command](args)


if __name__ == '__main__':
    main()
