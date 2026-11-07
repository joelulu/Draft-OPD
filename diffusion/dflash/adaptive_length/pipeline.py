"""Qwen3/DFlash adaptive-length experiments. Run with PYTHONPATH=diffusion/dflash python -m adaptive_length.pipeline."""
import argparse
import csv
import hashlib
import json
import os
import random
from contextlib import nullcontext
from collections import Counter
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.dflash import DFlashDraftModel
from .head import LengthHead, select_lengths, validate_lengths
from .runtime import RoundRuntime, clock
from .evaluation import actual_length, fixed_lengths, StageTimer
from .training import width_plan


def read_prompts(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt", row.get("messages", row.get("question")))
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"Invalid prompt at {path}:{index + 1}")
            # Split by content, not row number, so duplicate prompts never cross splits.
            key = hashlib.sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            rows.append((key, prompt))
    if not rows:
        raise ValueError(f"No prompts in {path}")
    return rows


def split_rows(rows, split):
    return [r for r in rows if split == "all" or ((int(r[0][:8], 16) % 5 == 0) == (split == "val"))]


def setup(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("Target/draft experiments require CUDA. Head training supports CPU.")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").to(device).eval()
    target.requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(args.draft, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(device).eval()
    if target.config.hidden_size != draft.config.hidden_size:
        raise ValueError("This entrypoint requires matching Qwen3 target/draft hidden dimensions.")
    rows = split_rows(read_prompts(args.data), args.split)
    if args.command == "eval" and args.head:
        checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
        used = set(checkpoint["used_prompt_keys"])
        rows = [r for r in rows if r[0] not in used]
    usable = []
    skipped = 0
    for key, prompt in rows:
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=args.enable_thinking)
        if len(tokenizer.encode(text, add_special_tokens=False)) <= args.max_prompt:
            usable.append((key, prompt))
        else:
            skipped += 1
        if args.max_samples and len(usable) >= args.max_samples:
            break
    rows = usable
    print(json.dumps(dict(rank=rank, usable_prompts=len(rows), skipped_long_prompts=skipped)), flush=True)
    if args.max_samples:
        rows = rows[:args.max_samples]
    rows = rows[rank::world]
    if not rows:
        raise ValueError("No prompts on this rank; increase max_samples or reduce processes.")
    return target, draft, tokenizer, rows, rank, world


def encode(tokenizer, prompt, args, device):
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=args.enable_thinking)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] > args.max_prompt:
        raise ValueError(f"Prompt has {ids.shape[1]} tokens, above --max-prompt={args.max_prompt}; "
                         "increase the limit rather than silently truncate chat markers.")
    return ids.to(device)


def stopped(runtime, prompt_len, budget, eos):
    generated = runtime.prefix[0, prompt_len:]
    return generated.numel() >= budget or (eos is not None and bool((generated == eos).any()))


def trim_generated(ids, budget, eos):
    ids = ids[:budget]
    if eos is not None:
        pos = (ids == eos).nonzero()
        if len(pos):
            ids = ids[:int(pos[0, 0]) + 1]
    return ids


def collect(args):
    target, draft, tokenizer, rows, rank, world = setup(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"labels_rank{rank:03d}.pt").exists():
        raise FileExistsError("Use a fresh label directory to avoid mixing collection runs.")
    features, anchors, labels, timings, keys = [], [], [], [], []
    metadata = dict(target=args.target, draft=args.draft, lengths=list(args.lengths),
                    seed=args.seed, split=args.split, max_new_tokens=args.max_new_tokens,
                    timing_repeats=args.timing_repeats,
                    layer_ids=draft.target_layer_ids, feature_schema="fused-pivot-before-anchor+anchor-embedding-v1")
    generator = random.Random(args.seed + rank)
    def flush():
        if features:
            torch.save(dict(pivot=torch.cat(features), anchor=torch.cat(anchors),
                            accepted=torch.tensor(labels), costs_ms=torch.tensor(timings),
                            prompt_keys=keys, metadata=metadata), output / f"labels_rank{rank:03d}.pt")
    with torch.inference_mode():
        for key, prompt in rows:
            ids = encode(tokenizer, prompt, args, target.device)
            state = RoundRuntime(target, draft, ids)
            for round_id in range(args.max_rounds):
                if stopped(state, ids.shape[1], args.max_new_tokens, tokenizer.eos_token_id):
                    break
                pivot, anchor = state.features()
                base_cached = state.draft_cache.get_seq_length()
                results, costs = {}, {}
                order = list(args.lengths)
                generator.shuffle(order)  # reduce systematic first-action timing bias
                for k in order:
                    if round_id == 0:
                        state.draft_cache.crop(base_cached)
                        state.verify(state.propose(k))
                    measurements = []
                    for _ in range(args.timing_repeats):
                        state.draft_cache.crop(base_cached)
                        t0 = clock()
                        block = state.propose(k)
                        result = state.verify(block)
                        measurements.append((clock() - t0) * 1000)
                    elapsed = float(torch.tensor(measurements).median())
                    results[k] = (block, result)
                    costs[k] = elapsed
                features.append(pivot.cpu().to(torch.bfloat16))
                anchors.append(anchor.cpu().to(torch.bfloat16))
                labels.append([results[k][1][0] for k in args.lengths])
                timings.append([costs[k] for k in args.lengths])
                keys.append(key)
                # Random behavior action covers states induced by all actual widths.
                chosen = generator.choice(args.lengths)
                state.advance(*results[chosen])
            flush()  # checkpoint completed prompts
            print(json.dumps(dict(rank=rank, prompt=key[:12], anchors=len(keys))), flush=True)
    # Each rank is independent; merge via train-head only after torchrun has exited.


def load_labels(directory):
    paths = sorted(Path(directory).glob("labels_rank*.pt"))
    if not paths:
        raise ValueError(f"No label shards in {directory}")
    shards = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    meta = shards[0]["metadata"]
    for s in shards:
        if s["metadata"] != meta:
            raise ValueError("Label shard metadata mismatch; do not mix drafts, lengths or splits.")
    return dict(pivot=torch.cat([s["pivot"] for s in shards]).float(),
                anchor=torch.cat([s["anchor"] for s in shards]).float(),
                accepted=torch.cat([s["accepted"] for s in shards]).float(),
                costs_ms=torch.cat([s["costs_ms"] for s in shards]),
                prompt_keys=sum([s["prompt_keys"] for s in shards], []), metadata=meta)


def train_head(args):
    torch.manual_seed(args.seed)
    data = load_labels(args.labels)
    keys = data["prompt_keys"]
    train = torch.tensor([int(k[:8], 16) % 5 != 0 for k in keys])
    if not train.any() or train.all():
        raise ValueError("Need both training and held-out prompt groups. Collect with --split all and more prompts.")
    device = torch.device(args.device)
    lengths = data["metadata"]["lengths"]
    head = LengthHead(data["pivot"].shape[-1], lengths, args.projection, args.features).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    # Cost calibration uses training prompts only. Never derive cost policy from held-out labels.
    costs = data["costs_ms"][train].median(0).values.float()
    if args.costs:
        costs = torch.tensor(json.loads(Path(args.costs).read_text())["costs_ms"])
    select_lengths(torch.zeros(len(lengths)), costs, lengths)  # validate cost table
    pivot, anchor, accepted = [data[k].to(device) for k in ("pivot", "anchor", "accepted")]
    train_idx = train.nonzero().flatten()
    val_idx = (~train).nonzero().flatten().to(device)
    best = float("-inf")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        head.train()
        for idx in train_idx[torch.randperm(len(train_idx))].split(args.batch_size):
            idx = idx.to(device)
            opt.zero_grad(set_to_none=True)
            loss = head.loss(pivot[idx], anchor[idx], accepted[idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(pivot[val_idx], anchor[val_idx])
            action = select_lengths(pred, costs, lengths)
            # Aggregate committed tokens / aggregate round cost, not mean per-round ratios.
            observed = accepted[val_idx]
            selected = observed.gather(1, action[:, None]).squeeze(1) + 1
            selected_cost = costs.to(device)[action]
            efficiency = float(selected.sum() / selected_cost.sum())
            fixed_scores = (accepted[train.to(device)].mean(0).cpu() + 1) / costs
            fixed = int(fixed_scores.argmax())
            fixed_eff = float((observed[:, fixed] + 1).mean().cpu() / costs[fixed])
            report = dict(epoch=epoch + 1, val_mae=float((pred-observed).abs().mean()),
                          val_estimated_tok_per_ms=efficiency, best_train_fixed_k=lengths[fixed],
                          val_fixed_estimated_tok_per_ms=fixed_eff,
                          train_anchors=int(train.sum()), val_anchors=int((~train).sum()))
        print(json.dumps(report), flush=True)
        if efficiency > best:
            best = efficiency
            torch.save(dict(settings=head.settings, state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()},
                            costs_ms=costs, metadata=data["metadata"], validation=report,
                            used_prompt_keys=sorted(set(keys))), args.output)
            head.to(device)


def load_head(path, device, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    meta = checkpoint["metadata"]
    if meta["target"] != args.target or meta["draft"] != args.draft:
        raise ValueError("Head target/draft paths differ from collection metadata; recollect for new weights.")
    head = LengthHead(**checkpoint["settings"]).to(device).eval()
    head.load_state_dict(checkpoint["state_dict"])
    if tuple(args.lengths) != head.lengths:
        raise ValueError("Evaluation lengths must match the head's action set.")
    return head, checkpoint["costs_ms"]


def summarize(records, modes):
    summary = []
    for mode in modes:
        group = [r for r in records if r["mode"] == mode]
        tokens, seconds = sum(r["tokens"] for r in group), sum(r["decode_s"] for r in group)
        spec = sum(r["spec_rounds"] for r in group)
        core = sum(r["non_tail_spec_rounds"] for r in group)
        row = dict(mode=mode, tokens=tokens, decode_s=seconds, decode_tok_s=tokens/seconds,
                   matches=sum(r["match"] for r in group), samples=len(group),
                   spec_rounds=spec, ar_steps=sum(r["ar_steps"] for r in group),
                   tail_spec_rounds=sum(r["tail_spec_rounds"] for r in group),
                   mean_accepted=sum(r["accepted_tokens"] for r in group)/spec if spec else None,
                   mean_committed=sum(r["spec_committed_tokens"] for r in group)/spec if spec else None,
                   non_tail_spec_rounds=core,
                   non_tail_mean_accepted=sum(r["non_tail_accepted_tokens"] for r in group)/core if core else None,
                   non_tail_mean_committed=sum(r["non_tail_committed_tokens"] for r in group)/core if core else None,
                   profile_rounds=group[0]["profile_rounds"],
                   selector_wall_ms=sum(r.get("selector_wall_ms", 0.) for r in group),
                   selector_calls=sum(r.get("selector_calls", 0) for r in group),
                   selector_time_fraction=sum(r.get("selector_wall_ms", 0.) for r in group)/(1000*seconds),
                   requested_k_histogram=json.dumps(dict(Counter(k for r in group for k in r["requested_actions"] if k))))
        for name in StageTimer.names:
            row[name] = sum(r[name] for r in group) if group[0]["profile_rounds"] else None
        summary.append(row)
    return summary


def evaluate(args):
    target, draft, tokenizer, rows, rank, world = setup(args)
    head, costs = load_head(args.head, target.device, args) if args.head else (None, None)
    classifiers = {}
    if getattr(args, "classifier", None):
        if args.head or args.enable_thinking:
            raise ValueError("KV classifiers use no-thinking collection and cannot be combined with the old head.")
        from .two_stage.classifier import build_classifier
        from .two_stage.cli import model_identity
        from .two_stage import K_VALUES
        if tuple(args.lengths) != K_VALUES:
            raise ValueError("KV classifier evaluation requires lengths 4 8 12 16 20 24 28 32.")
        target_id, draft_id = model_identity(args.target), model_identity(args.draft)
        used = set()
        for path in args.classifier:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint['metadata']['target'] != target_id or checkpoint['metadata']['draft'] != draft_id:
                raise ValueError("Classifier frozen-model fingerprints differ from evaluation weights.")
            classifier = build_classifier(checkpoint['settings']).to(target.device).eval()
            classifier.load_state_dict(checkpoint['state_dict'])
            name = f"classifier_{Path(path).stem}"
            if name in classifiers:
                raise ValueError("Duplicate classifier names")
            classifiers[name] = classifier
            used.update(checkpoint['used_prompt_keys'])
        if any(key in used for key, _ in rows):
            raise ValueError("Evaluation prompts overlap classifier training/validation; use labeled/test_prompts.jsonl.")
    baseline_lengths = getattr(args, "baseline_lengths", None) or args.lengths
    widths = (list(baseline_lengths) if getattr(args, "exact_lengths", False) else
              fixed_lengths(baseline_lengths, draft.block_size, args.fixed_lengths or []))
    if not widths or any(k < 1 or k > 32 for k in widths):
        raise ValueError("Baseline lengths must be in [1,32]")
    modes = ["ar"] + [f"fixed_{k}" for k in widths] + (["adaptive"] if head else []) + list(classifiers)
    records = []
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"rounds_rank{rank:03d}.jsonl").write_text("")
    metadata = dict(schema=2, native_block_size=int(draft.block_size), fixed_lengths=widths,
                    head_lengths=list(head.lengths) if head else None,
                    tail_policy="min(requested_K, remaining-1); AR only with one token left",
                    profile_rounds=args.profile_rounds, strict_match=args.strict_match,
                    seed=args.seed, max_new_tokens=args.max_new_tokens,
                    classifiers={name: model.settings for name, model in classifiers.items()})
    (outdir / f"metadata_rank{rank:03d}.json").write_text(json.dumps(metadata, indent=2))
    evaluation_start = time.monotonic()
    with torch.inference_mode():
        for key, prompt in rows:
            ids = encode(tokenizer, prompt, args, target.device)
            if args.warmup:
                warm = RoundRuntime(target, draft, ids)
                warm_lengths = sorted(set(widths) | (set(args.lengths) if classifiers else set()))
                for k in warm_lengths:
                    warm.propose(k)
                for classifier in classifiers.values():
                    classifier.choose(warm)
                del warm
            reference = None
            for mode in modes:
                t0 = clock()
                state = RoundRuntime(target, draft, ids)
                prefill_s = clock() - t0
                round_records = []
                selector_wall_ms, selector_calls = 0., 0
                timer = StageTimer(args.profile_rounds, target.device)
                def measure(name):
                    return timer.measure(name) if timer.enabled else nullcontext()
                t0 = clock()
                while not stopped(state, ids.shape[1], args.max_new_tokens, tokenizer.eos_token_id):
                    remaining = args.max_new_tokens - (state.prefix.shape[1] - ids.shape[1])
                    requested = 0
                    if mode != "ar" and remaining > 1:
                        if mode in classifiers:
                            selector_start = time.perf_counter()
                            with measure("head_ms"):
                                requested = classifiers[mode].choose(state)
                            selector_wall_ms += (time.perf_counter()-selector_start)*1000
                            selector_calls += 1
                        elif mode == "adaptive":
                            with measure("head_ms"):
                                pivot, anchor = state.features()
                                # Select the learned nominal action, then apply exactly the
                                # same tail clipping as every fixed policy. No extrapolated
                                # acceptance/cost labels for unseen tail widths are invented.
                                choice = select_lengths(head(pivot, anchor), costs, head.lengths)
                                requested = head.lengths[int(choice.item())]
                        else:
                            requested = int(mode.split("_")[1])
                    k = actual_length(requested, remaining) if requested else 0
                    if k:
                        with measure("draft_ms"):
                            block = state.propose(k)
                        with measure("verify_ms"):
                            result = state.verify(block, keep_cache=True)
                            committed = state.advance(block, result, cache_kept=True)
                        accepted = result[0]
                    else:
                        with measure("ar_ms"):
                            committed = state.ar_step()
                        accepted = 0
                    round_records.append(dict(requested_k=requested, actual_k=k,
                        remaining=remaining, tail=(0 < k < requested),
                        accepted_raw=accepted, accepted=accepted, committed=committed))
                decode_s = clock() - t0  # final CUDA synchronization resolves event timestamps
                generated = trim_generated(state.prefix[0, ids.shape[1]:], args.max_new_tokens,
                                            tokenizer.eos_token_id).cpu()
                # Verification can commit beyond EOS inside a block. Count only emitted
                # tokens in acceptance summaries; keep raw acceptance for diagnostics.
                excess = state.prefix.shape[1] - ids.shape[1] - len(generated)
                if excess:
                    last = round_records[-1]
                    last["committed"] -= excess
                    last["accepted"] = max(0, last["committed"] - 1)
                if mode == "ar":
                    reference = generated
                match = torch.equal(generated, reference)
                spec = [r for r in round_records if r["actual_k"]]
                core = [r for r in spec if not r["tail"]]
                rec = dict(prompt_key=key, mode=mode, tokens=len(generated), rounds=len(round_records),
                           prefill_s=prefill_s, decode_s=decode_s, decode_tok_s=len(generated)/decode_s,
                           match=match, actions=[r["actual_k"] for r in round_records],
                           requested_actions=[r["requested_k"] for r in round_records],
                           spec_rounds=len(spec), ar_steps=len(round_records)-len(spec),
                           tail_spec_rounds=len(spec)-len(core),
                           accepted_tokens=sum(r["accepted"] for r in spec),
                           spec_committed_tokens=sum(r["committed"] for r in spec),
                           non_tail_spec_rounds=len(core),
                           non_tail_accepted_tokens=sum(r["accepted"] for r in core),
                           non_tail_committed_tokens=sum(r["committed"] for r in core),
                           profile_rounds=args.profile_rounds, selector_wall_ms=selector_wall_ms,
                           selector_calls=selector_calls, **timer.resolve())
                rec["mean_accepted"] = rec["accepted_tokens"]/len(spec) if spec else None
                rec["mean_committed"] = rec["spec_committed_tokens"]/len(spec) if spec else None
                records.append(rec)
                if getattr(args, "progress_json", None):
                    elapsed = time.monotonic()-evaluation_start
                    total = len(rows)*len(modes)
                    progress = dict(phase="evaluation", completed=len(records), total=total,
                        elapsed_s=elapsed, eta_s=elapsed*(total-len(records))/len(records),
                        mode=mode, updated_at=time.time())
                    path = Path(args.progress_json)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name+".tmp")
                    temporary.write_text(json.dumps(progress, indent=2))
                    temporary.replace(path)
                print(json.dumps(rec), flush=True)
                # Save each completed method so a later interruption preserves results.
                with (outdir / f"rounds_rank{rank:03d}.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(prompt_key=key, mode=mode, rounds=round_records))+"\n")
                (outdir / f"results_rank{rank:03d}.jsonl").write_text(
                    "".join(json.dumps(r)+"\n" for r in records))
                if not match and args.strict_match:
                    raise RuntimeError(f"Greedy token mismatch: {key}, {mode}.")
    summary = summarize(records, modes)
    with (outdir / f"summary_rank{rank:03d}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def opd_loss(draft_logits, target_logits, accepted):
    """Forward KL on accepted prefix, reverse KL on rejected suffix, token mean.

    Target distributions condition on actual proposed tokens (including rejected
    tokens), matching the draft-induced-state principle of Draft-OPD.
    """
    student = F.log_softmax(draft_logits.float(), dim=-1)
    teacher = F.log_softmax(target_logits.detach().float(), dim=-1)
    forward = F.kl_div(student, teacher, log_target=True, reduction="none").sum(-1)
    reverse = F.kl_div(teacher, student, log_target=True, reduction="none").sum(-1)
    prefix = torch.arange(student.shape[1], device=student.device) < accepted
    return torch.where(prefix[None], forward, reverse).mean()


def train_draft(args):
    if args.split != "train":
        raise ValueError("Draft training requires --split train; reserve held-out prompt groups.")
    budget = args.candidate_tokens_per_update
    if args.steps < 1 or args.save_every < 1 or args.log_every < 1 or budget < 0:
        raise ValueError("Training steps/save/log intervals must be positive, budget nonnegative.")
    if budget:
        width_plan(args.lengths, budget, random.Random(args.seed))  # fail before loading weights
    target, draft, tokenizer, rows, rank, world = setup(args)
    # FP32 trainable parameters + AdamW states; frozen target remains BF16.
    # Autocast keeps draft CUDA matmuls BF16 without losing small optimizer updates.
    draft.float()
    model = draft
    if world > 1:
        torch.distributed.init_process_group("nccl")
        model = torch.nn.parallel.DistributedDataParallel(
            draft, device_ids=[target.device.index], broadcast_buffers=False)
    opt = torch.optim.AdamW(draft.parameters(), lr=args.lr)
    generator = random.Random(args.seed + rank)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"train_rank{rank:03d}.jsonl").exists():
        raise FileExistsError("Use a fresh training output directory; optimizer resume is not implemented.")
    metadata = dict(lengths=list(args.lengths), planned_steps=args.steps, seed=args.seed,
        world_size=world, candidate_tokens_per_rank_per_update=budget or None,
        prompt_schedule="one prompt per optimizer update; same rank/order/seed across arms",
        objective="accepted-forward-kl_rejected-suffix-reverse-kl",
        target=args.target, initial_draft=args.draft, learning_rate=args.lr,
        train_data=args.data, split=args.split, max_prompt=args.max_prompt,
        max_new_tokens=args.max_new_tokens, train_parameters_dtype="float32",
        local_prompt_keys=[key for key, _ in rows])
    (output / f"training_rank{rank:03d}.json").write_text(json.dumps(metadata, indent=2))
    seen_tokens, width_counts, start_time = 0, Counter(), time.perf_counter()
    for step in range(args.steps):
        # Prompt assignment is independent of accepted lengths and RNG consumption.
        # Rollout prefixes within this prompt may differ between trained drafters.
        key, prompt = rows[step % len(rows)]
        initial_ids = encode(tokenizer, prompt, args, target.device)
        prefix, prompt_len = initial_ids, initial_ids.shape[1]
        plan = width_plan(args.lengths, budget, generator) if budget else [generator.choice(args.lengths)]
        token_count, loss_value, accepted_sum = sum(plan), 0.0, 0
        opt.zero_grad(set_to_none=True)
        draft.train()
        for micro, k in enumerate(plan):
            # Recompute history: never backprop through previous verification rounds
            # or reuse draft K/V carrying a previous microbatch's freed graph.
            state = RoundRuntime(target, draft, prefix)
            sync = model.no_sync() if world > 1 and micro < len(plan)-1 else nullcontext()
            with sync:
                with torch.autocast(device_type=target.device.type, dtype=torch.bfloat16,
                                    enabled=target.device.type == "cuda"):
                    block, logits = state.propose(k, return_logits=True, forward_model=model)
                result, target_logits = state.verify(block, keep_cache=True, return_logits=True)
                loss = opd_loss(logits, target_logits, result[0]) * (k / token_count)
                loss.backward()
            loss_value += float(loss.detach())
            accepted_sum += result[0]
            width_counts[k] += 1
            with torch.no_grad():
                state.advance(block, result, cache_kept=True)
                prefix = state.prefix.detach()
                if stopped(state, prompt_len, args.max_new_tokens, tokenizer.eos_token_id):
                    prefix = initial_ids
            del state, logits, target_logits, loss, block, result
        grad_norm = float(torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0))
        opt.step()
        draft.eval()
        seen_tokens += token_count
        record = dict(step=step+1, prompt_key=key, rank=rank, world_size=world,
            loss=loss_value, grad_norm=grad_norm, candidate_tokens=token_count,
            cumulative_candidate_tokens_per_rank=seen_tokens,
            microbatches=len(plan), width_counts=dict(Counter(plan)),
            mean_accepted=accepted_sum/len(plan), elapsed_s=time.perf_counter()-start_time)
        with (output / f"train_rank{rank:03d}.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        if rank == 0 and (step % args.log_every == 0 or step + 1 == args.steps):
            print(json.dumps(record), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            if rank == 0:
                path = output / f"step_{step + 1:06d}"
                draft.save_pretrained(path)
                (path / "adaptive_training.json").write_text(json.dumps(dict(metadata,
                    steps=step+1, cumulative_candidate_tokens_per_rank=seen_tokens,
                    global_candidate_tokens=seen_tokens*world if budget else None,
                    rank0_width_counts=dict(width_counts)), indent=2))
            if world > 1:
                torch.distributed.barrier()
    if world > 1:
        torch.distributed.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("collect", "eval", "train-draft"):
        q = sub.add_parser(name)
        q.add_argument("--target", required=True)
        q.add_argument("--draft", required=True)
        q.add_argument("--data", required=True)
        q.add_argument("--output", required=True)
        q.add_argument("--lengths", type=int, nargs="+", default=[4,8,12,16,24,32])
        q.add_argument("--max-samples", type=int, default=128)
        q.add_argument("--max-prompt", type=int, default=512)
        q.add_argument("--max-new-tokens", type=int, default=256)
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--split", choices=["all", "train", "val"], default="all")
        q.add_argument("--enable-thinking", action="store_true")
        if name == "collect":
            q.add_argument("--max-rounds", type=int, default=32)
            q.add_argument("--timing-repeats", type=int, default=2)
        if name == "eval":
            q.add_argument("--head")
            q.add_argument("--baseline-lengths", type=int, nargs="+", help="Fixed baselines independent of classifier actions.")
            q.add_argument("--progress-json", help="Atomic inference progress/ETA file.")
            q.add_argument("--classifier", nargs="+", help="Frozen KV classifiers; evaluated with all fixed-K baselines.")
            q.add_argument("--exact-lengths", action="store_true", help="Do not automatically add the native candidate count.")
            q.add_argument("--strict-match", action="store_true")
            q.add_argument("--fixed-lengths", type=int, nargs="+", help="Additional fixed baselines; head actions stay unchanged.")
            q.add_argument("--profile-rounds", action="store_true", help="Optional stage CUDA events; compare speeds with profiling off.")
            q.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
        if name == "train-draft":
            q.set_defaults(split="train")
            q.add_argument("--steps", type=int, default=1000)
            q.add_argument("--candidate-tokens-per-update", type=int, default=0,
                           help="Exact supervised candidate budget per rank/update; 0 keeps one block per update.")
            q.add_argument("--save-every", type=int, default=250)
            q.add_argument("--log-every", type=int, default=10)
            q.add_argument("--lr", type=float, default=3e-4)
    q = sub.add_parser("train-head")
    q.add_argument("--labels", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--features", choices=["both", "anchor", "pivot"], default="both")
    q.add_argument("--epochs", type=int, default=30)
    q.add_argument("--batch-size", type=int, default=128)
    q.add_argument("--projection", type=int, default=128)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--device", default="cpu")
    q.add_argument("--costs", help="Optional independent B=1 timing table JSON with costs_ms array.")
    q.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if hasattr(args, "lengths"):
        args.lengths = validate_lengths(args.lengths)
        if args.max_new_tokens <= 0 or args.max_prompt <= 0 or args.max_samples < 0:
            p.error("Token limits must be positive; max-samples=0 means all prompts.")
    {"collect": collect, "train-head": train_head, "eval": evaluate, "train-draft": train_draft}[args.command](args)


if __name__ == "__main__":
    main()
