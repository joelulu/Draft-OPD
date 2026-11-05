"""Run with PYTHONPATH=diffusion/dflash python -m adaptive_length.cli --help."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
from .head import LengthHead, select_lengths, validate_lengths


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def prompts(args, tokenizer):
    with open(args.data, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for index, row in enumerate(rows[args.offset:args.offset + args.samples], args.offset):
        messages = row.get("prompt", row.get("messages", row.get("question")))
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Row {index}: expected prompt/messages list or question string.")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        # Preserve chat template boundaries: skip oversize prompts rather than cutting them.
        if ids.shape[1] > args.max_prompt:
            continue
        prompt_id = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        yield index, prompt_id, ids


def models(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model.dflash import DFlashDraftModel
    if not torch.cuda.is_available():
        raise RuntimeError("Collection and decoding require a CUDA GPU; head training supports CPU.")
    torch.manual_seed(args.seed)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda").eval()
    draft = DFlashDraftModel.from_pretrained(args.draft, torch_dtype=torch.bfloat16,
                                            attn_implementation="sdpa").to("cuda").eval()
    for module in (target, draft):
        module.requires_grad_(False)
    return target, draft, AutoTokenizer.from_pretrained(args.target)


def identity(args, target, draft):
    return dict(target=str(Path(args.target).resolve()), draft=str(Path(args.draft).resolve()),
                target_layer_ids=list(draft.target_layer_ids), hidden_size=draft.config.hidden_size,
                trained_total_block_size=draft.block_size, temperature=0, seed=args.seed,
                target_commit=getattr(target.config, "_commit_hash", None),
                draft_commit=getattr(draft.config, "_commit_hash", None))


@torch.inference_mode()
def collect(args):
    from .runtime import RoundRuntime, clock
    lengths = validate_lengths(args.lengths)
    target, draft, tokenizer = models(args)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = identity(args, target, draft)
    metadata.update(lengths=list(lengths), data=str(Path(args.data).resolve()), schema=1,
                    timing="synchronized B=1 draft+verification; excludes head and cache restore")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
    rng = random.Random(args.seed)
    records = []
    for index, prompt_id, ids in prompts(args, tokenizer):
        runtime = RoundRuntime(target, draft, ids.cuda())
        initial = ids.shape[1]
        rows = []
        while runtime.prefix.shape[1] - initial + max(lengths) + 1 <= args.max_new_tokens:
            pivot, anchor = runtime.features()
            accepted, draft_ms, verify_ms = [0] * len(lengths), [0.] * len(lengths), [0.] * len(lengths)
            cached = runtime.draft_cache.get_seq_length()
            order = list(range(len(lengths)))
            rng.shuffle(order)
            for action in order:
                k = lengths[action]
                # Warm each width on the first state of each prompt. Every timed action
                # starts with identical cache content, including context projection work.
                if not rows:
                    runtime.draft_cache.crop(cached)
                    warm = runtime.propose(k)
                    runtime.verify(warm)
                costs_d, costs_v = [], []
                for _ in range(args.repeats):
                    runtime.draft_cache.crop(cached)
                    begin = clock()
                    block = runtime.propose(k)
                    middle = clock()
                    result = runtime.verify(block)
                    end = clock()
                    costs_d.append((middle - begin) * 1000)
                    costs_v.append((end - middle) * 1000)
                accepted[action] = result[0]
                draft_ms[action], verify_ms[action] = statistics.median(costs_d), statistics.median(costs_v)
            rows.append(dict(pivot=pivot[0].cpu().half(), anchor=anchor[0].cpu().half(),
                             accepted=torch.tensor(accepted), draft_ms=torch.tensor(draft_ms),
                             verify_ms=torch.tensor(verify_ms), prefix_length=runtime.prefix.shape[1]))
            # Explore real verified states; the same chosen K does not determine labels for other K.
            runtime.draft_cache.crop(cached)
            runtime.step(rng.choice(lengths))
            if tokenizer.eos_token_id is not None and (runtime.prefix[:, initial:] == tokenizer.eos_token_id).any():
                break
        if rows:
            shard = dict(prompt_id=prompt_id, rows=rows)
            name = f"prompt_{index:06d}.pt"
            torch.save(shard, destination / name)
            records.append(dict(file=name, prompt_id=prompt_id, states=len(rows)))
            print(json.dumps(records[-1]), flush=True)
    if not records:
        raise RuntimeError("No states collected: check prompt limits and max_new_tokens >= max(K)+1.")
    (destination / "manifest.json").write_text(json.dumps(records, indent=2))


def train(args):
    torch.manual_seed(args.seed)
    source = Path(args.input)
    metadata = json.loads((source / "metadata.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    # Group duplicate prompts as well as all rounds from a prompt, preventing leakage.
    ids = sorted(set(entry["prompt_id"] for entry in manifest))
    if len(ids) < 2:
        raise ValueError("Need at least two distinct prompts for a held-out split.")
    random.Random(args.seed).shuffle(ids)
    val_ids = set(ids[:max(1, min(len(ids) - 1, round(len(ids) * args.val_fraction)))])
    splits = [[], []]
    for entry in manifest:
        shard = load(source / entry["file"])
        splits[int(shard["prompt_id"] in val_ids)].extend(shard["rows"])
    def tensors(rows):
        return {key: torch.stack([row[key] for row in rows]).float().to(args.device)
                for key in ("pivot", "anchor", "accepted", "draft_ms", "verify_ms")}
    training, validation = [tensors(rows) for rows in splits]
    head = LengthHead(metadata["hidden_size"], metadata["lengths"], features=args.features).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    # Cost calibration and fixed baseline use training prompts only.
    costs = (training["draft_ms"] + training["verify_ms"]).median(dim=0).values
    fixed = int(((training["accepted"] + 1).mean(0) / costs).argmax())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(len(training["pivot"]), device=args.device)
        for indices in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(training["pivot"][indices], training["anchor"][indices], training["accepted"][indices])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            predicted = head(validation["pivot"], validation["anchor"])
            error = torch.nn.functional.smooth_l1_loss(predicted, validation["accepted"]).item()
            selected = select_lengths(predicted, costs, head.lengths)
            row = torch.arange(len(selected), device=args.device)
            actual_cost = validation["draft_ms"] + validation["verify_ms"]
            def ratio(actions):
                return float((validation["accepted"][row, actions] + 1).sum() / actual_cost[row, actions].sum() * 1000)
            oracle = ((validation["accepted"] + 1) / actual_cost).argmax(-1)
            metrics = dict(epoch=epoch + 1, val_loss=error, paired_estimated_tok_s=ratio(selected),
                           fixed_estimated_tok_s=ratio(fixed), hindsight_estimated_tok_s=ratio(oracle))
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
            if error < best:
                best = error
                torch.save(dict(settings=head.settings, state_dict={k: v.cpu() for k, v in head.state_dict().items()},
                                costs_ms=costs.cpu(), fixed_k=head.lengths[fixed], metadata=metadata,
                                train_prompt_ids=[x for x in ids if x not in val_ids], val_prompt_ids=sorted(val_ids),
                                selection="minimum held-out SmoothL1 acceptance error", metrics=metrics), output)
    output.with_suffix(".metrics.json").write_text(json.dumps(history, indent=2))


@torch.inference_mode()
def evaluate(args):
    from .runtime import RoundRuntime, clock
    checkpoint = load(args.head)
    target, draft, tokenizer = models(args)
    current = identity(args, target, draft)
    for key in ("target", "draft", "target_layer_ids", "hidden_size"):
        if current[key] != checkpoint["metadata"][key]:
            raise ValueError(f"Checkpoint identity mismatch for {key}; recollect after changing a model.")
    head = LengthHead(**checkpoint["settings"]).cuda().eval()
    head.load_state_dict(checkpoint["state_dict"])
    costs = checkpoint["costs_ms"].cuda()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    seen = set(checkpoint["train_prompt_ids"] + checkpoint["val_prompt_ids"])
    for index, prompt_id, ids in prompts(args, tokenizer):
        if prompt_id in seen:
            raise ValueError("Evaluation prompt overlaps head training/validation; use a separate data file or offset.")
        reference = None
        policies = ["ar", f"fixed_{checkpoint['fixed_k']}", "adaptive"]
        if args.all_fixed:
            policies += [f"fixed_{k}" for k in head.lengths if k != checkpoint["fixed_k"]]
        # Warm all widths outside timed runs, including full vocab head and target verification.
        warmup = RoundRuntime(target, draft, ids.cuda())
        for k in head.lengths:
            cache_start = warmup.draft_cache.get_seq_length()
            warmup.verify(warmup.propose(k))
            warmup.draft_cache.crop(cache_start)
        head(*warmup.features())
        del warmup
        # Rotate speculative policy order to reduce a systematic cold/thermal advantage.
        tail = policies[1:]
        random.Random(args.seed + index).shuffle(tail)
        for policy in ["ar"] + tail:
            runtime = RoundRuntime(target, draft, ids.cuda())
            initial = ids.shape[1]
            actions = []
            begin = clock()
            while runtime.prefix.shape[1] - initial < args.max_new_tokens:
                remaining = args.max_new_tokens - (runtime.prefix.shape[1] - initial)
                if policy == "ar" or remaining < min(head.lengths) + 1:
                    k = 0
                    runtime.ar_step()
                else:
                    if policy == "adaptive":
                        prediction = head(*runtime.features())
                        action = int(select_lengths(prediction, costs, head.lengths, remaining).item())
                        k = head.lengths[action]
                    else:
                        requested = int(policy.split("_")[1])
                        # Same tail rule for fixed and adaptive: use the largest configured K that fits.
                        k = max(k for k in head.lengths if k <= requested and k + 1 <= remaining)
                    runtime.step(k)
                actions.append(k)
                new = runtime.prefix[:, initial:]
                if tokenizer.eos_token_id is not None and (new == tokenizer.eos_token_id).any():
                    break
            elapsed = clock() - begin
            tokens = runtime.prefix[0, initial:].tolist()
            if tokenizer.eos_token_id in tokens:
                tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
            if policy == "ar":
                reference = tokens
            row = dict(index=index, prompt_id=prompt_id, policy=policy, tokens=len(tokens),
                       decode_s=elapsed, tok_s=len(tokens) / elapsed, match_ar=tokens == reference,
                       rounds=len(actions), candidate_lengths=actions)
            results.append(row)
            print(json.dumps(row), flush=True)
            with (destination / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    if not results:
        raise RuntimeError("No evaluation prompts survived the length filter.")
    summary = {}
    for policy in sorted(set(row["policy"] for row in results)):
        rows = [row for row in results if row["policy"] == policy]
        summary[policy] = dict(tok_s=sum(row["tokens"] for row in rows) / sum(row["decode_s"] for row in rows),
                               match=sum(row["match_ar"] for row in rows), samples=len(rows),
                               rounds=sum(row["rounds"] for row in rows))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not all(row["match_ar"] for row in results):
        raise RuntimeError("AR token mismatch observed; inspect results before claiming lossless speedup.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--target", required=True)
        p.add_argument("--draft", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--samples", type=int, default=64)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--max-prompt", type=int, default=512)
        p.add_argument("--max-new-tokens", type=int, default=256)
        p.add_argument("--seed", type=int, default=42)
        if name == "collect":
            p.add_argument("--lengths", nargs="+", type=int, default=[4, 8, 12, 16, 24, 32])
            p.add_argument("--repeats", type=int, default=2)
        else:
            p.add_argument("--head", required=True)
            p.add_argument("--all-fixed", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--features", choices=["both", "anchor", "pivot"], default="both")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "collect" and args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.command == "train" and (not 0 < args.val_fraction < 1 or args.epochs < 1 or args.batch_size < 1):
        parser.error("invalid training settings")
    if args.command in ("collect", "eval") and (args.samples < 1 or args.max_new_tokens < 1 or args.max_prompt < 1):
        parser.error("samples and token limits must be positive")
    {"collect": collect, "train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
