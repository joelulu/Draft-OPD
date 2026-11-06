"""CPU-only paired fixed-K analysis; no model loading or training."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("initial", "fixed", "mixed")


def analyze(root):
    runs = defaultdict(dict)
    settings = None
    for path in sorted(root.rglob("results_rank*.jsonl")):
        ancestors = [p for p in path.parents if p.name in ARMS]
        if not ancestors:
            continue
        arm_dir = ancestors[0]
        run = str(arm_dir.parent.relative_to(root))
        arm = arm_dir.name
        meta = json.loads(path.with_name(path.name.replace("results_", "metadata_").replace(".jsonl", ".json")).read_text())
        if meta.get("profile_rounds"):
            raise ValueError(f"Profiling enabled: {path}")
        config = {k: meta[k] for k in ("schema", "native_block_size", "fixed_lengths", "tail_policy", "seed", "max_new_tokens")}
        if settings is not None and config != settings:
            raise ValueError(f"Different evaluation settings: {path}")
        settings = config
        records = runs[run].setdefault(arm, {})
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row["mode"].startswith("fixed_"):
                continue
            key = (row["prompt_key"], row["mode"])
            if key in records:
                raise ValueError(f"Duplicate prompt/mode: {path}: {key}")
            if row.get("profile_rounds") or not math.isfinite(row["decode_s"]) or row["decode_s"] <= 0 or row["tokens"] <= 0:
                raise ValueError(f"Invalid timing/token record: {path}: {key}")
            records[key] = row
    if not runs:
        raise ValueError(f"No initial/fixed/mixed evaluation records under {root}")
    expected = None
    modes = [f"fixed_{k}" for k in sorted(settings["fixed_lengths"])]
    for run, arms in runs.items():
        if set(arms) != set(ARMS):
            raise ValueError(f"Run {run} must contain all three arms; got {sorted(arms)}")
        for arm, records in arms.items():
            keys = set(records)
            prompts = {k[0] for k in keys}
            if not keys or keys != {(p, m) for p in prompts for m in modes}:
                raise ValueError(f"Incomplete prompt/K grid: {run}/{arm}")
            if expected is not None and keys != expected:
                raise ValueError(f"Prompt sets differ: {run}/{arm}")
            expected = keys
    prompts = sorted({k[0] for k in expected})
    pooled = {a: defaultdict(list) for a in ARMS}
    for arms in runs.values():
        for arm in ARMS:
            for key, row in arms[arm].items():
                pooled[arm][key].append(row)

    def speed(rows):
        return sum(r["tokens"] for r in rows) / sum(r["decode_s"] for r in rows)

    summary, choices = [], []
    for arm in ARMS:
        for mode in modes:
            rows = [r for p in prompts for r in pooled[arm][p, mode]]
            per_run = [speed([v for (p, m), v in arms[arm].items() if m == mode]) for arms in runs.values()]
            spec = sum(r["spec_rounds"] for r in rows)
            summary.append(dict(training=arm, mode=mode, decode_tok_s=speed(rows),
                                repeat_median_tok_s=statistics.median(per_run),
                                repeat_min_tok_s=min(per_run), repeat_max_tok_s=max(per_run),
                                mean_accepted=sum(r["accepted_tokens"] for r in rows)/spec if spec else None,
                                mean_committed=sum(r["spec_committed_tokens"] for r in rows)/spec if spec else None))
        selected = []
        for prompt in prompts:
            mode = max(modes, key=lambda m: speed(pooled[arm][prompt, m]))
            selected.extend(pooled[arm][prompt, mode])
            choices.append(dict(training=arm, prompt_key=prompt, mode=mode,
                                decode_tok_s=speed(pooled[arm][prompt, mode])))
        summary.append(dict(training=arm, mode="hindsight_per_prompt", decode_tok_s=speed(selected)))
    best = {a: max((r for r in summary if r["training"] == a and r["mode"].startswith("fixed_")),
                   key=lambda r: r["decode_tok_s"]) for a in ARMS}
    oracle = {a: next(r["decode_tok_s"] for r in summary if r["training"] == a and r["mode"] == "hindsight_per_prompt") for a in ARMS}
    report = dict(prompts=len(prompts), repeats=len(runs), settings=settings, best_fixed=best,
                  hindsight_tok_s=oracle,
                  mixed_hindsight_gain_over_fixed_best_pct=100*(oracle["mixed"]/best["fixed"]["decode_tok_s"]-1),
                  choices={a: dict(Counter(r["mode"] for r in choices if r["training"] == a)) for a in ARMS},
                  note="Optimistic in-sample per-request selection using measured throughput; includes selection bias and timing noise. Not a learned policy, a per-round oracle, or an output-quality/lossless claim. Outputs and token counts may differ between modes. Repeats must reuse identical prompts and generation settings.")
    return summary, choices, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary, choices, report = analyze(args.root)
    out = args.output or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("comparison", summary), ("choices", choices)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"prompts={report['prompts']} repeats={report['repeats']}")
    for arm in ARMS:
        best = report["best_fixed"][arm]
        print(f"{arm:8s} best={best['mode']:9s} {best['decode_tok_s']:.2f} tok/s; hindsight={report['hindsight_tok_s'][arm]:.2f} tok/s")
    print(f"mixed hindsight vs fixed best: {report['mixed_hindsight_gain_over_fixed_best_pct']:+.2f}%")
    print(report["note"])
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
