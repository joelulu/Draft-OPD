"""Validate matched-budget runs and combine their fixed-width evaluation tables."""
import argparse
import csv
import json
from pathlib import Path


def build_comparison(root):
    root = Path(root)
    plan = json.loads((root / "comparison_plan.json").read_text())
    manifests, training = {}, {}
    for arm in ("fixed", "mixed"):
        manifests[arm] = []
        totals, elapsed = 0, []
        for rank in range(plan["nproc"]):
            folder = root / arm / "draft"
            meta = json.loads((folder / f"training_rank{rank:03d}.json").read_text())
            logs = [json.loads(line) for line in (folder / f"train_rank{rank:03d}.jsonl").read_text().splitlines()]
            if len(logs) != plan["steps"] or any(
                row["step"] != i+1 or row["candidate_tokens"] != plan["candidate_tokens_per_rank_per_update"]
                for i, row in enumerate(logs)):
                raise ValueError(f"Incomplete or unmatched training budget: {arm}, rank {rank}")
            expected = plan["steps"] * plan["candidate_tokens_per_rank_per_update"]
            if logs[-1]["cumulative_candidate_tokens_per_rank"] != expected:
                raise ValueError("Cumulative candidate budget mismatch.")
            totals += expected
            elapsed.append(logs[-1]["elapsed_s"])
            manifests[arm].append(meta)
        training[arm] = dict(global_candidate_tokens=totals,
                            approximate_training_gpu_hours=max(elapsed)*plan["nproc"]/3600)
    matched = ("initial_draft", "target", "seed", "world_size", "learning_rate",
               "train_data", "split", "max_prompt", "max_new_tokens", "local_prompt_keys",
               "planned_steps", "candidate_tokens_per_rank_per_update", "train_parameters_dtype")
    for rank in range(plan["nproc"]):
        for name in matched:
            if manifests["fixed"][rank][name] != manifests["mixed"][rank][name]:
                raise ValueError(f"Training control differs: rank {rank}, {name}")
        if manifests["fixed"][rank]["lengths"] != [plan["fixed_k"]]:
            raise ValueError("Fixed arm did not train at its declared native width.")
        if manifests["mixed"][rank]["lengths"] != plan["mixed_lengths"]:
            raise ValueError("Mixed arm width set differs from experiment plan.")
    combined, prompt_sets = [], []
    for arm in ("initial", "fixed", "mixed"):
        folder = root / arm / "eval_fixed"
        records = [json.loads(line) for line in (folder / "results_rank000.jsonl").read_text().splitlines()]
        if any(r["mode"] == "adaptive" for r in records):
            raise ValueError("This training comparison must not use a length head.")
        identities = {(r["prompt_key"], r["mode"]) for r in records}
        if len(identities) != len(records):
            raise ValueError("Duplicate evaluation records.")
        prompt_sets.append(identities)
        with (folder / "summary_rank000.csv").open() as stream:
            for row in csv.DictReader(stream):
                combined.append(dict(training=arm, **row))
    if not prompt_sets[0] == prompt_sets[1] == prompt_sets[2]:
        raise ValueError("The three checkpoints were not evaluated on identical prompts and widths.")
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    report = dict(budget_matched=True, prompt_schedule_matched=True, training=training,
                  note="Candidate supervision and optimizer updates matched; FLOPs/wall time and on-policy prefixes may differ.")
    (root / "budget_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Results: {root / 'comparison.csv'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    build_comparison(parser.parse_args().root)
