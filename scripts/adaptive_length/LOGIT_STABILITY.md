# Cross-K logit stability and adaptive expansion

```bash
source ../.venv-draftopd/bin/activate
export RESULT_ROOT="$PWD/runs/fixed_vs_dynamic_20261107_051744"
export GSM8K_TEST_JSONL="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/eval_datasets/gsm8k.jsonl"
STABILITY_GPUS=6,7 EVAL_SAMPLES=16 STATES_PER_PROMPT=4 STATE_STRIDE=64 \
MAX_NEW_TOKENS=512 COLLECT_REPEATS=2 EVAL_REPEATS=3 \
  bash scripts/adaptive_length/run_logit_stability.sh
```

Local mixed holdout and GSM8K, first 16 unique prompts each, source order,
single-turn `turns` supported and retained. No downloads/training. Each dataset
runs on its own GPU; both drafts reside with target on that GPU. One GPU ID
runs sequentially. Defaults: four common target-AR states/prompt at generated
positions 0,64,128,192, requiring >=33 token capacity. Common prefix and reference
anchor are identical across models. At each state all K4..32 step4 are measured
twice in shuffled order. Prefix/anchor hashes and frozen-weight fingerprints
are saved. Repeated proposals/acceptance and cache lengths must agree.

## What is stored

Each K writes exactly K candidate positions, excluding anchor, up to32. Positions
above K are blank in 32-column marker tables. K32 is a reference, not truth or
an assertion that K32 is best. For every position:

- top5 token IDs, token text legend, original logits and full-softmax probabilities;
- entropy, top1/top2 logit margin;
- common target-AR reference ID, its probability/margin and hit, where available;
- actual target-verification posterior ID and hit. After first rejection this
  posterior is conditioned on the draft's preceding proposals, not common AR.

Top5 ties use repeated argmax, ascending token-ID ties, matching greedy proposal
top1. This avoids arbitrary `torch.topk` tie ordering. JS divergence uses full
FP32 vocabulary distributions in natural-log units; full logits are never saved.

Adjacent K and K-vs32 compare only shared positions. Saves each position's JS,
top1 equality, unordered top5 overlap, ordered top5 equality, peak probability
delta and margin delta. Summary includes mean/max/last4 JS, top1 change rate,
top5 distance (1-overlap) and mean peak-probability change. Candidate positions
are tagged before/at/after the smaller K's first rejection and K32's boundary.
Actual acceptance is EOS-adjusted; raw acceptance also remains available.

## Stop-rule experiment

First half of selected prompts is calibration; second half is held-out test.
No prompt/state from test selects a threshold. With defaults, 8 calibration
prompts and 8 test prompts per dataset. Calibration and held-out test state
statistics can be small; this is an exploratory experiment, not a final claim.

Candidate stopping signals: common-prefix top1 change, unordered top5 distance,
full-distribution mean JS, peak probability change; four thresholds each and
one/two consecutive stable transitions. Expansion starts K4, first possible
stop K8 (or K12 with patience2), returns the currently computed larger block.
No rule looks at target acceptance or K32-vs-current during online decisions.

Calibration picks the highest optimistic same-state speed proxy among rules
retaining >=95% of oracle committed tokens. If none qualifies it explicitly
marks failure and picks highest retention (speed breaks ties). Proxy costs
include every tried draft, one chosen verification, context prep once; **omit
decision metric and CPU overhead** and are not decode-speed claims. Online test
is authoritative. EOS-adjusted states are excluded from calibration/held-out
state rule summaries. `stability_vs_future_gain.csv` checks how often low change
still permits future acceptance gains, separately for calibration/test.

## Actual online evaluation

After threshold selection, only held-out test prompts run fixedK16, fixedK32,
and selected stability rule for both drafts, three repeats, randomized job order.
Each runs its own full sequence until EOS or512 tokens. No counterfactual search
is included in measured runs. Every tried draft forward and full signal
computation is charged; target verifies only once per selected block. Confirmed
draft context KV is reused, but changing K recomputes noise hidden states.
This is not a perfect-selector experiment. Outputs may differ between policies;
no quality/lossless claim. Decision wall diagnostics can include asynchronous
draft work; total decode wall throughput is the comparison metric.

## Outputs under each dataset directory

- `states.jsonl`: complete raw per-state K trials, positions and pair comparisons.
- `token_legend.json`: token IDs to decoded text (individual token pieces).
- `top1_grid_32.csv`: state/K rows aligned to positions1..32, unused positions blank.
- `top5_grid_32.csv`: same 32-position grid, each filled cell contains five
  token IDs/text/probabilities; K8 fills only positions1..8.
- `top5_markers.csv`: all K/position top5, margins, entropy, reference and hit data.
- `position_changes.csv`, `change_summary.csv`: adjacent and K32-reference changes,
  grouped by rejection boundary. These descriptive summaries pool all prompts;
  threshold selection remains calibration-only.
- `k_curves.csv`: pooled descriptive acceptance, top1/top5 changes vsK32, JS,
  and same-K repeat JS noise floor. Printed to console as well.
- `calibration_rules.csv`: all32 rules on calibration states only.
- `chosen_rules.json`: chosen rule, calibration metrics and independent test metrics,
  frozen fingerprints and disjoint prompt keys.
- `stability_vs_future_gain.csv`: conditional later-gain diagnostics, cal/test separate.
- `online/results.jsonl`, `online/comparison.csv`: true full-decode test performance,
  acceptance, draft calls per verification, total candidate work/output token.

`RUN_ONLINE=0` runs collection/analysis only. Individual phases can be rerun with
`python scripts/adaptive_length/logit_stability.py analyze --output EXISTING_DIR`.
Online eval requires the same target/drafts/data and sample/seed/length settings
as collection, and a fresh `online/` directory (existing results are preserved).
Collection requires a fresh root; partial records are kept after interruption.

CPU checks use tiny random Qwen3/DFlash, no downloaded weights:

```bash
PYTHONPATH=diffusion/dflash python -m pytest -q scripts/adaptive_length/test_logit_stability.py
```
