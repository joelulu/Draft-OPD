# Two datasets, fixed/mixed draft, fixed K4..32

```bash
source ../.venv-draftopd/bin/activate
export RESULT_ROOT="$PWD/runs/fixed_vs_dynamic_20261107_051744"
export GSM8K_TEST_JSONL="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/eval_datasets/gsm8k.jsonl"
# matplotlib must be available; install if missing: python -m pip install matplotlib
SWEEP_GPUS=6,7 EVAL_SAMPLES=16 MAX_NEW_TOKENS=512 EVAL_REPEATS=3 \
  bash scripts/adaptive_length/run_two_dataset_k_sweep.sh
```

One GPU per dataset, parallel when two GPU IDs are supplied; a single ID runs
datasets sequentially. Each process loads one target and both frozen drafts on
its GPU, then interleaves all 16 `(draft,K)` methods in randomized order within
each prompt/repeat. This reduces model-order confounding without comparing
models on different GPUs. It requires memory for both drafts plus target and
one decode runtime (no full optimizer/training state). There is no predictor
or oracle search. First 16 unique prompts are used in source order. The mixed
holdout source defaults to existing `labeled/test_prompts.jsonl`; GSM8K is local
only, supports either `question` or single-string `turns`, preserving reasoning
instructions. No dataset download. Actual selected prompts, input file hash and
weight fingerprints are saved. Overlong prompts cause an error rather than
silently substituting new samples; raise MAX_PROMPT if necessary.

Primary throughput is full B1 greedy decode, until EOS or 512 emitted tokens.
Prefill and warmup are excluded, every timed replay uses a new state. Same K
across models is compared with sum(tokens)/sum(decode wall time); per-prompt win
counts are also reported. Outputs/lengths may differ, so this is not a quality
or lossless speed claim. Heatmaps compare each policy's own decode trajectory,
not a shared-state causal intervention.

Each dataset additionally collects up to four **common target-AR states** per
prompt (generated positions 0,64,128,192 by default, stopping on EOS and requiring
33 output tokens of remaining capacity). Each probe reconstructs fresh target
and draft caches on that exact common prefix, uses the same reference anchor,
and tries every K for both drafts. Probes are excluded from throughput. Fresh
caches avoid stale draft context when AR advancement skips many draft steps.
This phase answers whether larger K changes acceptance at a controlled state;
not whether longer trajectories or output lengths explain a mean increase.
Set PROBE_STATES=0 to omit probes, PROBE_STATES/PROBE_STRIDE/PROBE_REPEATS to tune.
Rows encountering EOS inside any of the two models' trials are excluded from
both models' paired report, ensuring a shared denominator. Different prompts
may contribute fewer states; per-state weighting is explicit.

Results are in a fresh `two_dataset_k_sweep_TIMESTAMP/`:

- `throughput_all.csv`, `summary.txt`: both datasets' K-by-K throughput,
  same-K mixed/fixed percent gains, per-prompt wins, accepted length and rounds.
- **`mixed_holdout_acceptance_heatmap.png` and `gsm8k_acceptance_heatmap.png`**
  (also PDF): each figure has fixed/mixed panels with a shared color scale.
  X=K, Y=accepted candidate count excluding anchor, color=P(accept=a | K).
  Blue line=mean acceptance; n per column=valid rounds. Gray cells are a>K
  or missing data. First repeat only (not counting deterministic repeats as
  independent observations); tail-clipped, AR and EOS-adjusted rounds excluded.
  Heatmap means can differ slightly from throughput's all-round means.
- Each dataset directory: `throughput.csv`, `acceptance_distribution.csv`,
  `results.jsonl`, `rounds.csv`, `selected_prompts.jsonl`, logs and metadata.
- `same_state_acceptance.csv`: same prefix/anchor, fixed vs mixed acceptance
  means and better/equal/worse counts at each K.
- `same_state_k_changes.csv`: each small/large K pair's better/equal/worse
  counts, including improvements when the smaller K already had capacity for
  the larger block's accepted prefix. Larger K is not assumed monotonic.
- `paired_states.jsonl`, `same_state_scope.json`: raw paired acceptance/timing,
  common prefix hashes, anchors, probe positions and sample exclusions.

CPU tests with real tiny randomly initialized Qwen3/DFlash, no downloaded weights:

```bash
PYTHONPATH=diffusion/dflash python -m pytest -q scripts/adaptive_length/test_k_sweep.py
```

Plot-only regeneration (no GPU rerun):

```bash
python scripts/adaptive_length/report_k_sweep.py --root /absolute/path/to/two_dataset_k_sweep_TIMESTAMP
```
