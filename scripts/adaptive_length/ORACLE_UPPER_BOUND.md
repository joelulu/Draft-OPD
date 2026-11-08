# Fixed/mixed local-oracle benchmark

## Simple GSM8K eight-prompt comparison (no predictor)

```bash
export RESULT_ROOT="$PWD/runs/fixed_vs_dynamic_20261107_051744"
EVAL_GPU=7 EVAL_SAMPLES=8 MAX_NEW_TOKENS=512 EVAL_REPEATS=3 \
  bash scripts/adaptive_length/run_gsm8k_oracle.sh
```

Downloads/uses cached `openai/gsm8k`, config `main`, split `test`. Selects the
first eight unique questions in source order, no shuffle. Only raw question is
input as a user message; reference answers are saved for provenance, never fed
to the model. Source indices and dataset fingerprint are saved. If Hugging Face
is unavailable, set `GSM8K_TEST_JSONL=/absolute/path/to/test.jsonl` for an existing
GSM8K test file containing `question` and optional `answer`. Local-file provenance
is recorded honestly; test membership is then the caller's responsibility.

Both drafts run fixed K16 and acceptance oracle only. No predictor is created,
warmed, or executed; speed-oracle search and separate profiling are skipped.
Prints first timed repeat's full `accepted(actual_K)` trajectory per prompt,
e.g. `4(16), 6(16), 3(8), AR`. Accepted excludes anchor; committed is accepted+1.
Oracle candidates remain 4..32 at interval4, **not** K=accepted. Near the output
budget, actual K can clip to other integers. `rounds.csv` retains requested K,
actual K, effective emitted acceptance, raw acceptance and committed length,
including EOS adjustment. Printing and CSV writing occur outside timed decode.

`simple_comparison.csv` reports four rows and gains relative to each draft's own
K16 plus the common fixed-trained K16 baseline. `round_distribution.csv` groups
actual K and acceptance histograms using first repeat only. `results.jsonl`
retains all repeats. Existing full matrix mode remains unchanged apart from
additional saved round details. Prompts stop on EOS or the configured token cap;
the cap is not a guarantee that the full reasoning answer finishes.

Run from the repository root with the existing fixed/mixed exported drafts.
This does not train a predictor, modify existing models, or reuse the mixed
draft's labels as if they belonged to the fixed draft.

```bash
source ../.venv-draftopd/bin/activate
export RESULT_ROOT="$PWD/runs/fixed_vs_dynamic_20261107_051744"
# Quick functional smoke; not enough for performance conclusions.
EVAL_GPU=7 EVAL_SAMPLES=2 MAX_NEW_TOKENS=128 EVAL_REPEATS=1 SEARCH_REPEATS=2 \
  bash scripts/adaptive_length/run_oracle_upper_bound.sh
# Main run; one GPU, both drafts evaluated sequentially.
EVAL_GPU=7 EVAL_SAMPLES=16 MAX_NEW_TOKENS=512 EVAL_REPEATS=3 SEARCH_REPEATS=3 \
  bash scripts/adaptive_length/run_oracle_upper_bound.sh
```

Each draft gets eight fixed candidate counts (4..32, interval 4), an acceptance
oracle, a one-round speed oracle, and two perfect-acceptance-selector cost
experiments (full history and last 64 historical positions). K excludes anchor.
Every method reuses the same selected unique prompts. Greedy, B=1, thinking off.
EOS is honored and may yield different emitted token counts and outputs.

For each oracle, discovery advances its own selected trajectory. At each state,
all eight choices start with the same pending context/cache boundary. Warmup and
shuffled repeated draft+verify trials are discovery work only. Trial proposals
and acceptance must repeat exactly; caches must return to incoming lengths.
The chosen action is executed and the entire trace is saved. Acceptance oracle
maximizes EOS/budget-adjusted accepted candidates; ties choose smallest nominal
K. Speed oracle maximizes emitted/(median synchronized draft+verify ms), not a
calibrated cross-state cost. Duplicate tail-clipped actions are allowed with the
same tie rule. This is a greedy **local** oracle, not the globally fastest
sequence policy or a guarantee that local-speed oracle beats acceptance oracle.

Independent replay measures actual whole-decode wall time without any oracle
search. Output tokens and each round's chosen K, raw acceptance and committed
count must equal discovery, including when reader cost is added. Any mismatch
raises instead of printing a performance claim. Each repeat randomizes method
order. Frozen weight fingerprints, prompt keys and settings are saved.

The perfect-selector cost experiments execute the actual frozen native DFlash
first-layer anchor-only Q/KV/norm/RoPE/GQA/output/residual/MLP path, followed by
FP32 LayerNorm + Linear(H,128) + SiLU + Linear(128,8) and argmax readback. The
randomly initialized head's answer is discarded and oracle supplies the perfect
class. These measure the cost of **this architecture**, not trained accuracy or
every possible predictor. Window64 crops historical native KV to its last 64
absolute positions (plus anchor self); native sliding-window masking still
applies. It is not a DLite slot compressor. No future MASKs are used. Context
materialization populates real draft KV and is reused by proposal, so it is not
charged twice. Full history keeps all cached keys, subject to native layer mask.

Outputs in a fresh `oracle_upper_bound_TIMESTAMP/` directory:

- `comparison.csv`: primary uninstrumented throughput, acceptance, committed
  length, spec rounds/1000 emitted tokens, candidates/output token, selection
  wall time, ratios to fixed-trained K16 and each arm's best fixed K.
- `results.jsonl`: one primary measurement per prompt/method/repeat, including
  output hash and requested K histogram. Totals use sum(tokens)/sum(time).
- `schedules/{fixed,mixed}/*_{accept,speed}.json`: each oracle's own trace,
  generated tokens and all counterfactual acceptance/timing trials.
- `profile_comparison.csv`: a **separate** event-instrumented replay once per
  prompt/method, showing draft/verify/selector/advance/AR GPU milliseconds per
  1000 tokens and per speculative round. Instrumented wall time is excluded
  from primary throughput. AR per-round values use the speculative denominator;
  use per-1000-token values for AR interpretation.
- `selector_budget.json`: token-normalized additive wall-time slack per oracle
  selector call, relative to own K16 and own best fixed K. A negative value means
  the oracle already loses without predictor. This is an approximate budget,
  not a hardware-independent latency bound.
- `run.log`, `metadata.json`: progress, provenance and experiment scope.

Selection wall time includes pending context preparation that otherwise occurs
in draft generation. GPU events include host-enqueue gaps within event scopes;
do not interpret their sums as an exhaustive kernel-only decomposition. Compare
selector+draft jointly and use total primary wall throughput for the conclusion.
Search trials have synchronization boundaries absent from primary replay, so the
speed oracle optimizes a timing surrogate. Independent replay is the check that
any local gain translates to a real decode gain. No output-quality or lossless
claim is made. An oracle chooses using future verification unavailable to a
deployable predictor; speed winner selection is optimistic and timing-sensitive.

CPU regression checks (no downloaded weights):

```bash
PYTHONPATH=diffusion/dflash python -m pytest -q \
  scripts/adaptive_length/test_oracle_upper_bound.py
```
