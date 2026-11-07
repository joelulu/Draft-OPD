# Original verl OPD with mixed response widths

For the complete two-stage variable-rollout/classifier pipeline see [TWO_STAGE.md](TWO_STAGE.md).

This is separate from `compare_training.sh`, the standalone reference trainer.
`native_opd.sh fixed|mixed` calls the repository's original
`verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh`.
It uses its existing FSDP, SGLang rollout, scalar OPD loss, rejection replay,
position decay and checkpoint mechanism. No full-vocabulary reference loss or
length-prediction head is added.

## What changes

- `fixed`: `verl_dflash_train_candidate_lengths=[]`; the original native-width
  response/rejected forward path is preserved.
- `mixed`: response training forwards sample an actual candidate count from
  `[4,8,12,16,20,24,28,32]`, with input width K+1. One width is sampled per composed
  gradient-bearing training forward, not per optimizer update. Eval/no-grad
  logprob computation uses native width.
- Rollout and rejected-proposal replay retain native width (usually 15 candidates).
  In mixed mode the rejected stream runs separately at its native width; short
  response blocks never discard rejected offsets larger than sampled K.
- Small K tiles each originally supervised accepted segment into smaller blocks.
  Thus original response supervision positions are preserved. Large K uses more
  MASK slots but does not fabricate acceptance labels past rollout boundaries.
  It trains the accepted prefix under a larger actual bidirectional draft block.
- Original teacher probabilities and loss functions are preserved. Repartitioned
  response anchors/offsets, and therefore position-decay weights when enabled,
  can change; this is not claimed to be an identical gradient estimator.
- No change is made to draft checkpoint `block_size` or SGLang allocation/graphs.

This is **mixed training widths with native rollout**, not fully multi-width
on-policy rollout. It does not test learned length selection, and it does not
provide on-policy rejection trajectories for K=24/32. A later rollout extension
must also change worker buffers, verification lengths and feedback metadata.

## Run on the server

The nested checkout can reuse `../.venv-draftopd` and `../data/...`. The launcher
prepends this checkout's verl/SGLang paths and prints the loaded student file so
an editable installation pointing at the parent cannot silently load older code.
Model and CUDA defaults follow the server paths supplied by the user.

First validate each arm with short rollouts and a handful of update batches:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
for ARM in fixed mixed; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  EXP_NAME="native-opd-${ARM}-smoke_${STAMP}" \
  CKPT_DIR="$PWD/checkpoints/native-opd-${ARM}-smoke_${STAMP}" \
  NATIVE_OPD_BATCH_STEPS=5 UPDATE_ACCUMULATION_STEPS=1 \
  TRAIN_PROMPT_BSZ=8 MAX_RESPONSE_LENGTH=256 \
  SAVE_FREQ=5 SAVE_START_STEP=5 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False \
  bash scripts/adaptive_length/native_opd.sh "$ARM"
done
```

`NATIVE_OPD_BATCH_STEPS` is the upstream trainer's `total_training_steps`,
which counts rollout/training batches. With update accumulation >1 this is not
optimizer update count. Use the same accumulation, data, seed, LR and batch limit
for both arms. Native default LR here is `3e-5`; set `LR=3e-4` explicitly to reuse
the user's earlier setting. Upstream defaults for scalar loss/stream weights are
in forward-ins.sh. Target/draft/teacher paths and LR remain environment-overridable.

After smoke passes, remove NATIVE_OPD_BATCH_STEPS (epochs govern run length),
restore the desired batch/response settings, and compare the original fixed arm
with mixed under the same controls. This native experiment matches prompt/update
settings, not the previous reference trainer's exact 240-candidate budget. Record
actual target/rejected token counts and additional draft computation.

Metrics: `distillation/opd_train_response_candidates`,
`distillation/opd_native_replay_candidates`, `distillation/opd_mixed_response_training`
and `distillation/opd_train_width_<K>_fraction` show active lengths. Rejection replay
should remain at native K. The native mixed path currently rejects TV mode.

The original launcher's blanket `pkill python`/`pkill ray` was removed. An explicit
`RESET_RAY_RUNTIME=True` enables `ray stop --force` in the base launcher; this wrapper
keeps it disabled. Optional GPU stress-on-exit is disabled here. Dynamic draft
widths also disable torch.compile in both arms of this launcher to avoid repeated
shape compilation during the initial experiment.

## Checks

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  diffusion/dflash/tests/test_native_mixed_opd.py \
  diffusion/dflash/tests/test_adaptive_length.py
```

Tests run the actual native composed student with tiny real Qwen3/DFlash modules:
K=4/12/32 input widths, native rejection replay (including an offset beyond sampled
K), unchanged response mask and teacher labels, draft-only gradients, native-only
numerical equivalence, eval/no-grad behavior and unsupported TV guards. They do
not launch Ray/FSDP/SGLang or validate GPU memory/runtime; server smoke is required.

## SDPA padding without the flash_attn package

The generic padding dispatcher now uses the repository's existing pure
PyTorch/einops padding helpers when `flash_attn.bert_padding` is absent. Native
FlashAttention helpers remain preferred when available; unrelated import errors
are not suppressed. This only removes an unnecessary dependency for
index/pad/unpad operations. It does not emulate FlashAttention kernels.
The native launcher runs a tiny padding roundtrip before initializing Ray/models,
so this dependency path is checked before an expensive multi-GPU startup.

Regression tests: `python -m pytest -q diffusion/dflash/tests/test_attention_padding_fallback.py`.

## Diagnose fixed versus mixed after export

Analyze an existing three-arm evaluation, with no GPU/model loading:

```bash
python scripts/adaptive_length/analyze_native_eval.py \
  --root runs/native_opd_eval_20261106_100359
```

This requires complete, identical prompt/K grids across initial/fixed/mixed,
matching generation metadata, valid positive timings, and profiling disabled.
It writes `analysis/comparison.csv`, `choices.csv`, and `report.json`.
Throughput is total emitted tokens divided by total decode time, never an
arithmetic mean of per-prompt token rates. The per-prompt hindsight selection
chooses the fastest K after seeing all results. It is an optimistic in-sample
reference affected by timing noise and selection bias, not a trained policy,
a bound on per-round selection, or evidence about output quality.

Repeat measurements using already exported models (no training/re-export):

```bash
MODEL_ROOT="$PWD/runs/native_opd_eval_20261106_100359" \
VAL_JSONL="$PWD/../data/opd_val_32.jsonl" \
EVAL_GPU=7 REPEATS=3 EVAL_SAMPLES=32 MAX_NEW_TOKENS=256 \
bash scripts/adaptive_length/repeat_native_eval.sh
```

Fixed/mixed order alternates; initial is measured each repeat. Every evaluation
uses B=1, the same GPU and data, no head, and non-strict matching. Use an idle
GPU. Output must be a fresh directory. Analysis reports pooled throughput and
per-repeat median/range; hindsight choices use pooled measurements per prompt.
The runner records the data path and SHA256. This does not establish that data
is independent of training: replace VAL_JSONL with a separate held-out test set
before making generalization claims. Different speculative modes can emit
different outputs/token counts, so the oracle optimizes their measured token
rates, not a common fixed-output latency task.
