# Two-stage state-conditioned block length

K always excludes the anchor. Actions are **4,8,12,16,20,24,28,32**.
No length 15 is added to classifier evaluation. The old continuous acceptance
regressor/head is separate and is not loaded by this pipeline.

## Implemented scope

1. Stage 1: native verl scalar Draft-OPD. Eager SGLang spec-v1 samples one K per
   scheduler decode batch/round (all requests in that batch share K). Each request
   can encounter different K on successive rounds. Allocation reserves 33 tokens;
   actual speculative tensors and FA3 eager metadata follow the sampled K+1.
   Original rollout anchors, accepted lengths and actual K are recorded even when
   replay-dis diagnostics are disabled. The student groups those original blocks
   by K, applies original scalar response/rejection losses, and replays rejected
   proposals with that original K. Frozen teacher hidden is reused across groups.
   All FSDP ranks execute the same width/forward schedule; empty groups perform
   zero-loss dummy forwards. This costs extra compute; no equal-FLOPs claim.
   TV, random response anchors, replay-dis diagnostics and overlap/CUDA graphs are
   not supported in this experimental variable-width stage. Existing native OPD
   commands retain their old behavior unless `stage1_opd.sh` is invoked.
2. Stage 2: freeze a local exported target/draft pair; collect eight actual
   proposals/verifications per state, independently restoring caches. The same
   pending confirmed-context projection is included for every K. Warmup precedes
   timed trials; trial order is shuffled. Acceptance must agree across repeats.
3. Save **both labels**: maximum accepted candidates (smallest K on a tie), and
   maximum `(accepted+1)/cost`. Costs use medians from training states in log2
   context-length bins; validation/test timing never defines the table. Missing
   bins use the nearest training bin. Raw draft/verify/repeated total timings and
   all acceptances are retained so alternative labels can be generated later.
   Labels count emitted tokens through the first EOS; raw verification acceptance
   is also retained. States are only sampled with at least 33 output tokens left.
4. Train four CE classifiers: full_kv/window_16 × accept/speed. Both read the
   **target** KV of every layer/head; full means every cached position, not the
   original token text and not the draft KV. Window_16 slices the sequence axis
   before attention. A trainable anchor-embedding projection provides q; it is
   available before selecting K and is not the result of a draft block forward.
   Layer/head query biases + attention aggregation + MLP produce eight logits.
   All four classifiers start from identical weights. Only predictor parameters
   train; frozen target KV is streamed/rebuilt per prompt, without storing full
   cache copies per state. Classifier training is B=1 with gradient accumulation
   on one GPU. Collection can use independent replicas on several GPUs.
5. B=1 HF reference inference chooses K before every block; no classifier alters
   draft context visibility. Evaluate all four predictors and all eight fixed K
   in the same invocation. Overall decode timing includes KV reading, predictor
   and action-selection CPU/GPU synchronization. Detailed stage event profiling
   is optional and should be off for throughput comparisons. Native SGLang
   predictor inference/continuous batching is a later serving integration.

## Run CPU checks first

```bash
cd ~/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/Draft-OPD/Draft-OPD-joelulu
source ../.venv-draftopd/bin/activate
git pull --ff-only origin research/qwen3-4b-adaptive-length
bash scripts/adaptive_length/two_stage.sh test
```

Tests execute tiny real Qwen3/DFlash models, actual student forward/backward,
cache rollback, per-width rejected replay, dual labels, window isolation, model
freezing and a collect/label/train/eval round trip. Worker buffer/backend methods
are exercised on CPU via AST extraction, not CUDA attention kernels. These tests
**do not validate real CUDA SGLang/FSDP execution**. Run the GPU smoke below before
starting a long job. No server jobs are started by repository tests.

## Stage 1 GPU smoke (eight cards)

```bash
export RESULT_ROOT="$PWD/runs/two_stage_smoke_$(date +%Y%m%d_%H%M%S)"
export CKPT_DIR="$RESULT_ROOT/stage1"
export MAIN_MODEL_PATH=/home/jovyan/LMM/lmm_model/Qwen3-4B
export DRAFT_MODEL_PATH=/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STUDENT_WORLD_SIZE=8 \
ARM=mixed bash scripts/adaptive_length/two_stage.sh stage1-smoke
```

This runs 5 training batches with update accumulation 1, response budget 128,
training prompt batch 8, save at step 5 and no validation. Watch the per-K width
fractions, rejection supervision masks and checkpoint completion. An unsupported
or missing width record fails explicitly; it never falls back to native K=15.

Export draft weights without deleting any actor checkpoint:

```bash
python verl/scripts/extract_dflash_draft_from_fsdp.py \
  --actor-dir "$CKPT_DIR/global_step_5/actor" \
  --reference-draft-dir "$DRAFT_MODEL_PATH" \
  --target-dir "$RESULT_ROOT/draft" --dtype bfloat16
export DRAFT_MODEL_PATH="$RESULT_ROOT/draft"
```

## Stage 2 GPU smoke (one card)

Keep RESULT_ROOT and DRAFT_MODEL_PATH from the export above:

```bash
CUDA_VISIBLE_DEVICES=7 NPROC=1 \
bash scripts/adaptive_length/two_stage.sh collect-smoke

CUDA_VISIBLE_DEVICES=7 \
bash scripts/adaptive_length/two_stage.sh classifier-smoke

CUDA_VISIBLE_DEVICES=7 \
bash scripts/adaptive_length/two_stage.sh eval-smoke
```

The collector selects 16/8/8 prompt groups before prompt-length filtering, at most
2 states each, budget 128, and 3 timing repeats. Training runs one epoch. Evaluation
uses up to 4 test prompts. This checks execution, not research effectiveness.
The whole raw/labeled/classifier output must be fresh; reruns do not overwrite
existing trained weights or mix partial collections. Collection writes per-rank
completion markers and labeling requires every expected rank to finish.

Existing exported draft weights can also be used to check stage 2 independently:
set a fresh RESULT_ROOT and point DRAFT_MODEL_PATH at the chosen exported model.

## Formal runs

Stage 1 (choose a fresh checkpoint directory per arm):

```bash
export CKPT_DIR="$PWD/checkpoints/two_stage_mixed_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STUDENT_WORLD_SIZE=8 \
ARM=mixed NATIVE_OPD_BATCH_STEPS=200 UPDATE_ACCUMULATION_STEPS=1 \
TRAIN_PROMPT_BSZ=24 MAX_RESPONSE_LENGTH=1024 LR=3e-5 \
SAVE_FREQ=50 SAVE_START_STEP=50 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False \
bash scripts/adaptive_length/two_stage.sh train-draft
```

Use `ARM=fixed` for the same protocol with K=16 only. Start both arms from the
same initial draft. 200 batches are a pilot, not a convergence claim. Full data
remains the supplied 16,000-prompt file; original native hyperparameters govern
sampling/epochs unless overridden. Export the selected actor as above, replacing
step 5 with the chosen checkpoint step.

Stage 2:

```bash
export RESULT_ROOT="$PWD/runs/two_stage_formal_$(date +%Y%m%d_%H%M%S)"
export DRAFT_MODEL_PATH=/absolute/path/to/exported/draft

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
TRAIN_PROMPTS=2000 VAL_PROMPTS=256 TEST_PROMPTS=256 \
STATES_PER_PROMPT=16 MAX_NEW_TOKENS=512 \
bash scripts/adaptive_length/two_stage.sh collect

CUDA_VISIBLE_DEVICES=7 HEAD_EPOCHS=10 \
bash scripts/adaptive_length/two_stage.sh train-classifiers

CUDA_VISIBLE_DEVICES=7 EVAL_SAMPLES=256 MAX_NEW_TOKENS=512 \
bash scripts/adaptive_length/two_stage.sh eval
```

Do not copy the placeholder draft path without replacing it. Collection counts
are caps before prompt-length/EOS filtering. Report actual usable prompt/state
counts printed by labeling. With 32,000 training states and 3 timing repeats,
there are up to 768,000 timed train-state candidate verifications, plus warmup
and trajectory generation. Do not interpret state count as prompt count.

Collection defaults to deterministic prompt-hash train/val/test partitions for
predictor training. The draft may have trained on all of that file: this is a
**predictor holdout**, not a claim that the draft has never seen the prompts.
For an external test set, set `TEST_JSONL=/absolute/path/to/test.jsonl` during
collection; exact canonical prompt duplicates are excluded from train/val.
This is exact deduplication, not semantic/near-duplicate detection. Test files
support the existing `prompt`/`messages`/`question` JSONL schema.

The same frozen weights and generation setup must be used for labeling/training
and inference. Local config+weight bytes are SHA256-fingerprinted; changing the
checkpoint requires recollection. Classifier inference rejects overlap with its
training/validation prompts. Validation CE chooses each epoch checkpoint; final
throughput comparison uses only held-out test prompts, never the selection set.

## Outputs

- `raw/{train,val,test}.raw.rankNNN.jsonl`: prefixes, anchors, per-K acceptance,
  raw repeated stage timings, and split/source information.
- `labeled/{train,val,test}.jsonl`: original state measurements plus both K labels,
  speed scores and costs; `costs.json`, model manifest, and test prompt JSONL.
- `classifiers/{full_kv,window_16}_{accept,speed}.pt`: four predictors, weights,
  frozen-model fingerprints, training/validation prompt keys and validation stats.
- `eval_TIMESTAMP/results_rank000.jsonl`: per-prompt fixed/dynamic results,
  requested/actual K, acceptance, tail handling, decode time and match flag.
  Selector host-visible wall time and its fraction of decode time are also saved;
  this diagnostic includes synchronization and is not an isolated kernel time.
- `eval_TIMESTAMP/summary_rank000.csv`: pooled tokens/time; console compares
  predictors against the same draft's fastest fixed K.

AR token mismatches are recorded and do not abort. Speed labels do not establish
output quality or losslessness. Dynamic policies can visit different states than
the mixed-random collection policy; later data refresh can add policy trajectories.
