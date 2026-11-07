#!/usr/bin/env bash
# End-to-end workflow, or restart from existing exported drafts and labeled data.
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODE=${1:-all}
case "$MODE" in all|resume|classifiers|eval|status) ;; *) echo 'usage: run_fixed_vs_dynamic.sh all|resume|classifiers|eval|status' >&2; exit 2 ;; esac
if [[ -z ${PYTHON_BIN:-} ]]; then
    if [[ -f .venv-draftopd/bin/activate ]]; then source .venv-draftopd/bin/activate; else source ../.venv-draftopd/bin/activate; fi
    PYTHON_BIN=$(command -v python)
fi
if [[ -z ${RESULT_ROOT:-} ]]; then
    if [[ "$MODE" != all ]]; then echo 'Set RESULT_ROOT to the existing experiment directory' >&2; exit 2; fi
    export RESULT_ROOT="$ROOT/runs/fixed_vs_dynamic_$(date +%Y%m%d_%H%M%S)"
fi
export RESULT_ROOT=$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RESULT_ROOT")
CLASSIFIER_DIR=${CLASSIFIER_DIR:-$RESULT_ROOT/classifiers_ddp}
if [[ "$MODE" == status ]]; then
    exec "$PYTHON_BIN" scripts/adaptive_length/workflow_tools.py status --root "$RESULT_ROOT" --classifiers "$CLASSIFIER_DIR"
fi
mkdir -p "$RESULT_ROOT"
# Avoid launching two independent optimizers against the same output directory.
exec 9>"$RESULT_ROOT/.workflow.lock"
flock -n 9 || { echo 'Another workflow holds this experiment lock' >&2; exit 2; }
STAGE=setup
mark() { STAGE=$1; "$PYTHON_BIN" scripts/adaptive_length/workflow_tools.py mark --root "$RESULT_ROOT" --stage "$STAGE"; }
finish() {
    local code=$1
    trap - EXIT
    set +e
    local state=done
    if (( code != 0 )); then state=stopped_or_failed; fi
    "$PYTHON_BIN" scripts/adaptive_length/workflow_tools.py mark --root "$RESULT_ROOT" --stage "$STAGE" --status "$state"
    exit "$code"
}
trap 'finish "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export MAIN_MODEL_PATH=${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}
export INITIAL_DRAFT=${INITIAL_DRAFT:-/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16}
export TRAIN_JSONL=${TRAIN_JSONL:-$ROOT/../data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl}
export PYTHONPATH="$ROOT/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=${CUDA_HOME:-/home/jovyan/LLM/cuda-12.9}
export PATH="$CUDA_HOME/bin:$PATH"
unset LD_PRELOAD
CUDA_RT=$("$PYTHON_BIN" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$CUDA_RT:$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
CLASSIFIER_GPUS=${CLASSIFIER_GPUS:-0,1,2,3,4,5,6,7}
IFS=, read -r -a GPU_LIST <<< "$CLASSIFIER_GPUS"
CLASSIFIER_NPROC=${CLASSIFIER_NPROC:-${#GPU_LIST[@]}}
GLOBAL_BATCH=${CLASSIFIER_GLOBAL_BATCH:-16}
[[ "$CLASSIFIER_NPROC" =~ ^[1-9][0-9]*$ && "$GLOBAL_BATCH" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid classifier GPU/batch setting' >&2; exit 2; }
(( CLASSIFIER_NPROC <= ${#GPU_LIST[@]} && GLOBAL_BATCH % CLASSIFIER_NPROC == 0 )) || { echo 'Global batch must divide by NPROC, with enough visible GPUs' >&2; exit 2; }
ACCUMULATE=$((GLOBAL_BATCH/CLASSIFIER_NPROC))
STEPS=${DRAFT_TRAIN_STEPS:-1000}
STAMP=$(date +%Y%m%d_%H%M%S)
if [[ "$MODE" == all || "$MODE" == resume ]]; then
    for ARM in fixed mixed; do
        if [[ -f "$RESULT_ROOT/models/$ARM/model.safetensors" && -f "$RESULT_ROOT/models/$ARM/config.json" ]]; then
            echo "Reuse exported $ARM draft"
            continue
        fi
        ACTOR="$RESULT_ROOT/checkpoints/$ARM/global_step_$STEPS/actor"
        if [[ ! -f "$ACTOR/fsdp_config.json" ]]; then
            mark "train_$ARM"
            CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" STUDENT_WORLD_SIZE="${STUDENT_WORLD_SIZE:-8}" \
            DRAFT_MODEL_PATH="$INITIAL_DRAFT" ARM="$ARM" \
            EXP_NAME="two-stage-${ARM}-$(basename "$RESULT_ROOT")" CKPT_DIR="$RESULT_ROOT/checkpoints/$ARM" \
            NATIVE_OPD_BATCH_STEPS="$STEPS" UPDATE_ACCUMULATION_STEPS=1 \
            TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-24}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}" LR="${LR:-3e-5}" \
            SAVE_FREQ=250 SAVE_START_STEP=250 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False \
            bash scripts/adaptive_length/two_stage.sh train-draft 2>&1 | tee "$RESULT_ROOT/train_${ARM}_${STAMP}.log"
        fi
        mark "export_$ARM"
        "$PYTHON_BIN" verl/scripts/extract_dflash_draft_from_fsdp.py --actor-dir "$ACTOR" \
            --reference-draft-dir "$INITIAL_DRAFT" --target-dir "$RESULT_ROOT/models/$ARM" --dtype bfloat16 \
            2>&1 | tee "$RESULT_ROOT/export_${ARM}_${STAMP}.log"
    done
    if [[ ! -f "$RESULT_ROOT/labeled/test_prompts.jsonl" || ! -f "$RESULT_ROOT/labeled/train.jsonl" || ! -f "$RESULT_ROOT/labeled/val.jsonl" ]]; then
        mark collect
        if [[ -d "$RESULT_ROOT/raw" ]]; then mv "$RESULT_ROOT/raw" "$RESULT_ROOT/raw_previous_$STAMP"; fi
        if [[ -d "$RESULT_ROOT/labeled" ]]; then mv "$RESULT_ROOT/labeled" "$RESULT_ROOT/labeled_previous_$STAMP"; fi
        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC="${COLLECT_NPROC:-8}" DRAFT_MODEL_PATH="$RESULT_ROOT/models/mixed" \
        bash scripts/adaptive_length/two_stage.sh collect 2>&1 | tee "$RESULT_ROOT/collect_${STAMP}.log"
    else
        echo 'Reuse completed labeled states'
    fi
fi
for MODEL in fixed mixed; do test -f "$RESULT_ROOT/models/$MODEL/config.json"; done
for FILE in train.jsonl val.jsonl test_prompts.jsonl manifest.json; do test -f "$RESULT_ROOT/labeled/$FILE"; done
if [[ "$MODE" != eval ]]; then
    mark classifiers
    LAUNCH=("$PYTHON_BIN")
    if (( CLASSIFIER_NPROC > 1 )); then LAUNCH+=(-m torch.distributed.run --standalone --nproc_per_node "$CLASSIFIER_NPROC"); fi
    # Fresh directory for the first v2 run; original classifiers/ and logs are preserved.
    CUDA_VISIBLE_DEVICES="$CLASSIFIER_GPUS" "${LAUNCH[@]}" -m adaptive_length.two_stage.distributed_train \
        --target "$MAIN_MODEL_PATH" --draft "$RESULT_ROOT/models/mixed" \
        --data "$RESULT_ROOT/labeled" --output "$CLASSIFIER_DIR" \
        --epochs "${HEAD_EPOCHS:-10}" --accumulate "$ACCUMULATE" --lr "${CLASSIFIER_LR:-1e-4}" \
        --log-every "${CLASSIFIER_LOG_EVERY:-50}" --log-seconds "${CLASSIFIER_LOG_SECONDS:-15}" \
        --save-every "${CLASSIFIER_SAVE_EVERY:-100}" --resume \
        2>&1 | tee -a "$RESULT_ROOT/classifiers_ddp.log"
    if [[ "$MODE" == classifiers ]]; then exit 0; fi
fi
CHECKPOINTS=("$CLASSIFIER_DIR/full_kv_accept.pt" "$CLASSIFIER_DIR/full_kv_speed.pt" "$CLASSIFIER_DIR/window_16_accept.pt" "$CLASSIFIER_DIR/window_16_speed.pt")
for FILE in "${CHECKPOINTS[@]}"; do test -f "$FILE"; done
EVAL_DATA=${EVAL_JSONL:-$RESULT_ROOT/labeled/test_prompts.jsonl}
for ARM in baseline mixed; do
    mark "eval_$ARM"
    OUT="$RESULT_ROOT/evaluation/$ARM"
    # Re-evaluate after classifier training; previous measurements are preserved.
    if [[ -d "$OUT" ]]; then mv "$OUT" "${OUT}_previous_$STAMP"; fi
    FLAGS=()
    MODEL="$RESULT_ROOT/models/fixed"
    if [[ "$ARM" == mixed ]]; then
        MODEL="$RESULT_ROOT/models/mixed"
        FLAGS+=(--classifier "${CHECKPOINTS[@]}")
    fi
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}" "$PYTHON_BIN" -m adaptive_length.pipeline eval \
        --target "$MAIN_MODEL_PATH" --draft "$MODEL" --data "$EVAL_DATA" --output "$OUT" \
        --split all --max-samples "${EVAL_SAMPLES:-256}" --max-prompt "${MAX_PROMPT:-512}" \
        --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-512}" --seed 42 \
        --lengths 4 8 12 16 20 24 28 32 --baseline-lengths 16 --exact-lengths \
        --progress-json "$OUT/progress.json" "${FLAGS[@]}" \
        2>&1 | tee "$RESULT_ROOT/evaluation_${ARM}_${STAMP}.log"
done
mark summarize
"$PYTHON_BIN" scripts/adaptive_length/workflow_tools.py summarize --root "$RESULT_ROOT" | tee "$RESULT_ROOT/final_comparison.txt"
echo "Complete: $RESULT_ROOT/final_comparison.csv"
