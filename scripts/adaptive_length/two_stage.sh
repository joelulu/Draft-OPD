#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
STAGE=${1:-help}
if (( $# )); then shift; fi
case "$STAGE" in
train-draft|stage1-smoke)
    if [[ "$STAGE" == stage1-smoke ]]; then
        export NATIVE_OPD_BATCH_STEPS=${NATIVE_OPD_BATCH_STEPS:-5}
        export UPDATE_ACCUMULATION_STEPS=1 TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-8}
        export MAX_RESPONSE_LENGTH=128 SAVE_FREQ=5 SAVE_START_STEP=5 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False
    fi
    exec bash scripts/adaptive_length/stage1_opd.sh "${ARM:-mixed}" "$@" ;;
collect|collect-smoke|label|train-classifiers|classifier-smoke|eval|eval-smoke|test) ;;
*) echo 'usage: two_stage.sh train-draft|stage1-smoke|collect|collect-smoke|label|train-classifiers|classifier-smoke|eval|eval-smoke|test' >&2; exit 2 ;;
esac
if [[ -z ${PYTHON_BIN:-} ]]; then
    if [[ -f .venv-draftopd/bin/activate ]]; then source .venv-draftopd/bin/activate; else source ../.venv-draftopd/bin/activate; fi
    PYTHON_BIN=$(command -v python)
fi
export PYTHONPATH="$ROOT/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$STAGE" == test ]]; then
    exec "$PYTHON_BIN" -m pytest -q diffusion/dflash/tests/test_two_stage.py diffusion/dflash/tests/test_native_mixed_opd.py diffusion/dflash/tests/test_variable_rollout.py
fi
: "${RESULT_ROOT:?Set RESULT_ROOT to a persistent experiment directory}"
RESULT_ROOT=$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RESULT_ROOT")
if [[ "$STAGE" == label ]]; then
    exec "$PYTHON_BIN" -m adaptive_length.two_stage.cli label --data "$RESULT_ROOT/raw" --output "$RESULT_ROOT/labeled" "$@"
fi
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH to the exported frozen draft checkpoint}"
TARGET=${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}
DATA=${TRAIN_JSONL:-$ROOT/../data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl}
CUDA_ROOT=${CUDA_ROOT:-/home/jovyan/LLM/cuda-12.9}
export CUDA_HOME="$CUDA_ROOT" PATH="$CUDA_ROOT/bin:$PATH"
unset LD_PRELOAD
CUDA_RT=$("$PYTHON_BIN" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$CUDA_RT:$CUDA_ROOT/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
COMMON=(--target "$TARGET" --draft "$DRAFT_MODEL_PATH")
if [[ "$STAGE" == collect || "$STAGE" == collect-smoke ]]; then
    FLAGS=()
    if [[ -n ${TEST_JSONL:-} ]]; then FLAGS+=(--test-data "$TEST_JSONL"); fi
    if [[ "$STAGE" == collect-smoke ]]; then
        export TRAIN_PROMPTS=16 VAL_PROMPTS=8 TEST_PROMPTS=8 STATES_PER_PROMPT=2 MAX_NEW_TOKENS=128
    fi
    LAUNCH=("$PYTHON_BIN")
    if (( ${NPROC:-1} > 1 )); then LAUNCH+=(-m torch.distributed.run --standalone --nproc_per_node "$NPROC"); fi
    "${LAUNCH[@]}" -m adaptive_length.two_stage.cli collect "${COMMON[@]}" \
        --data "$DATA" --output "$RESULT_ROOT/raw" \
        --train-prompts "${TRAIN_PROMPTS:-2000}" --val-prompts "${VAL_PROMPTS:-256}" --test-prompts "${TEST_PROMPTS:-256}" \
        --states-per-prompt "${STATES_PER_PROMPT:-16}" --state-stride "${STATE_STRIDE:-16}" \
        --max-prompt "${MAX_PROMPT:-512}" --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
        --timing-repeats "${TIMING_REPEATS:-3}" "${FLAGS[@]}" "$@"
    exec "$PYTHON_BIN" -m adaptive_length.two_stage.cli label --data "$RESULT_ROOT/raw" --output "$RESULT_ROOT/labeled"
fi
if [[ "$STAGE" == train-classifiers || "$STAGE" == classifier-smoke ]]; then
    if [[ "$STAGE" == classifier-smoke ]]; then export HEAD_EPOCHS=1; fi
    exec "$PYTHON_BIN" -m adaptive_length.two_stage.cli train "${COMMON[@]}" \
        --data "$RESULT_ROOT/labeled" --output "$RESULT_ROOT/classifiers" \
        --epochs "${HEAD_EPOCHS:-10}" --accumulate "${CLASSIFIER_ACCUMULATE:-16}" --lr "${CLASSIFIER_LR:-1e-4}" \
        --log-every "${CLASSIFIER_LOG_EVERY:-100}" --log-seconds "${CLASSIFIER_LOG_SECONDS:-30}" "$@"
fi
if [[ "$STAGE" == eval || "$STAGE" == eval-smoke ]]; then
    if [[ "$STAGE" == eval-smoke ]]; then export EVAL_SAMPLES=4 MAX_NEW_TOKENS=128; fi
    CHECKPOINTS=("$RESULT_ROOT/classifiers/full_kv_accept.pt" "$RESULT_ROOT/classifiers/full_kv_speed.pt"
                 "$RESULT_ROOT/classifiers/window_16_accept.pt" "$RESULT_ROOT/classifiers/window_16_speed.pt")
    for path in "${CHECKPOINTS[@]}"; do test -f "$path"; done
    EVAL_DIR="$RESULT_ROOT/eval_$(date +%Y%m%d_%H%M%S)"
    "$PYTHON_BIN" -m adaptive_length.pipeline eval "${COMMON[@]}" \
        --data "${EVAL_JSONL:-$RESULT_ROOT/labeled/test_prompts.jsonl}" --output "$EVAL_DIR" \
        --split all --max-samples "${EVAL_SAMPLES:-256}" --max-prompt "${MAX_PROMPT:-512}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-512}" --lengths 4 8 12 16 20 24 28 32 --exact-lengths \
        --classifier "${CHECKPOINTS[@]}" "$@"
    exec "$PYTHON_BIN" scripts/adaptive_length/compare_classifiers.py --root "$EVAL_DIR"
fi
