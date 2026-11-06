#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
STAGE=${1:-all}
case "$STAGE" in all|train-draft|collect|train-head|eval) ;; *) echo "usage: $0 [all|train-draft|collect|train-head|eval]" >&2; exit 2;; esac
if [[ -z ${PYTHON_BIN:-} ]]; then
    if [[ -f "$ROOT/.venv-draftopd/bin/activate" ]]; then
        source "$ROOT/.venv-draftopd/bin/activate"
    else
        source "$ROOT/../.venv-draftopd/bin/activate"
    fi
    PYTHON_BIN=$(command -v python)
fi
CUDA_ROOT=${CUDA_ROOT:-/home/jovyan/LLM/cuda-12.9}
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
unset LD_PRELOAD
TORCH_CUDA_RT=$("$PYTHON_BIN" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$TORCH_CUDA_RT:$CUDA_ROOT/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU:-7}}
export PYTHONPATH="$ROOT/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
TARGET_MODEL=${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16}
TRAIN_DATA=${TRAIN_JSONL:-$ROOT/data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/runs/adaptive_length}
NPROC=${NPROC:-1}
mkdir -p "$RESULT_ROOT"
printf 'stage=%s\nresults=%s\npython=%s\ngpu=%s\n' "$STAGE" "$RESULT_ROOT" "$PYTHON_BIN" "$CUDA_VISIBLE_DEVICES"
COMMON=(--target "$TARGET_MODEL" --draft "$DRAFT_MODEL" --max-prompt "${MAX_PROMPT:-512}" --max-new-tokens "${MAX_NEW_TOKENS:-256}")
# Each collector rank is an independent B=1 replica; draft training uses DDP.
LAUNCH=("$PYTHON_BIN")
if (( NPROC > 1 )); then LAUNCH+=( -m torch.distributed.run --standalone --nproc_per_node "$NPROC" ); fi
if [[ "$STAGE" == train-draft ]]; then
    read -r -a TRAIN_WIDTHS <<< "${TRAIN_LENGTHS:-4 8 12 16 24 32}"
    "${LAUNCH[@]}" -m adaptive_length.pipeline train-draft "${COMMON[@]}" --data "$TRAIN_DATA" \
        --max-samples "${TRAIN_SAMPLES:-0}" --steps "${TRAIN_STEPS:-1000}" \
        --lengths "${TRAIN_WIDTHS[@]}" --candidate-tokens-per-update "${CANDIDATE_TOKENS_PER_UPDATE:-0}" \
        --lr "${LR:-3e-4}" --save-every "${SAVE_FREQ:-250}" --output "$RESULT_ROOT/draft" \
        2>&1 | tee "$RESULT_ROOT/train_draft.log"
fi
if [[ "$STAGE" == all || "$STAGE" == collect ]]; then
    "${LAUNCH[@]}" -m adaptive_length.pipeline collect "${COMMON[@]}" --data "$TRAIN_DATA" \
        --max-samples "${COLLECT_SAMPLES:-128}" --max-rounds "${MAX_ROUNDS:-32}" \
        --output "$RESULT_ROOT/paired" 2>&1 | tee "$RESULT_ROOT/collect.log"
fi
if [[ "$STAGE" == all || "$STAGE" == train-head ]]; then
    "$PYTHON_BIN" -m adaptive_length.pipeline train-head --labels "$RESULT_ROOT/paired" \
        --output "$RESULT_ROOT/head.pt" --epochs "${HEAD_EPOCHS:-30}" \
        --features "${HEAD_FEATURES:-both}" --device "${HEAD_DEVICE:-cpu}" \
        2>&1 | tee "$RESULT_ROOT/train_head.log"
fi
if [[ "$STAGE" == all || "$STAGE" == eval ]]; then
    # Prefer user's separate validation file. Otherwise take held-out prompt groups.
    VAL_DATA=${VAL_JSONL:-$ROOT/data/opd_val_32.jsonl}
    EVAL_SPLIT=all
    if [[ ! -f "$VAL_DATA" ]]; then VAL_DATA=$TRAIN_DATA; EVAL_SPLIT=val; fi
    EVAL_FLAGS=()
    if [[ ${STRICT_MATCH:-0} == 1 ]]; then EVAL_FLAGS+=(--strict-match); fi
    if [[ ${PROFILE_ROUNDS:-0} == 1 ]]; then EVAL_FLAGS+=(--profile-rounds); fi
    if [[ ${NO_HEAD:-0} != 1 ]]; then
        HEAD_PATH=${HEAD_PATH:-$RESULT_ROOT/head.pt}
        EVAL_FLAGS+=(--head "$HEAD_PATH")
    fi
    EVAL_SPLIT=${EVAL_DATA_SPLIT:-$EVAL_SPLIT}
    EVAL_OUTPUT=${EVAL_OUTPUT:-$RESULT_ROOT/eval_v2}
    "$PYTHON_BIN" -m adaptive_length.pipeline eval "${COMMON[@]}" --data "$VAL_DATA" \
        --split "$EVAL_SPLIT" --max-samples "${EVAL_SAMPLES:-32}" \
        "${EVAL_FLAGS[@]}" --output "$EVAL_OUTPUT" 2>&1 | tee "$RESULT_ROOT/eval.log"
fi
