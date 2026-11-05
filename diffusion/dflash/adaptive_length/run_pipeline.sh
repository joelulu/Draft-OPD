#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
STAGE=${1:-all}
case "$STAGE" in all|collect|train|eval) ;; *) echo "usage: $0 [all|collect|train|eval]" >&2; exit 2;; esac
if [[ -z ${PYTHON_BIN:-} ]]; then
    source "$ROOT/.venv-draftopd/bin/activate"
    PYTHON_BIN=$(command -v python)
fi
CUDA_ROOT=${CUDA_ROOT:-/home/jovyan/LLM/cuda-12.9}
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
unset LD_PRELOAD
TORCH_CUDA_RT=$($PYTHON_BIN -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$TORCH_CUDA_RT:$CUDA_ROOT/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${GPU:-7}
export PYTHONPATH="$ROOT/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
TARGET_MODEL=${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16}
TRAIN_DATA=${TRAIN_JSONL:-$ROOT/data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/runs/adaptive_length_$(date +%Y%m%d_%H%M%S)}
COLLECT_SAMPLES=${COLLECT_SAMPLES:-128}
EVAL_SAMPLES=${EVAL_SAMPLES:-32}
COLLECT_OFFSET=${COLLECT_OFFSET:-0}
VAL_DATA=${VAL_JSONL:-$ROOT/data/opd_val_32.jsonl}
if [[ -f "$VAL_DATA" ]]; then
    EVAL_OFFSET=${EVAL_OFFSET:-0}
else
    VAL_DATA=$TRAIN_DATA
    EVAL_OFFSET=${EVAL_OFFSET:-$((COLLECT_OFFSET + COLLECT_SAMPLES))}
fi
mkdir -p "$RESULT_ROOT"
printf 'stage=%s\nresults=%s\npython=%s\ngpu=%s\n' "$STAGE" "$RESULT_ROOT" "$PYTHON_BIN" "$CUDA_VISIBLE_DEVICES"
COMMON=(--target "$TARGET_MODEL" --draft "$DRAFT_MODEL" --max-prompt "${MAX_PROMPT:-512}" --max-new-tokens "${MAX_NEW_TOKENS:-256}")
if [[ "$STAGE" == all || "$STAGE" == collect ]]; then
    test -d "$TARGET_MODEL"; test -d "$DRAFT_MODEL"; test -f "$TRAIN_DATA"
    "$PYTHON_BIN" -m adaptive_length.cli collect "${COMMON[@]}" --data "$TRAIN_DATA" \
        --samples "$COLLECT_SAMPLES" --offset "$COLLECT_OFFSET" --repeats "${TIMING_REPEATS:-2}" \
        --output "$RESULT_ROOT/paired" 2>&1 | tee "$RESULT_ROOT/collect.log"
fi
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
    "$PYTHON_BIN" -m adaptive_length.cli train --input "$RESULT_ROOT/paired" \
        --output "$RESULT_ROOT/head.pt" --epochs "${HEAD_EPOCHS:-30}" \
        --features "${HEAD_FEATURES:-both}" --device "${HEAD_DEVICE:-cpu}" \
        2>&1 | tee "$RESULT_ROOT/train.log"
fi
if [[ "$STAGE" == all || "$STAGE" == eval ]]; then
    test -f "$VAL_DATA"
    "$PYTHON_BIN" -m adaptive_length.cli eval "${COMMON[@]}" --data "$VAL_DATA" \
        --samples "$EVAL_SAMPLES" --offset "$EVAL_OFFSET" --head "$RESULT_ROOT/head.pt" \
        --all-fixed --output "$RESULT_ROOT/eval" 2>&1 | tee "$RESULT_ROOT/eval.log"
fi
