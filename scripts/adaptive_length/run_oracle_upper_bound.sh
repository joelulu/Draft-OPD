#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
RESULT_ROOT=${RESULT_ROOT:?Set RESULT_ROOT to the existing fixed_vs_dynamic run}
export PYTHONPATH="$ROOT/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1
CUDA_ROOT=${CUDA_ROOT:-/home/jovyan/LLM/cuda-12.9}
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
unset LD_PRELOAD
TORCH_CUDA_RT=$("$PYTHON_BIN" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$TORCH_CUDA_RT:$CUDA_ROOT/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUTPUT=${ORACLE_OUTPUT:-$RESULT_ROOT/oracle_upper_bound_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTPUT"
echo "Oracle output: $OUTPUT"
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}" "$PYTHON_BIN" scripts/adaptive_length/oracle_upper_bound.py \
    --target "${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}" \
    --fixed-draft "$RESULT_ROOT/models/fixed" --mixed-draft "$RESULT_ROOT/models/mixed" \
    --data "${EVAL_JSONL:-$RESULT_ROOT/labeled/test_prompts.jsonl}" --output "$OUTPUT" \
    --samples "${EVAL_SAMPLES:-16}" --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
    --max-prompt "${MAX_PROMPT:-512}" --repeats "${EVAL_REPEATS:-3}" \
    --search-repeats "${SEARCH_REPEATS:-3}" --seed 42 "$@" \
    2>&1 | tee "$OUTPUT/run.log"
