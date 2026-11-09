#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
RESULT_ROOT=${RESULT_ROOT:?Set RESULT_ROOT to fixed_vs_dynamic run}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT=${CONFUSED_OUTPUT:-$RESULT_ROOT/confused_token_$(date +%Y%m%d_%H%M%S)}
export PYTHONPATH="$ROOT/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CUDA_ROOT=${CUDA_ROOT:-/home/jovyan/LLM/cuda-12.9}
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
unset LD_PRELOAD
TORCH_CUDA_RT=$("$PYTHON_BIN" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$TORCH_CUDA_RT:$CUDA_ROOT/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Python creates OUTPUT exclusively; tee log beside it, never pre-create OUTPUT.
mkdir -p "$(dirname "$OUTPUT")"
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}" "$PYTHON_BIN" scripts/adaptive_length/confused_token_audit.py \
 --target "${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}" \
 --fixed-draft "${FIXED_DRAFT_PATH:-$RESULT_ROOT/models/fixed}" \
 --data "${GSM8K_TEST_JSONL:-$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/eval_datasets/gsm8k.jsonl}" \
 --samples "${EVAL_SAMPLES:-16}" --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
 --output "$OUTPUT" "$@" 2>&1 | tee "$OUTPUT.log"
