#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1
CUDA_ROOT=${CUDA_ROOT:-/home/jovyan/LLM/cuda-12.9}
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
unset LD_PRELOAD
TORCH_CUDA_RT=$("$PYTHON_BIN" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$TORCH_CUDA_RT:$CUDA_ROOT/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}" "$PYTHON_BIN" scripts/adaptive_length/export_top5_matrix.py "$@"
