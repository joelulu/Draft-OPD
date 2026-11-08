#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
RESULT_ROOT=${RESULT_ROOT:?Set RESULT_ROOT to the existing fixed_vs_dynamic run}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT=${SWEEP_OUTPUT:-$RESULT_ROOT/two_dataset_k_sweep_$(date +%Y%m%d_%H%M%S)}
SAMPLES=${EVAL_SAMPLES:-16}
GSM8K_LOCAL=${GSM8K_TEST_JSONL:-$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/eval_datasets/gsm8k.jsonl}
MIXED_DATA=${MIXED_TEST_JSONL:-$RESULT_ROOT/labeled/test_prompts.jsonl}
test -f "$GSM8K_LOCAL"
test -f "$MIXED_DATA"
"$PYTHON_BIN" -c 'import matplotlib' || { echo 'Install plotting dependency: python -m pip install matplotlib' >&2; exit 1; }
mkdir -p "$OUTPUT"
"$PYTHON_BIN" scripts/adaptive_length/prepare_gsm8k_oracle.py \
    --local-test-jsonl "$GSM8K_LOCAL" --samples "$SAMPLES" --output "$OUTPUT/gsm8k_test.jsonl"
export PYTHONPATH="$ROOT/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CUDA_ROOT=${CUDA_ROOT:-/home/jovyan/LLM/cuda-12.9}
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
unset LD_PRELOAD
TORCH_CUDA_RT=$("$PYTHON_BIN" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$TORCH_CUDA_RT:$CUDA_ROOT/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_dataset() {
    local NAME=$1 GPU=$2 DATA=$3
    mkdir -p "$OUTPUT/$NAME"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/adaptive_length/benchmark_k_sweep.py \
        --target "${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}" \
        --fixed-draft "$RESULT_ROOT/models/fixed" --mixed-draft "$RESULT_ROOT/models/mixed" \
        --data "$DATA" --dataset "$NAME" --output "$OUTPUT/$NAME" \
        --samples "$SAMPLES" --max-new-tokens "${MAX_NEW_TOKENS:-512}" --max-prompt "${MAX_PROMPT:-512}" \
        --repeats "${EVAL_REPEATS:-3}" --probe-states "${PROBE_STATES:-4}" \
        --probe-stride "${PROBE_STRIDE:-64}" --probe-repeats "${PROBE_REPEATS:-2}" \
        2>&1 | tee "$OUTPUT/$NAME/run.log"
}
IFS=',' read -r -a GPUS <<< "${SWEEP_GPUS:-${EVAL_GPU:-7}}"
echo "Results: $OUTPUT; datasets: mixed_holdout and local GSM8K; GPUs: ${GPUS[*]}"
if (( ${#GPUS[@]} >= 2 )); then
    run_dataset mixed_holdout "${GPUS[0]}" "$MIXED_DATA" &
    PID_MIXED=$!
    run_dataset gsm8k "${GPUS[1]}" "$OUTPUT/gsm8k_test.jsonl" &
    PID_GSM8K=$!
    EXIT_STATUS=0
    wait "$PID_MIXED" || EXIT_STATUS=1
    wait "$PID_GSM8K" || EXIT_STATUS=1
    if (( EXIT_STATUS )); then echo "A dataset failed; inspect logs under $OUTPUT" >&2; exit 1; fi
else
    run_dataset mixed_holdout "${GPUS[0]}" "$MIXED_DATA"
    run_dataset gsm8k "${GPUS[0]}" "$OUTPUT/gsm8k_test.jsonl"
fi
"$PYTHON_BIN" scripts/adaptive_length/report_k_sweep.py --root "$OUTPUT" 2>&1 | tee "$OUTPUT/summary.txt"
