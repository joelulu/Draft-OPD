#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export RESULT_ROOT=${RESULT_ROOT:?Set RESULT_ROOT to the existing fixed_vs_dynamic run}
export EVAL_SAMPLES=${EVAL_SAMPLES:-8}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
export EVAL_REPEATS=${EVAL_REPEATS:-3}
export SEARCH_REPEATS=${SEARCH_REPEATS:-2}
export ORACLE_OUTPUT=${ORACLE_OUTPUT:-$RESULT_ROOT/gsm8k_oracle_$(date +%Y%m%d_%H%M%S)}
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1
PREPARE=()
if [[ -n ${GSM8K_TEST_JSONL:-} ]]; then PREPARE+=(--local-test-jsonl "$GSM8K_TEST_JSONL"); fi
"$PYTHON_BIN" scripts/adaptive_length/prepare_gsm8k_oracle.py \
    --output "$ORACLE_OUTPUT/gsm8k_test.jsonl" --samples "$EVAL_SAMPLES" "${PREPARE[@]}"
export EVAL_JSONL="$ORACLE_OUTPUT/gsm8k_test.jsonl"
exec bash scripts/adaptive_length/run_oracle_upper_bound.sh --simple --print-rounds --no-profile "$@"
