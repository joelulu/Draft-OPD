#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
: "${MODEL_ROOT:?Set MODEL_ROOT to the previous evaluation root containing models/fixed and models/mixed}"
: "${VAL_JSONL:?Set VAL_JSONL to the chosen evaluation JSONL}"
test -f "$VAL_JSONL"
MODEL_ROOT=$(cd "$MODEL_ROOT" && pwd)
VAL_JSONL=$(realpath "$VAL_JSONL")
INITIAL_DRAFT=${INITIAL_DRAFT:-/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16}
for MODEL in "$INITIAL_DRAFT" "$MODEL_ROOT/models/fixed" "$MODEL_ROOT/models/mixed"; do test -f "$MODEL/config.json"; done
if [[ -z ${PYTHON_BIN:-} ]]; then
    if [[ -f .venv-draftopd/bin/activate ]]; then source .venv-draftopd/bin/activate; else source ../.venv-draftopd/bin/activate; fi
    PYTHON_BIN=$(command -v python)
fi
OUT_ROOT=${OUT_ROOT:-$ROOT/runs/native_opd_repeat_$(date +%Y%m%d_%H%M%S)}
REPEATS=${REPEATS:-3}
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'REPEATS must be positive' >&2; exit 2; }
# New output only: never mix fresh runs with stale/partial results.
mkdir "$OUT_ROOT"
OUT_ROOT=$(cd "$OUT_ROOT" && pwd)
"$PYTHON_BIN" - "$OUT_ROOT" "$MODEL_ROOT" "$VAL_JSONL" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, models, data = sys.argv[1:]
Path(out, 'inputs.json').write_text(json.dumps(dict(model_root=models, data=data,
    data_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest()), indent=2))
PY
for (( REP=1; REP<=REPEATS; REP++ )); do
    if (( REP % 2 )); then ARMS=(fixed mixed initial); else ARMS=(mixed fixed initial); fi
    for ARM in "${ARMS[@]}"; do
        MODEL="$MODEL_ROOT/models/$ARM"
        if [[ "$ARM" == initial ]]; then MODEL="$INITIAL_DRAFT"; fi
        echo "repeat=$REP arm=$ARM"
        PYTHON_BIN="$PYTHON_BIN" CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}" NPROC=1 \
        NO_HEAD=1 STRICT_MATCH=0 PROFILE_ROUNDS=0 EVAL_DATA_SPLIT=all \
        EVAL_SAMPLES="${EVAL_SAMPLES:-32}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
        MAX_PROMPT="${MAX_PROMPT:-512}" VAL_JSONL="$VAL_JSONL" DRAFT_MODEL_PATH="$MODEL" \
        RESULT_ROOT="$OUT_ROOT/repeat_$REP/$ARM" EVAL_OUTPUT="$OUT_ROOT/repeat_$REP/$ARM/eval" \
        bash scripts/adaptive_length/run_pipeline.sh eval
    done
done
"$PYTHON_BIN" scripts/adaptive_length/analyze_native_eval.py --root "$OUT_ROOT"
