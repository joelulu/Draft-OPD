#!/usr/bin/env bash
# Fixed-vs-mixed OPD; no length-head training or policy is used.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
STAGE=${1:-all}
case "$STAGE" in all|train|eval|summarize) ;; *) echo "usage: $0 [all|train|eval|summarize]" >&2; exit 2;; esac
if [[ -z ${PYTHON_BIN:-} ]]; then
    if [[ -f .venv-draftopd/bin/activate ]]; then source .venv-draftopd/bin/activate;
    else source ../.venv-draftopd/bin/activate; fi
    export PYTHON_BIN=$(command -v python)
fi
export PYTHONPATH="$ROOT/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
export MAIN_MODEL_PATH=${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}
INITIAL_DRAFT=${DRAFT_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16}
export TRAIN_JSONL=${TRAIN_JSONL:-$ROOT/data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl}
if [[ ! -f "$TRAIN_JSONL" ]]; then
    export TRAIN_JSONL="$ROOT/../data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl"
fi
# Use the content-hash holdout from the same file, independently of the old head experiment.
export VAL_JSONL="$TRAIN_JSONL"
export EVAL_DATA_SPLIT=val
export NO_HEAD=1 STRICT_MATCH=0 PROFILE_ROUNDS=0
export LR=${LR:-3e-4}
export TRAIN_STEPS=${TRAIN_STEPS:-200}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-256}
export CANDIDATE_TOKENS_PER_UPDATE=${CANDIDATE_TOKENS_PER_UPDATE:-240}
export SAVE_FREQ=${SAVE_FREQ:-$TRAIN_STEPS}
export EVAL_SAMPLES=${EVAL_SAMPLES:-64}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
export NPROC=${NPROC:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU:-7}}
COMPARISON_ROOT=${RESULT_ROOT:-$ROOT/runs/length_training_compare_$(date +%Y%m%d_%H%M%S)}
if [[ "$STAGE" == eval || "$STAGE" == summarize ]]; then
    test -f "$COMPARISON_ROOT/comparison_plan.json"
fi
mkdir -p "$COMPARISON_ROOT"
export COMPARISON_ROOT INITIAL_DRAFT
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
    test -d "$MAIN_MODEL_PATH"; test -d "$INITIAL_DRAFT"; test -f "$TRAIN_JSONL"
    # Read the real native width rather than inferring it from a checkpoint name.
    FIXED_K=$("$PYTHON_BIN" -c 'import json,os,pathlib; print(int(json.loads((pathlib.Path(os.environ["INITIAL_DRAFT"])/"config.json").read_text())["block_size"])-1)')
    export FIXED_K
    "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
from adaptive_length.training import width_plan
import random
root = Path(os.environ["COMPARISON_ROOT"])
plan_path = root / "comparison_plan.json"
if plan_path.exists():
    raise FileExistsError("Use a new RESULT_ROOT for a new comparison; eval/summarize can reuse an existing one.")
fixed, budget = int(os.environ["FIXED_K"]), int(os.environ["CANDIDATE_TOKENS_PER_UPDATE"])
mixed = [4,8,12,16,24,32]
for lengths in ([fixed], mixed):
    width_plan(lengths, budget, random.Random(42))
plan_path.write_text(json.dumps(dict(initial_draft=os.environ["INITIAL_DRAFT"],
    fixed_k=fixed, mixed_lengths=mixed, steps=int(os.environ["TRAIN_STEPS"]),
    candidate_tokens_per_rank_per_update=budget, nproc=int(os.environ["NPROC"])), indent=2))
PY
    for ARM in fixed mixed; do
        if [[ "$ARM" == fixed ]]; then WIDTHS="$FIXED_K"; else WIDTHS="4 8 12 16 24 32"; fi
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU:-7}} \
        DRAFT_MODEL_PATH="$INITIAL_DRAFT" TRAIN_LENGTHS="$WIDTHS" \
        RESULT_ROOT="$COMPARISON_ROOT/$ARM" \
        bash scripts/adaptive_length/run_pipeline.sh train-draft
    done
fi
if [[ "$STAGE" == all || "$STAGE" == eval ]]; then
    # Resolve initial checkpoint and final training step from the saved experiment plan.
    INITIAL_DRAFT=$("$PYTHON_BIN" -c 'import json,os,pathlib; print(json.loads((pathlib.Path(os.environ["COMPARISON_ROOT"])/"comparison_plan.json").read_text())["initial_draft"])')
    STEPS=$("$PYTHON_BIN" -c 'import json,os,pathlib; print(json.loads((pathlib.Path(os.environ["COMPARISON_ROOT"])/"comparison_plan.json").read_text())["steps"])')
    printf -v CHECKPOINT 'step_%06d' "$STEPS"
    for ARM in initial fixed mixed; do
        if [[ "$ARM" == initial ]]; then MODEL="$INITIAL_DRAFT";
        else MODEL="$COMPARISON_ROOT/$ARM/draft/$CHECKPOINT"; fi
        test -d "$MODEL"
        CUDA_VISIBLE_DEVICES=${EVAL_GPU:-${CUDA_VISIBLE_DEVICES%%,*}} \
        NPROC=1 DRAFT_MODEL_PATH="$MODEL" RESULT_ROOT="$COMPARISON_ROOT/$ARM" \
        EVAL_OUTPUT="$COMPARISON_ROOT/$ARM/eval_fixed" \
        bash scripts/adaptive_length/run_pipeline.sh eval
    done
fi
if [[ "$STAGE" == all || "$STAGE" == eval || "$STAGE" == summarize ]]; then
    "$PYTHON_BIN" -m adaptive_length.compare_training --root "$COMPARISON_ROOT"
fi
printf 'Comparison: %s\n' "$COMPARISON_ROOT"
