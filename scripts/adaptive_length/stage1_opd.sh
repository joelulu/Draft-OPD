#!/usr/bin/env bash
# Full variable-width native OPD: rollout, response training, rejection replay.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
ARM=${1:-mixed}
case "$ARM" in mixed) WIDTHS='[4,8,12,16,20,24,28,32]' ;; fixed) WIDTHS='[16]' ;; *) echo 'usage: stage1_opd.sh fixed|mixed [Hydra overrides]' >&2; exit 2 ;; esac
if (( $# )); then shift; fi
export NATIVE_MIXED_LENGTHS="$WIDTHS"
export NATIVE_ROLLOUT_MODE=matched-variable-width
export EXP_NAME=${EXP_NAME:-two-stage-opd-$ARM-$(date +%Y%m%d_%H%M%S)}
CONTEXT_CAP=$(( ${MAX_PROMPT:-512} + ${MAX_RESPONSE_LENGTH:-4096} + 33 ))
# Both arms use eager spec-v1 and maximum allocation 33. Actual forwards use K+1.
exec bash scripts/adaptive_length/native_opd.sh mixed \
  '++actor_rollout_ref.model.override_config.verl_dflash_match_rollout_widths=True' \
  "++actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_dflash_candidate_lengths=$WIDTHS" \
  '++actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_draft_tokens=33' \
  '++actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True' \
  'actor_rollout_ref.rollout.enforce_eager=True' \
  "actor_rollout_ref.rollout.max_model_len=$CONTEXT_CAP" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=$CONTEXT_CAP" \
  'trainer.logger=[console]' "$@"
