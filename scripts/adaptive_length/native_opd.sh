#!/usr/bin/env bash
# Original verl/FSDP OPD, optional mixed RESPONSE training widths.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ARM=${1:-mixed}
case "$ARM" in fixed|mixed) shift || true ;; *) echo "usage: $0 [fixed|mixed] [Hydra overrides...]" >&2; exit 2;; esac
cd "$ROOT"
if [[ -z ${PYTHON_BIN:-} ]]; then
    if [[ -f .venv-draftopd/bin/activate ]]; then source .venv-draftopd/bin/activate;
    else source ../.venv-draftopd/bin/activate; fi
    export PYTHON_BIN=$(command -v python)
fi
# Upstream launches python3; keep it in the same venv as the preflight interpreter.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT/verl:$ROOT/sglang-dflash/python${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export WANDB_ENTITY=${WANDB_ENTITY:-joelulu-}
export MAIN_MODEL_PATH=${MAIN_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$MAIN_MODEL_PATH}
export DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16}
export TRAIN_JSONL=${TRAIN_JSONL:-$ROOT/data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl}
if [[ ! -f "$TRAIN_JSONL" ]]; then
    export TRAIN_JSONL="$ROOT/../data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl"
fi
VAL_JSONL=${VAL_JSONL:-$ROOT/../data/opd_val_32.jsonl}
CUDA_ROOT=${CUDA_ROOT:-/home/jovyan/LLM/cuda-12.9}
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
unset LD_PRELOAD
TORCH_CUDA_RT=$("$PYTHON_BIN" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="$TORCH_CUDA_RT:$CUDA_ROOT/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export STUDENT_WORLD_SIZE=${STUDENT_WORLD_SIZE:-8}
export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-24}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export MAX_PROMPT=${MAX_PROMPT:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export LR=${LR:-3e-5}
export train_epochs=${train_epochs:-8}
export TEST_FREQ=${TEST_FREQ:-250}
export SAVE_FREQ=${SAVE_FREQ:-500}
export SAVE_START_STEP=${SAVE_START_STEP:-500}
export ENABLE_THINKING=False
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.2}
export RUN_GPU_STRESS_TEST_ON_EXIT=False
export RESET_RAY_RUNTIME=False
export EXP_NAME=${EXP_NAME:-qwen3-4b-native-opd-$ARM-$(date +%Y%m%d_%H%M%S)}
export CKPT_DIR=${CKPT_DIR:-$ROOT/checkpoints/$EXP_NAME}
# Empty config is the unmodified native-width response/rejected stream path.
WIDTH_CONFIG='[]'
if [[ "$ARM" == mixed ]]; then WIDTH_CONFIG=${NATIVE_MIXED_LENGTHS:-'[4,8,12,16,24,32]'}; fi
EXTRA=()
if [[ -n ${NATIVE_OPD_BATCH_STEPS:-} ]]; then
    EXTRA+=("trainer.total_training_steps=$NATIVE_OPD_BATCH_STEPS")
fi
for path in "$MAIN_MODEL_PATH" "$TEACHER_MODEL_PATH" "$DRAFT_MODEL_PATH"; do test -d "$path"; done
test -f "$TRAIN_JSONL"; test -f "$VAL_JSONL"
"$PYTHON_BIN" - <<'PY'
import inspect
from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM
path = inspect.getfile(ComposedDFlashStudentForCausalLM)
if not hasattr(ComposedDFlashStudentForCausalLM, '_get_response_train_block_size'):
    raise RuntimeError(f'Loaded an older verl checkout: {path}')
print(f'Using native OPD student: {path}')
import torch
from verl.utils.attention_utils import unpad_input, pad_input
x = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
mask = torch.tensor([[0, 1, 1]])
values, indices, *_ = unpad_input(x, mask)
restored = pad_input(values, indices, 1, 3)
assert torch.equal(restored[:, 1:], x[:, 1:])
print('Padding preflight passed (flash_attn is optional for padding helpers).')
PY
mkdir -p "$CKPT_DIR"
printf 'arm=%s\nresponse_train_lengths=%s\nrollout_and_rejected_replay=native\ncheckpoint=%s\n' \
    "$ARM" "$WIDTH_CONFIG" "$CKPT_DIR"
exec bash verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh \
    "actor_rollout_ref.model.path=$MAIN_MODEL_PATH" \
    "++actor_rollout_ref.model.override_config.verl_dflash_main_model_path=$MAIN_MODEL_PATH" \
    "++actor_rollout_ref.model.override_config.verl_dflash_draft_model_path=$DRAFT_MODEL_PATH" \
    "++actor_rollout_ref.model.override_config.verl_dflash_train_candidate_lengths=$WIDTH_CONFIG" \
    "++actor_rollout_ref.model.override_config.verl_dflash_attention_impl=sdpa" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=True" \
    "actor_rollout_ref.actor.use_torch_compile=False" \
    "distillation.teacher_models.teacher_model.model_path=$TEACHER_MODEL_PATH" \
    "data.val_files=['$VAL_JSONL']" \
    "trainer.update_accumulation_steps=${UPDATE_ACCUMULATION_STEPS:-10}" \
    "trainer.rollout_speed_test_only=False" \
    "trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}" \
    "trainer.logger=['console','wandb']" \
    "trainer.project_name=qwen3-4b-native-opd-length" \
    "${EXTRA[@]}" "$@"
