"""Exercise the actual verl student directly, without importing the Ray trainer."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import PretrainedConfig, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'diffusion/dflash'))
from model.dflash import DFlashDraftModel

spec = importlib.util.spec_from_file_location('native_dflash_student_under_test',
    ROOT / 'verl/verl/models/transformers/dflash_student.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
Student = native.ComposedDFlashStudentForCausalLM


def make_student(widths=None):
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128, attn_implementation='sdpa')
    target = Qwen3ForCausalLM(config)
    dc = Qwen3Config(**config.to_dict())
    dc.num_hidden_layers = 1
    dc.num_target_layers = 2
    dc.block_size = 8  # 7 native candidates
    dc.layer_types = ['full_attention']
    dc.dflash_config = dict(target_layer_ids=[0], mask_token_id=40)
    dc._attn_implementation = 'sdpa'
    wrapper = PretrainedConfig(verl_dflash_attention_impl='sdpa',
        verl_dflash_train_candidate_lengths=widths)
    return Student(wrapper, target, DFlashDraftModel(dc)).train()


def batch():
    ids = torch.arange(15)[None]
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        dflash_prompt_lengths=torch.tensor([3]), dflash_response_lengths=torch.tensor([12]),
        dflash_reject_token_indices=torch.tensor([[2,6,11]]),
        dflash_rejected_draft_anchor_indices=torch.tensor([[-1]]),
        dflash_rejected_draft_offsets=torch.tensor([[5]]),
        dflash_rejected_draft_token_ids=torch.tensor([[23]]),
        dflash_rejected_draft_teacher_logprobs=torch.tensor([[-2.5]]),
        dflash_rejected_draft_mask=torch.tensor([[True]]),
        dflash_use_composed_teacher_logprobs=True)


@pytest.mark.parametrize('k', [4, 12, 32])
def test_native_rejected_replay_keeps_its_original_width(k):
    student = make_student([k])
    widths = []
    forward = student._run_dflash_draft_forward
    def capture(**kwargs):
        widths.append(kwargs['draft_block_size'])
        return forward(**kwargs)
    student._run_dflash_draft_forward = capture
    output = student(**batch())
    assert widths == [k+1, 8]
    baseline = make_student(None)(**batch())
    torch.testing.assert_close(output['dflash_loss_mask'], baseline['dflash_loss_mask'])
    torch.testing.assert_close(output['dflash_teacher_log_probs'], baseline['dflash_teacher_log_probs'])
    assert output['dflash_rejected_draft_loss_mask'].item()
    assert output['dflash_rejected_draft_teacher_log_probs'].item() == -2.5
    assert output['dflash_opd_native_replay_candidates'].item() == 7
    assert output['dflash_opd_train_response_candidates'].item() == k
    loss = -(output['dflash_log_probs'] * output['dflash_loss_mask']).sum()
    loss -= output['dflash_rejected_draft_student_log_probs'].sum()
    loss.backward()
    assert student.draft_model.fc.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in student.main_model.parameters())
    assert student.draft_model.block_size == 8


def test_native_width_configuration_preserves_original_logprobs_and_masks():
    baseline = make_student(None)
    explicit_native = make_student([7])
    a, b = baseline(**batch()), explicit_native(**batch())
    for name in ('dflash_log_probs', 'dflash_loss_mask', 'dflash_teacher_log_probs',
                 'dflash_rejected_draft_student_log_probs', 'dflash_rejected_draft_loss_mask'):
        torch.testing.assert_close(a[name], b[name], atol=2e-6, rtol=2e-6)


def test_eval_and_no_grad_keep_native_width_and_tv_rejects_mixed_mode():
    student = make_student([4,8,12,16,24,32])
    with torch.no_grad():
        assert student(**batch())['dflash_opd_train_response_candidates'].item() == 7
    student.eval()
    assert student._get_response_train_block_size() == 8
    student.train()
    args = batch()
    args['dflash_use_tv_loss'] = True
    with pytest.raises(NotImplementedError, match='TV'):
        student(**args)
    student.config.verl_dflash_train_candidate_lengths = [32, 4]
    with pytest.raises(ValueError, match='increasing'):
        student._get_train_candidate_lengths()
