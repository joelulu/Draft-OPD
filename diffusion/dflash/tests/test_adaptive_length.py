"""CPU checks: pre-draft features, cache rollback, length choice, and exact greedy output."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_length.head import LengthHead, select_lengths


def test_selection_uses_cost_and_output_budget():
    expected = torch.tensor([[3., 7., 8.]])
    assert select_lengths(expected, [2., 8., 20.], [4, 8, 12]).item() == 0
    assert select_lengths(expected, [20., 2., 20.], [4, 8, 12], remaining=6).item() == 0
    with pytest.raises(ValueError):
        select_lengths(expected, [1., 0., 1.], [4, 8, 12])
    with pytest.raises(ValueError):
        select_lengths(expected, [1., 1., 1.], [4, 8, 12], remaining=2)


@pytest.mark.parametrize("features", ["both", "anchor", "pivot"])
def test_head_trains_and_roundtrips(features, tmp_path):
    torch.manual_seed(42)
    head = LengthHead(8, lengths=[4, 8], features=features)
    pivot, anchor = torch.randn(6, 8), torch.randn(6, 8)
    expected = torch.tensor([[2., 3.]]).repeat(6, 1)
    opt = torch.optim.Adam(head.parameters(), lr=0.01)
    before = head.loss(pivot, anchor, expected).item()
    for _ in range(12):
        opt.zero_grad()
        head.loss(pivot, anchor, expected).backward()
        opt.step()
    assert head.loss(pivot, anchor, expected).item() < before
    path = tmp_path / "head.pt"
    torch.save(dict(settings=head.settings, state_dict=head.state_dict()), path)
    saved = torch.load(path, weights_only=True)
    restored = LengthHead(**saved["settings"])
    restored.load_state_dict(saved["state_dict"])
    torch.testing.assert_close(head(pivot, anchor), restored(pivot, anchor))


@pytest.mark.parametrize("all_accepted", [False, True])
def test_paired_runtime_and_variable_steps_match_greedy(all_accepted):
    # Exercise real DFlash + tiny randomly initialized Qwen3 on CPU, including full
    # rejection and changing widths. No downloaded model or mocked cache behavior.
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from model.dflash import DFlashDraftModel
    from adaptive_length.runtime import RoundRuntime
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=128, attn_implementation="sdpa")
    target = Qwen3ForCausalLM(config).eval()
    if all_accepted:
        with torch.no_grad():
            target.lm_head.weight.zero_()
    dc = Qwen3Config(**config.to_dict())
    dc.num_hidden_layers = 1
    dc.num_target_layers = 2
    dc.block_size = 5
    dc.layer_types = ["full_attention"]
    dc.dflash_config = dict(target_layer_ids=[0], mask_token_id=30)
    dc._attn_implementation = "sdpa"
    draft = DFlashDraftModel(dc).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        runtime = RoundRuntime(target, draft, ids)
        # Features cannot depend on the chosen speculative width.
        original = tuple(x.clone() for x in runtime.features())
        cached = runtime.draft_cache.get_seq_length()
        first = runtime.propose(4)
        result = runtime.verify(first)
        assert runtime.target_cache.get_seq_length() == 3
        if all_accepted:
            assert result[0] == 4
        runtime.draft_cache.crop(cached)
        repeated = runtime.propose(4)
        torch.testing.assert_close(first, repeated)
        runtime.verify(repeated)
        for a, b in zip(original, runtime.features()):
            torch.testing.assert_close(a, b)
        runtime.advance(first, result)  # paired collection restores committed target KV
        for k in [4, 8, 2, 4]:
            runtime.step(k)
        # Greedy reference is recomputed without a cache to catch position/cache errors.
        expected = ids.clone()
        while expected.shape[1] < runtime.prefix.shape[1]:
            next_token = target(expected).logits[:, -1:].argmax(-1)
            expected = torch.cat((expected, next_token), dim=1)
        torch.testing.assert_close(runtime.prefix, expected)
        runtime.ar_step()
        torch.testing.assert_close(runtime.prefix[:, -1:], target(expected).logits[:, -1:].argmax(-1))


def test_opd_loss_has_student_gradients_only():
    from adaptive_length.pipeline import opd_loss
    student = torch.randn(1, 4, 7, requires_grad=True)
    teacher = torch.randn(1, 4, 7, requires_grad=True)
    loss = opd_loss(student, teacher, accepted=2)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert (student.grad.abs().sum(-1) > 0).all()
    assert teacher.grad is None


def test_head_pipeline_groups_prompts_and_saves_costs(tmp_path):
    from adaptive_length.pipeline import train_head, split_rows
    source = tmp_path / "paired"
    source.mkdir()
    # All duplicate occurrences of a content hash must stay in one group.
    keys = ["00000001", "00000001", "00000002", "00000005", "0000000a"]
    torch.save(dict(pivot=torch.randn(5, 8), anchor=torch.randn(5, 8),
                    accepted=torch.tensor([[2., 3.]]).repeat(5, 1),
                    costs_ms=torch.tensor([[3., 6.]]).repeat(5, 1),
                    prompt_keys=keys, metadata=dict(lengths=[4,8])), source / "labels_rank000.pt")
    output = tmp_path / "head.pt"
    args = SimpleNamespace(seed=42, labels=str(source), output=str(output), costs=None,
                           device="cpu", features="both", projection=8,
                           lr=0.001, epochs=2, batch_size=4)
    train_head(args)
    saved = torch.load(output, weights_only=True)
    assert saved["used_prompt_keys"] == sorted(set(keys))
    assert saved["validation"]["train_anchors"] == 3
    assert saved["validation"]["val_anchors"] == 2
    torch.testing.assert_close(saved["costs_ms"], torch.tensor([3., 6.]))
    rows = [(key, []) for key in keys]
    assert not set(k for k, _ in split_rows(rows, "train")) & set(k for k, _ in split_rows(rows, "val"))


def test_real_draft_training_backward_and_fresh_cache():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from model.dflash import DFlashDraftModel
    from adaptive_length.runtime import RoundRuntime
    from adaptive_length.pipeline import opd_loss
    torch.manual_seed(33)
    config = Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=128, attn_implementation="sdpa")
    target = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    dc = Qwen3Config(**config.to_dict())
    dc.num_hidden_layers = 1
    dc.num_target_layers = 2
    dc.block_size = 5
    dc.layer_types = ["full_attention"]
    dc.dflash_config = dict(target_layer_ids=[0], mask_token_id=30)
    dc._attn_implementation = "sdpa"
    draft = DFlashDraftModel(dc)
    opt = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    prefix = torch.tensor([[1, 2, 3]])
    for k in [4, 8]:
        runtime = RoundRuntime(target, draft, prefix)
        block, logits = runtime.propose(k, return_logits=True)
        result, teacher_logits = runtime.verify(block, keep_cache=True, return_logits=True)
        loss = opd_loss(logits, teacher_logits, result[0])
        opt.zero_grad()
        loss.backward()
        assert draft.fc.weight.grad is not None
        assert torch.isfinite(draft.fc.weight.grad).all()
        assert all(p.grad is None for p in target.parameters())
        opt.step()
        with torch.no_grad():
            runtime.advance(block, result, cache_kept=True)
            prefix = runtime.prefix.detach()
