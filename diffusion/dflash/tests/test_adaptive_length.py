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


def test_training_cli_groups_duplicate_prompts_and_saves_costs(tmp_path):
    import json
    from adaptive_length.cli import train, load
    source = tmp_path / "paired"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps(dict(hidden_size=8, lengths=[4, 8])))
    manifest = []
    for index, prompt_id in enumerate(["same", "same", "other", "third", "fourth"]):
        row = dict(pivot=torch.randn(8), anchor=torch.randn(8), accepted=torch.tensor([2, 3]),
                   draft_ms=torch.tensor([1., 2.]), verify_ms=torch.tensor([2., 4.]))
        name = f"prompt_{index}.pt"
        torch.save(dict(prompt_id=prompt_id, rows=[row]), source / name)
        manifest.append(dict(file=name, prompt_id=prompt_id, states=1))
    (source / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "head.pt"
    args = SimpleNamespace(seed=42, input=str(source), output=str(output), val_fraction=0.25,
                           device="cpu", features="both", lr=0.001, epochs=2, batch_size=4)
    train(args)
    saved = load(output)
    assert not set(saved["train_prompt_ids"]) & set(saved["val_prompt_ids"])
    assert set(saved["train_prompt_ids"] + saved["val_prompt_ids"]) == {"same", "other", "third", "fourth"}
    assert saved["fixed_k"] == 4
    torch.testing.assert_close(saved["costs_ms"], torch.tensor([3., 6.]))
