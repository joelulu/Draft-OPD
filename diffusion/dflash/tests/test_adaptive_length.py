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


def test_shared_tail_policy_and_native_baseline():
    from adaptive_length.evaluation import actual_length, fixed_lengths
    assert actual_length(32, 3) == 2
    assert actual_length(4, 3) == 2
    assert actual_length(12, 1) == 0
    assert fixed_lengths([4, 8, 16, 32], 16) == [4, 8, 15, 16, 32]


@pytest.mark.parametrize("budget", [1, 3, 19])
@pytest.mark.parametrize("eos", [None, 0])
def test_evaluation_clips_all_policies_and_preserves_accounting(tmp_path, monkeypatch, budget, eos):
    import json
    import csv
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from model.dflash import DFlashDraftModel
    from adaptive_length import pipeline
    config = Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=128, attn_implementation="sdpa")
    target = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        target.lm_head.weight.zero_()  # all draft proposals accepted, exact predictable tail
    dc = Qwen3Config(**config.to_dict())
    dc.num_hidden_layers = 1
    dc.num_target_layers = 2
    dc.block_size = 16
    dc.layer_types = ["full_attention"]
    dc.dflash_config = dict(target_layer_ids=[0], mask_token_id=30)
    dc._attn_implementation = "sdpa"
    draft = DFlashDraftModel(dc).eval()
    class Head(nn.Module):
        lengths = (4, 8)
        def forward(self, pivot, anchor):
            return pivot.new_tensor([[1., 7.]])
    monkeypatch.setattr(pipeline, "setup", lambda args: (
        target, draft, SimpleNamespace(eos_token_id=eos), [("key", [])], 0, 1))
    monkeypatch.setattr(pipeline, "encode", lambda *args: torch.tensor([[1, 2, 3]]))
    monkeypatch.setattr(pipeline, "load_head", lambda *args: (Head(), [1., 1.]))
    args = SimpleNamespace(head="existing-head.pt", lengths=(4, 8), fixed_lengths=[12],
        output=str(tmp_path), profile_rounds=True, strict_match=False, warmup=True,
        seed=42, max_new_tokens=budget)
    pipeline.evaluate(args)
    records = [json.loads(line) for line in (tmp_path / "results_rank000.jsonl").read_text().splitlines()]
    assert {r["mode"] for r in records} == {"ar", "fixed_4", "fixed_8", "fixed_12", "fixed_15", "adaptive"}
    for rec in records:
        assert rec["tokens"] == (1 if eos == 0 else budget) and rec["match"]
        assert rec["head_ms"] >= 0
        if rec["mode"] != "ar":
            assert rec["ar_steps"] <= 1
        assert rec["spec_committed_tokens"] + rec["ar_steps"] == rec["tokens"]
        assert rec["accepted_tokens"] + rec["spec_rounds"] == rec["spec_committed_tokens"]
    rounds = [json.loads(line) for line in (tmp_path / "rounds_rank000.jsonl").read_text().splitlines()]
    for method in rounds:
        for step in method["rounds"]:
            if step["actual_k"]:
                assert step["actual_k"] == min(step["requested_k"], step["remaining"] - 1)
    with (tmp_path / "summary_rank000.csv").open() as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == len(records)


def test_exact_candidate_budget_plans():
    import random
    from adaptive_length.training import width_plan
    assert width_plan([15], 240, random.Random(42)) == [15] * 16
    widths = [4, 8, 12, 16, 24, 32]
    plan = width_plan(widths, 240, random.Random(42))
    assert sum(plan) == 240 and set(plan) <= set(widths)
    assert plan == width_plan(widths, 240, random.Random(42))
    with pytest.raises(ValueError, match="cannot be tiled"):
        width_plan([15], 64, random.Random(42))


def test_matched_budget_real_training_updates_and_exports(tmp_path, monkeypatch):
    import json
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from model.dflash import DFlashDraftModel
    from adaptive_length import pipeline
    torch.manual_seed(72)
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
    initial = DFlashDraftModel(dc).state_dict()
    frozen = {k: v.clone() for k, v in target.state_dict().items()}
    rows = [("first", []), ("second", [])]
    monkeypatch.setattr(pipeline, "encode", lambda *args: torch.tensor([[1, 2, 3]]))
    for arm, lengths in (("fixed", (4,)), ("mixed", (4, 8))):
        draft = DFlashDraftModel(dc)
        draft.load_state_dict(initial)
        before = draft.fc.weight.detach().clone()
        monkeypatch.setattr(pipeline, "setup", lambda args: (
            target, draft, SimpleNamespace(eos_token_id=None), rows, 0, 1))
        args = SimpleNamespace(split="train", candidate_tokens_per_update=24,
            steps=2, save_every=2, log_every=1, seed=42, lr=1e-3, lengths=lengths,
            output=str(tmp_path / arm), target="target", draft="initial", data="train.jsonl",
            max_prompt=32, max_new_tokens=16)
        pipeline.train_draft(args)
        assert not torch.equal(draft.fc.weight, before)
        logs = [json.loads(s) for s in (tmp_path / arm / "train_rank000.jsonl").read_text().splitlines()]
        assert [r["prompt_key"] for r in logs] == ["first", "second"]
        assert [r["candidate_tokens"] for r in logs] == [24, 24]
        assert logs[-1]["cumulative_candidate_tokens_per_rank"] == 48
        saved = json.loads((tmp_path / arm / "step_000002/adaptive_training.json").read_text())
        assert saved["global_candidate_tokens"] == 48
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, frozen[key])


def test_comparison_report_rejects_unmatched_budgets(tmp_path):
    import csv
    import json
    from adaptive_length.compare_training import build_comparison
    plan = dict(nproc=1, steps=1, candidate_tokens_per_rank_per_update=240,
                fixed_k=15, mixed_lengths=[4,8,12,16,24,32])
    (tmp_path / 'comparison_plan.json').write_text(json.dumps(plan))
    for arm in ['initial', 'fixed', 'mixed']:
        folder = tmp_path / arm / 'eval_fixed'
        folder.mkdir(parents=True)
        (folder / 'results_rank000.jsonl').write_text(json.dumps(dict(prompt_key='heldout', mode='fixed_15'))+'\n')
        with (folder / 'summary_rank000.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['mode','decode_tok_s'])
            writer.writeheader()
            writer.writerow(dict(mode='fixed_15', decode_tok_s=100))
        if arm == 'initial':
            continue
        folder = tmp_path / arm / 'draft'
        folder.mkdir()
        meta = dict(initial_draft='same', target='target', seed=42, world_size=1,
            learning_rate=0.001, train_data='same-data', split='train', max_prompt=512,
            max_new_tokens=256, local_prompt_keys=['train-prompt'], planned_steps=1,
            candidate_tokens_per_rank_per_update=240, train_parameters_dtype='float32',
            lengths=[15] if arm == 'fixed' else plan['mixed_lengths'])
        (folder / 'training_rank000.json').write_text(json.dumps(meta))
        (folder / 'train_rank000.jsonl').write_text(json.dumps(dict(step=1, candidate_tokens=240,
            cumulative_candidate_tokens_per_rank=240, elapsed_s=1.0))+'\n')
    report = build_comparison(tmp_path)
    assert report['budget_matched']
    assert (tmp_path / 'comparison.csv').exists()
    path = tmp_path / 'mixed/draft/train_rank000.jsonl'
    row = json.loads(path.read_text())
    row['candidate_tokens'] = 239
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='unmatched training budget'):
        build_comparison(tmp_path)
