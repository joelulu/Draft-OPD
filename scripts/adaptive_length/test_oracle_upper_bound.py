"""Logic tests plus real tiny Qwen3/DFlash CPU oracle and replay checks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time
import pytest

spec = importlib.util.spec_from_file_location('oracle_upper_bound', Path(__file__).with_name('oracle_upper_bound.py'))
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_ties_speed_and_tail_eos():
    trials = [dict(requested_k=k, emitted=5, total_ms=[t, t, t]) for k, t in [(4, 3), (8, 2)]]
    assert oracle.select_trial(trials, 'accept')['requested_k'] == 4
    assert oracle.select_trial(trials, 'speed')['requested_k'] == 8
    assert oracle.emitted_count([1, 2, 3, 4], 3, 2, None) == 2
    assert oracle.emitted_count([1, 2, 3, 4], 3, 4, 2) == 2


@pytest.fixture
def tiny():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from model.dflash import DFlashDraftModel
    from adaptive_length.runtime import RoundRuntime
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=256, attn_implementation='sdpa')
    target = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    dc = Qwen3Config(**config.to_dict())
    dc.num_hidden_layers, dc.num_target_layers, dc.block_size = 1, 2, 33
    dc.layer_types = ['full_attention']
    dc.dflash_config = dict(target_layer_ids=[0], mask_token_id=30)
    dc._attn_implementation = 'sdpa'
    draft = DFlashDraftModel(dc).eval().requires_grad_(False)
    args = SimpleNamespace(seed=42, max_new_tokens=13, search_repeats=2)
    return torch, target, draft, RoundRuntime, args


@pytest.mark.parametrize('objective', ['accept', 'speed'])
def test_real_oracle_replay_and_native_cost_paths(tiny, objective):
    torch, target, draft, Runtime, args = tiny
    ids = torch.tensor([[1, 2, 3]])
    head = torch.nn.Sequential(torch.nn.LayerNorm(16), torch.nn.Linear(16, 128),
                              torch.nn.SiLU(), torch.nn.Linear(128, 8)).eval()
    with torch.inference_mode():
        schedule = oracle.discover(target, draft, ids, objective, args, None,
                                   Runtime, time.perf_counter, torch)
        assert len(schedule['generated']) == args.max_new_tokens
        baseline = oracle.replay(target, draft, ids, schedule, None, None, head,
            args, None, Runtime, time.perf_counter, torch)
        for context in ('full_kv', 'window_64'):
            result = oracle.replay(target, draft, ids, schedule, None, context, head,
                args, None, Runtime, time.perf_counter, torch, profile=True)
            assert result['output_sha256'] == baseline['output_sha256']
            assert result['selector_calls'] == result['spec_rounds']
            assert result['profile']['selector_gpu_ms'] > 0
        broken = dict(schedule, generated=[999])
        with pytest.raises(RuntimeError, match='Replay differs'):
            oracle.replay(target, draft, ids, broken, None, None, head,
                args, None, Runtime, time.perf_counter, torch)


def test_window64_matches_full_below64_and_crops_above(tiny):
    torch, target, draft, Runtime, args = tiny
    class Capture(torch.nn.Module):
        def forward(self, x):
            self.features = x.clone()
            return x[:, :8]
    capture = Capture()
    with torch.inference_mode():
        for length in (5, 70):
            state = Runtime(target, draft, torch.ones((1, length), dtype=torch.long))
            oracle.native_predictor_probe(state, 'full_kv', capture, torch)
            full = capture.features.clone()
            cache = [(v.keys.clone(), v.values.clone()) for v in state.draft_cache.layers]
            oracle.native_predictor_probe(state, 'window_64', capture, torch)
            if length <= 64:
                torch.testing.assert_close(full, capture.features)
            else:
                assert not torch.allclose(full, capture.features, atol=1e-8)
            for layer, (keys, values) in zip(state.draft_cache.layers, cache):
                torch.testing.assert_close(layer.keys, keys)
                torch.testing.assert_close(layer.values, values)


def test_eos_and_single_token_ar(tiny):
    torch, target, draft, Runtime, args = tiny
    with torch.no_grad():
        target.lm_head.weight.zero_()
    ids = torch.tensor([[1, 2]])
    with torch.inference_mode():
        for budget, eos in [(1, None), (8, 0)]:
            args.max_new_tokens = budget
            schedule = oracle.discover(target, draft, ids, 'accept', args, eos,
                Runtime, time.perf_counter, torch)
            r = oracle.replay(target, draft, ids, schedule, None, None, None,
                args, eos, Runtime, time.perf_counter, torch)
            assert r['tokens'] == 1
            assert r['accepted_tokens'] == 0


def test_summary_uses_token_weighted_throughput(tmp_path):
    rows = []
    for arm in ('fixed', 'mixed'):
        for tokens, seconds in [(10, 1), (20, 4)]:
            rows.append(dict(arm=arm, mode='fixed_16', prompt_key=str(tokens),
                tokens=tokens, decode_s=seconds, spec_rounds=2, selector_calls=0,
                accepted_tokens=2, committed_tokens=4, candidate_tokens=32, selector_wall_ms=0))
    summary = oracle.summarize(rows, tmp_path/'comparison.csv')
    assert all(r['tok_s'] == 6 for r in summary)
