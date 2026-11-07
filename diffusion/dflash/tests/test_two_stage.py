"""Real tiny-model state/caching/classifier checks; no pretrained weights needed."""
import json
import random
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from model.dflash import DFlashDraftModel
from adaptive_length.runtime import RoundRuntime
from adaptive_length.two_stage import K_VALUES
from adaptive_length.two_stage.classifier import KVClassifier, cache_pairs
from adaptive_length.two_stage.labels import calibrate, costs_for, make_labels
from adaptive_length.two_stage.cli import iter_states, paired_measure, select_prompts, label


def models():
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=256, attn_implementation='sdpa')
    target = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    dc = Qwen3Config(**config.to_dict())
    dc.num_hidden_layers = 1
    dc.num_target_layers = 2
    dc.block_size = 16
    dc.layer_types = ['full_attention']
    dc.dflash_config = dict(target_layer_ids=[0], mask_token_id=40)
    dc._attn_implementation = 'sdpa'
    return target, DFlashDraftModel(dc).eval().requires_grad_(False)


def classifier(mode):
    return KVClassifier(anchor_dim=16, head_dim=8, layers=2, kv_heads=1, context_mode=mode, hidden=16)


def test_dual_labels_and_ties():
    result = make_labels([4,7,8,8,9,9,9,9], [20,22,24,27,30,34,38,42])
    assert result['label_accept_k'] == 20
    assert result['label_speed_k'] == 12
    assert make_labels([2]*8, [1]*8)['label_accept_k'] == 4
    assert make_labels([2]*8, [1]*8)['label_speed_k'] == 4
    with pytest.raises(ValueError):
        make_labels([33]*8, [1]*8)
    with pytest.raises(ValueError):
        make_labels([0]*8, [float('nan')]*8)


def test_costs_calibrated_only_from_train_states():
    rows = [dict(split='train', context_length=20, total_times_ms=[[1,2,3]]*8)]
    table = calibrate(rows)
    assert costs_for(table, 100) == [2]*8
    rows[0]['split'] = 'test'
    with pytest.raises(ValueError, match='training'):
        calibrate(rows)


def test_window16_ignores_older_kv_full_context_reads_it():
    torch.manual_seed(42)
    pairs = [(torch.randn(1,1,40,8), torch.randn(1,1,40,8)) for _ in range(2)]
    changed = [(k.clone(), v.clone()) for k, v in pairs]
    for k, v in changed:
        k[..., :24, :] += 100
        v[..., :24, :] += 100
    q = torch.randn(1,16)
    window, full = classifier('window_16'), classifier('full_kv')
    torch.testing.assert_close(window(pairs,q), window(changed,q))
    assert not torch.allclose(full(pairs,q), full(changed,q))


@pytest.mark.parametrize('mode', ['full_kv', 'window_16'])
def test_classifier_learns_and_never_backprops_into_kv_or_anchor(mode):
    torch.manual_seed(42)
    model = classifier(mode)
    pairs = [(torch.randn(1,1,20,8,requires_grad=True), torch.randn(1,1,20,8,requires_grad=True)) for _ in range(2)]
    q = torch.randn(1,16)
    opt = torch.optim.AdamW(model.parameters(), lr=.02)
    y = torch.tensor([3])
    before = torch.nn.functional.cross_entropy(model(pairs,q), y).item()
    for _ in range(12):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(pairs,q), y)
        loss.backward()
        opt.step()
    assert loss.item() < before
    assert all(k.grad is None and v.grad is None for k,v in pairs)
    restored = KVClassifier(**model.settings)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(pairs,q),restored(pairs,q))


def test_real_paired_trials_restore_target_and_draft_and_are_order_independent():
    target, draft = models()
    with torch.no_grad():
        runtime = RoundRuntime(target,draft,torch.tensor([[1,2,3]]))
        prefix, anchor = runtime.prefix.clone(), runtime.anchor.clone()
        original_cache = [(k.clone(),v.clone()) for k,v in cache_pairs(runtime.target_cache)]
        first = paired_measure(runtime,2,random.Random(1))
        second = paired_measure(runtime,1,random.Random(99))
        assert first['accepted_lengths'] == second['accepted_lengths']
        torch.testing.assert_close(prefix,runtime.prefix)
        torch.testing.assert_close(anchor,runtime.anchor)
        for (ka,va),(kb,vb) in zip(original_cache,cache_pairs(runtime.target_cache)):
            torch.testing.assert_close(ka,kb)
            torch.testing.assert_close(va,vb)
        assert runtime.target_cache.get_seq_length() == 3
        assert runtime.draft_cache.get_seq_length() == 0
        for k in K_VALUES:
            block = runtime.propose(k)
            assert block.shape[1] == k+1
        runtime.step(8)
        assert runtime.prefix.shape[1] > 3


def test_streamed_target_cache_matches_full_prefix_reconstruction():
    target,_ = models()
    rows = [dict(prompt_key='a', prefix_ids=list(range(1,n+1)), context_length=n, anchor_id=3) for n in [3,5,8]]
    for row, pairs, _ in iter_states(target,rows):
        reference = RoundRuntime(target,models()[1],torch.tensor([row['prefix_ids']]))
        for (ka,va),(kb,vb) in zip(pairs,cache_pairs(reference.target_cache)):
            torch.testing.assert_close(ka,kb,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(va,vb,atol=2e-6,rtol=2e-6)


def test_prompt_partitions_are_disjoint_and_repeatable(tmp_path):
    path = tmp_path/'data.jsonl'
    path.write_text('\n'.join(json.dumps(dict(prompt=f'q{i}')) for i in range(200)))
    args = SimpleNamespace(data=str(path),test_data=None,seed=42,train_prompts=20,val_prompts=10,test_prompts=10)
    a,b = select_prompts(args),select_prompts(args)
    assert a==b
    keys=[{key for key,_ in a[s]} for s in ['train','val','test']]
    assert not keys[0]&keys[1] and not keys[0]&keys[2] and not keys[1]&keys[2]


def test_label_command_saves_both_labels_and_test_prompts(tmp_path):
    raw,out = tmp_path/'raw', tmp_path/'labeled'
    raw.mkdir()
    (raw/'manifest_rank000.json').write_text(json.dumps(dict(target='x',draft='y')))
    (raw/'complete_rank000.json').write_text('{}')
    for split in ['train','val','test']:
        row = dict(state_id=split,prompt_key=split,split=split,context_length=20,
            k_values=list(K_VALUES),accepted_lengths=[4,7,8,8,9,9,9,9],
            total_times_ms=[[v] for v in [20,22,24,27,30,34,38,42]],prompt=[dict(role='user',content=split)])
        (raw/f'{split}.raw.rank000.jsonl').write_text(json.dumps(row)+'\n')
    label(SimpleNamespace(data=str(raw),output=str(out)))
    for split in ['train','val','test']:
        row=json.loads((out/f'{split}.jsonl').read_text())
        assert row['label_accept_k']==20 and row['label_speed_k']==12
        assert (out/f'{split}_prompts.jsonl').is_file()


def test_complete_collect_label_train_and_dynamic_eval_on_tiny_models(tmp_path, monkeypatch):
    from adaptive_length.two_stage import cli
    from adaptive_length import pipeline
    target, draft = models()
    with torch.no_grad():
        target.lm_head.weight.zero_()  # predictable full acceptance and no EOS
    class Tokenizer:
        eos_token_id = None
        def apply_chat_template(self, *args, **kwargs):
            return 'prompt'
        def encode(self, text, **kwargs):
            return torch.tensor([[1,2,3]])
    tokenizer = Tokenizer()
    monkeypatch.setattr(cli, 'model_identity', lambda path: path)
    monkeypatch.setattr(cli, 'load_models', lambda args, draft=True: (target, models()[1] if draft else None))
    monkeypatch.setattr(cli.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: tokenizer)
    raw, labeled, trained = tmp_path/'raw', tmp_path/'labeled', tmp_path/'classifiers'
    data = tmp_path/'prompts.jsonl'
    data.write_text('\n'.join(json.dumps(dict(prompt=f'q{i}')) for i in range(100)))
    cli.collect(SimpleNamespace(target='target',draft='draft',data=str(data),test_data=None,
        output=str(raw),train_prompts=2,val_prompts=2,test_prompts=2,states_per_prompt=2,
        state_stride=4,max_prompt=32,max_new_tokens=96,timing_repeats=1,seed=42))
    cli.label(SimpleNamespace(data=str(raw),output=str(labeled)))
    cli.train(SimpleNamespace(target='target',draft='draft',data=str(labeled),output=str(trained),
        seed=42,epochs=1,accumulate=3,lr=.001))
    paths=sorted(trained.glob('*.pt'))
    assert len(paths)==4
    prompt_rows=cli.read_prompts(labeled/'test_prompts.jsonl')
    monkeypatch.setattr(pipeline,'setup',lambda args: (target,draft,tokenizer,prompt_rows,0,1))
    monkeypatch.setattr(pipeline,'encode',lambda *args: torch.tensor([[1,2,3]]))
    out=tmp_path/'eval'
    args=SimpleNamespace(target='target',draft='draft',head=None,classifier=list(map(str,paths)),
        enable_thinking=False,lengths=list(K_VALUES),fixed_lengths=None,exact_lengths=True,
        output=str(out),profile_rounds=False,strict_match=False,warmup=True,seed=42,max_new_tokens=40)
    pipeline.evaluate(args)
    records=[json.loads(v) for v in (out/'results_rank000.jsonl').read_text().splitlines()]
    assert len({r['mode'] for r in records})==13  # AR + eight K + four classifiers
    assert all(r['tokens']==40 for r in records)
    assert all(r['match'] for r in records)
    assert all(r['selector_calls'] > 0 and r['selector_wall_ms'] > 0 for r in records if r['mode'].startswith('classifier_'))
    assert all(p.grad is None for p in target.parameters())
    # Force the classifier to K=16 and prove it reproduces fixed-K execution.
    ckpt=torch.load(paths[0],map_location='cpu',weights_only=True)
    ckpt['state_dict']['head.2.weight'].zero_()
    ckpt['state_dict']['head.2.bias'].fill_(-100)
    ckpt['state_dict']['head.2.bias'][3]=100
    fixed_path=tmp_path/'forced.pt'
    torch.save(ckpt,fixed_path)
    args.classifier=[str(fixed_path)]
    args.output=str(tmp_path/'forced_eval')
    pipeline.evaluate(args)
    forced=[json.loads(v) for v in (Path(args.output)/'results_rank000.jsonl').read_text().splitlines()]
    fixed=next(r for r in forced if r['mode']=='fixed_16')
    selected=next(r for r in forced if r['mode']=='classifier_forced')
    assert fixed['actions']==selected['actions']
    assert fixed['accepted_tokens']==selected['accepted_tokens']


def test_paired_labels_count_emitted_tokens_before_eos():
    target,draft = models()
    with torch.no_grad():
        target.lm_head.weight.zero_()
        runtime=RoundRuntime(target,draft,torch.tensor([[1,2,3]]))
        measured=paired_measure(runtime,1,random.Random(42),eos=0)
    assert measured['raw_accepted_lengths']==list(K_VALUES)
    assert measured['accepted_lengths']==[0]*8
    assert measured['committed_lengths']==[1]*8
