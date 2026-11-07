"""Compare selector features/cache directly with original DFlash decoder code."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from transformers import DynamicCache,Qwen3Config,Qwen3ForCausalLM

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.dflash import DFlashDraftModel
from model.utils import extract_context_feature
from adaptive_length.runtime import RoundRuntime
from adaptive_length.two_stage.classifier import cache_pairs,build_classifier,DFlashLengthClassifier
from adaptive_length.two_stage.dflash_features import materialize_context,read_anchor,iter_dflash_states


def models():
    torch.manual_seed(17)
    config=Qwen3Config(vocab_size=41,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=4,
        max_position_embeddings=256,attn_implementation='sdpa')
    target=Qwen3ForCausalLM(config).eval().requires_grad_(False)
    dc=Qwen3Config(**config.to_dict())
    dc.num_hidden_layers=2
    dc.num_target_layers=2
    dc.block_size=16
    dc.layer_types=['full_attention','full_attention']
    dc.dflash_config=dict(target_layer_ids=[0,1],mask_token_id=40)
    dc._attn_implementation='sdpa'
    return target,DFlashDraftModel(dc).eval().requires_grad_(False)


def context(target,draft,ids):
    with torch.no_grad():
        output=target(ids,output_hidden_states=True)
        return extract_context_feature(output.hidden_states,draft.target_layer_ids)


@pytest.mark.parametrize('mode',['full_kv','window_16'])
def test_native_qk_rope_gqa_oproj_and_mlp_equal_direct_decoder(mode):
    target,draft=models()
    ids=torch.arange(1,25)[None]
    hidden=context(target,draft,ids)
    cache=DynamicCache()
    materialize_context(draft,cache,hidden,0)
    position=ids.shape[1]
    embedding=target.get_input_embeddings()(torch.tensor([[7]]))
    before=[(k.clone(),v.clone()) for k,v in cache_pairs(cache)]
    actual=read_anchor(draft,cache,embedding,position,mode)
    # Original layer, independently rebuilding K/V from target features.
    first=max(0,position-16) if mode=='window_16' else 0
    projected=draft.hidden_norm(draft.fc(hidden[:,first:]))
    positions=torch.arange(first,position+1)[None]
    expected=draft.layers[0](hidden_states=embedding,target_hidden=projected,
        position_ids=positions,position_embeddings=draft.rotary_emb(embedding,positions),
        attention_mask=torch.ones(1,1,1,position-first+1,dtype=torch.bool))[:,0]
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    assert actual.shape==(1,draft.config.hidden_size)
    assert cache.get_seq_length()==position
    for (a,b),(c,d) in zip(before,cache_pairs(cache)):
        torch.testing.assert_close(a,c)
        torch.testing.assert_close(b,d)


def test_confirmed_context_cache_matches_native_block_for_every_layer():
    target,draft=models()
    ids=torch.arange(1,21)[None]
    hidden=context(target,draft,ids)
    materialized,native=DynamicCache(),DynamicCache()
    materialize_context(draft,materialized,hidden[:,:9],0)
    materialize_context(draft,materialized,hidden[:,9:],9)
    block=torch.tensor([[2,40,40,40,40]])
    with torch.no_grad():
        draft(noise_embedding=target.get_input_embeddings()(block),target_hidden=hidden,
            position_ids=torch.arange(ids.shape[1]+block.shape[1])[None],past_key_values=native,use_cache=True)
    native.crop(ids.shape[1])
    for (a,b),(c,d) in zip(cache_pairs(materialized),cache_pairs(native)):
        torch.testing.assert_close(a,c,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(b,d,atol=2e-6,rtol=2e-6)


def test_selector_reads_do_not_change_subsequent_variable_width_generation():
    target,draft=models()
    ids=torch.arange(1,21)[None]
    with torch.no_grad():
        a,b=RoundRuntime(target,draft,ids),RoundRuntime(target,draft,ids)
        for k in [4,16,32,8]:
            b.dflash_length_features('full_kv')
            b.dflash_length_features('window_16')
            pa,pb=a.propose(k),b.propose(k)
            torch.testing.assert_close(pa,pb)
            ra,rb=a.verify(pa,keep_cache=True),b.verify(pb,keep_cache=True)
            assert ra[0]==rb[0]
            a.advance(pa,ra,cache_kept=True)
            b.advance(pb,rb,cache_kept=True)
            torch.testing.assert_close(a.prefix,b.prefix)


def test_last16_changes_only_visibility_not_absolute_rope_positions():
    target,draft=models()
    ids=torch.arange(1,25)[None]
    hidden=context(target,draft,ids)
    cache=DynamicCache()
    materialize_context(draft,cache,hidden,0)
    embedding=target.get_input_embeddings()(torch.tensor([[7]]))
    actual=read_anchor(draft,cache,embedding,24,'window_16')
    projected=draft.hidden_norm(draft.fc(hidden[:,-16:]))
    wrong_positions=torch.arange(17)[None]
    wrong=draft.layers[0](hidden_states=embedding,target_hidden=projected,
        position_ids=wrong_positions,position_embeddings=draft.rotary_emb(embedding,wrong_positions),
        attention_mask=torch.ones(1,1,1,17,dtype=torch.bool))[:,0]
    # Global translation of Q AND K is RoPE-equivalent; never mix absolute K and reset Q.
    torch.testing.assert_close(actual,wrong,atol=2e-6,rtol=2e-6)
    from adaptive_length.two_stage.dflash_features import _ReadOnlyPrefix
    keys,values=cache_pairs(cache)[0]
    mixed_positions=torch.tensor([[16]])  # reset Q only, while cached K remains absolute
    inconsistent=draft.layers[0](hidden_states=embedding,
        target_hidden=embedding.new_empty(1,0,16),
        position_ids=mixed_positions,position_embeddings=draft.rotary_emb(embedding,mixed_positions),
        past_key_value=_ReadOnlyPrefix(keys[...,-16:,:],values[...,-16:,:]),
        attention_mask=torch.ones(1,1,1,17,dtype=torch.bool))[:,0]
    assert not torch.allclose(actual,inconsistent,atol=1e-7,rtol=1e-7)


def test_streamed_training_features_match_inference_features():
    target,draft=models()
    rows=[dict(prompt_key='p',prefix_ids=list(range(1,n+1)),context_length=n,anchor_id=7) for n in [3,9,22]]
    for row,features,_ in iter_dflash_states(target,draft,rows):
        runtime=RoundRuntime(target,draft,torch.tensor([row['prefix_ids']]))
        runtime.anchor=torch.tensor([[row['anchor_id']]])
        for mode in features:
            torch.testing.assert_close(features[mode],runtime.dflash_length_features(mode),atol=2e-6,rtol=2e-6)


def test_only_length_head_gets_gradients_and_checkpoint_factory_roundtrips():
    target,draft=models()
    rows=[dict(prompt_key='p',prefix_ids=[1,2,3],context_length=3,anchor_id=7)]
    _,features,_=next(iter_dflash_states(target,draft,rows))
    model=DFlashLengthClassifier(16,'full_kv')
    loss=torch.nn.functional.cross_entropy(model(features['full_kv']),torch.tensor([2]))
    loss.backward()
    assert model.head[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in target.parameters())
    assert all(p.grad is None for p in draft.parameters())
    restored=build_classifier(model.settings)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(features['full_kv']),restored(features['full_kv']))
    assert set(restored.state_dict())==set(model.state_dict())
    assert not any('q_proj' in name or 'k_proj' in name for name in restored.state_dict())


def test_training_checkpoint_and_real_dynamic_eval_roundtrip(tmp_path,monkeypatch):
    import json
    from adaptive_length.two_stage.distributed_train import train_models
    from adaptive_length import pipeline
    from adaptive_length.two_stage import cli
    target,draft=models()
    rows={split:[dict(prompt_key=split,prefix_ids=list(range(1,n+1)),context_length=n,anchor_id=7,
        label_accept_k=12,label_speed_k=8,speed_scores=[1.,4.,3.,2.,1.,1.,1.,1.]) for n in [3,6]]
        for split in ('train','val')}
    args=SimpleNamespace(output=str(tmp_path/'heads'),resume=False,data_sha256='fixture',seed=42,
        accumulate=2,lr=.001,epochs=1,log_every=1,log_seconds=1,save_every=1,reader='dflash')
    train_models(args,target,dict(target='target',draft='draft'),rows,draft=draft)
    paths=sorted((tmp_path/'heads').glob('*_accept.pt'))+sorted((tmp_path/'heads').glob('*_speed.pt'))
    assert len(paths)==4
    for path in paths:
        ckpt=torch.load(path,weights_only=True)
        assert ckpt['settings']['kind']=='dflash_layer0_anchor_only_v1'
        assert sum(ckpt['validation']['predicted_k_counts'].values())==2
    monkeypatch.setattr(cli,'model_identity',lambda path:path)
    monkeypatch.setattr(pipeline,'setup',lambda args:(target,draft,SimpleNamespace(eos_token_id=None),[('test',[])],0,1))
    monkeypatch.setattr(pipeline,'encode',lambda *args:torch.tensor([[1,2,3]]))
    evaluation=SimpleNamespace(target='target',draft='draft',head=None,classifier=list(map(str,paths)),
        enable_thinking=False,lengths=[4,8,12,16,20,24,28,32],baseline_lengths=[16],exact_lengths=True,
        fixed_lengths=None,output=str(tmp_path/'eval'),profile_rounds=False,strict_match=False,
        warmup=True,seed=42,max_new_tokens=40)
    pipeline.evaluate(evaluation)
    records=[json.loads(v) for v in (tmp_path/'eval/results_rank000.jsonl').read_text().splitlines()]
    assert len(records)==6  # AR, fixed16, four native predictors
    assert all(r['tokens']==40 for r in records)
    assert all(r['selector_calls']>0 for r in records if r['mode'].startswith('classifier_'))
    assert all(p.grad is None for p in target.parameters())
    assert all(p.grad is None for p in draft.parameters())


def test_small_native_feature_subset_is_trainable_without_backbone_gradients():
    from adaptive_length.two_stage.reader_probe import fit_subset
    target,draft=models()
    ks=[4,8,12,16,20,24,28,32]
    rows=[dict(prompt_key=f'p{i}',prefix_ids=[1,2,3],context_length=3,anchor_id=7+i,
        label_accept_k=ks[i],label_speed_k=ks[7-i]) for i in range(8)]
    result=fit_subset(target,draft,rows,samples=8,steps=80,batch=8,lr=.01,log_every=80)
    assert result['diagnostic_only']
    assert all(v['accuracy']==1 for v in result['metrics'].values())
