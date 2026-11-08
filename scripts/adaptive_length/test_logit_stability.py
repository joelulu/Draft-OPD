import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import pytest

sys.path.insert(0,str(Path(__file__).parent))
import logit_stability as exp


@pytest.fixture
def tiny():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from model.dflash import DFlashDraftModel
    from adaptive_length.runtime import RoundRuntime
    torch.set_num_threads(1)
    torch.manual_seed(5)
    config=Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        head_dim=8,max_position_embeddings=128,attn_implementation='sdpa')
    target=Qwen3ForCausalLM(config).eval().requires_grad_(False)
    dc=Qwen3Config(**config.to_dict())
    dc.num_hidden_layers,dc.num_target_layers,dc.block_size=1,2,33
    dc.layer_types=['full_attention']
    dc.dflash_config=dict(target_layer_ids=[0],mask_token_id=30)
    dc._attn_implementation='sdpa'
    draft=DFlashDraftModel(dc).eval().requires_grad_(False)
    return torch,target,draft,RoundRuntime


def test_features_full_distribution_and_shift_invariance(tiny):
    torch,_,_,_=tiny
    values=torch.randn(1,4,31)
    f=exp.logit_features(values,torch)
    shifted=exp.logit_features(values+8,torch)
    c=exp.compare_features(f,shifted,torch)
    assert c['mean_js']<1e-6
    assert c['top1_change']==0
    assert c['top5_distance']==0
    assert c['peak_delta']<1e-6
    same=exp.logit_features(torch.cat([values,torch.randn(1,4,31)],1),torch)
    assert exp.compare_features(f,same,torch)['mean_js']==pytest.approx(0,abs=1e-7)
    tied=exp.logit_features(torch.zeros(1,4,31),torch)
    assert tied['ids'][0].tolist()==[0,1,2,3,4]


@pytest.mark.parametrize('metric',['top1_change','top5_distance','mean_js','peak_delta'])
def test_online_signal_matches_collected_signal(tiny,metric):
    torch,_,_,_=tiny
    a,b=torch.randn(1,4,31),torch.randn(1,8,31)
    ba=torch.cat([torch.ones(1,1,dtype=torch.long),a.argmax(-1)],1)
    bb=torch.cat([torch.ones(1,1,dtype=torch.long),b.argmax(-1)],1)
    expected=exp.compare_features(exp.logit_features(a,torch),exp.logit_features(b,torch),torch)[metric]
    actual=exp.feature_change(exp.rule_feature(a,ba,metric,torch),exp.rule_feature(b,bb,metric,torch),metric,torch)
    assert actual==pytest.approx(expected,abs=1e-6)


def test_collector_records_all_K_positions_reference_and_repeat_noise(tiny):
    torch,target,draft,Runtime=tiny
    with torch.no_grad(): target.lm_head.weight.zero_()
    with torch.inference_mode():
        row=exp.collect_state(Runtime(target,draft,torch.tensor([[1,2,3]])),[0]*32,2,
            __import__('random').Random(42),None,time.perf_counter,torch)
    for k in exp.KS:
        t=row['trials'][str(k)]
        assert t['accepted']==k
        assert len(t['positions'])==k
        assert t['positions'][0]['top5_ids'][0]==t['block_ids'][1]==0
        assert t['reference_prefix_accept']==k
        assert t['repeat_js_max']==pytest.approx([0],abs=1e-7)
    assert row['pairs']['4:8']['top1_change']==0
    assert row['pairs']['4:32']['mean_js']==pytest.approx(0,abs=1e-7)


def test_random_model_reference_prefix_hits_match_actual_acceptance(tiny):
    torch,target,draft,Runtime=tiny
    ids=torch.tensor([[1,2,3]])
    with torch.inference_mode():
        reference=exp.reference_sequence(target,draft,ids,33,None,Runtime)
        state=Runtime(target,draft,ids)
        assert int(state.anchor)==reference[0]
        row=exp.collect_state(state,reference[1:],1,__import__('random').Random(42),
            None,time.perf_counter,torch)
    for k in exp.KS:
        trial=row['trials'][str(k)]
        assert trial['reference_available']==k
        assert trial['reference_prefix_accept']==trial['accepted_raw']


def synthetic_state(arm='fixed',split='cal',key='cal'):
    trials={str(k):dict(accepted=k,accepted_raw=k,eos_adjusted=False,
        draft_ms=[1.],verify_ms=[10.],positions=[dict(position=i+1,
            top5_ids=[0,1,2,3,4],top5_logits=[0.]*5,top5_probs=[.1]*5)
            for i in range(k)],repeat_js_max=[0.]) for k in exp.KS}
    pairs={f'{a}:{b}':dict(top1_change=0.,top5_distance=0.,mean_js=0.,peak_delta=0.,
        small_k=a,large_k=b,adjacent=exp.KS.index(b)==exp.KS.index(a)+1,accept_gain=b-a,positions=[])
        for a,b in ({(exp.KS[i],exp.KS[i+1]) for i in range(7)}|{(k,32) for k in exp.KS[:-1]})}
    return dict(arm=arm,split=split,prompt_key=key,generated_position=0,
                prefix_sha256=key,anchor_id=1,context_prep_ms=1.,trials=trials,pairs=pairs)


def test_selection_never_uses_future_or_target_labels():
    row=synthetic_state()
    rule=dict(metric='top1_change',threshold=0.,patience=1)
    assert exp.choose_from_state(row,rule)==(8,[4,8])
    altered=copy.deepcopy(row)
    altered['trials']['32']['accepted']=0
    altered['pairs']['4:32']['top1_change']=1
    assert exp.choose_from_state(altered,rule)==(8,[4,8])
    metrics=exp.rule_summary([row],rule)
    assert metrics['mean_draft_calls']==2
    assert metrics['harmful_stop_rate']==1
    assert metrics['mean_forgone_future_accept']==24


def test_analysis_uses_calibration_only_and_reports_capacity_false_stop(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps(dict(fingerprints={})))
    (tmp_path/'collect_complete.json').write_text('{}')
    (tmp_path/'token_legend.json').write_text(json.dumps({str(i):str(i) for i in range(5)}))
    rows=[synthetic_state(arm,split,split) for arm in ('fixed','mixed') for split in ('cal','test')]
    path=tmp_path/'states.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    args=SimpleNamespace(output=str(tmp_path))
    exp.analyze(args)
    first=json.loads((tmp_path/'chosen_rules.json').read_text())
    for arm in ('fixed','mixed'):
        assert not first['choices'][arm]['calibration_retention_target_met']
        assert first['choices'][arm]['test']['harmful_stop_rate']==1
    for r in rows:
        if r['split']=='test':
            for p in r['pairs'].values():
                for metric in ('top1_change','top5_distance','mean_js','peak_delta'): p[metric]=1
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    exp.analyze(args)
    second=json.loads((tmp_path/'chosen_rules.json').read_text())
    for arm in ('fixed','mixed'):
        assert first['choices'][arm]['rule']==second['choices'][arm]['rule']


def test_real_online_uses_multiple_drafts_but_verifies_once_per_round(tiny):
    torch,target,draft,Runtime=tiny
    with torch.no_grad(): target.lm_head.weight.zero_()
    class CountRuntime(Runtime):
        verifications=0
        def verify(self,*args,**kwargs):
            type(self).verifications+=1
            return super().verify(*args,**kwargs)
    args=SimpleNamespace(max_new_tokens=40)
    rule=dict(metric='top1_change',threshold=0.,patience=1)
    with torch.inference_mode():
        row=exp.adaptive_decode(target,draft,torch.tensor([[1,2]]),rule,args,None,
                               CountRuntime,time.perf_counter,torch)
    assert row['tokens']==40
    assert row['draft_calls']>row['spec_rounds']
    assert CountRuntime.verifications==row['spec_rounds']
    assert all(len(r['tried'])<=8 for r in row['round_details'])
    assert all(p.grad is None for p in target.parameters())
