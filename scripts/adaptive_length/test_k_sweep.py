import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
import report_k_sweep as report
import benchmark_k_sweep as bench
import prepare_gsm8k_oracle as prepare


def record(arm,k,repeat=0):
    rd=dict(round=1,requested_k=k,actual_k=k,accepted=2,accepted_raw=2,committed=3,tail=False)
    return dict(arm=arm,k=k,prompt_key='p',repeat=repeat,tokens=3,
        decode_s=1 if arm=='fixed' else .8,spec_rounds=1,accepted_tokens=2,
        output_sha256='same',round_details=[rd])


def records():
    return [record(arm,k,repeat) for arm in ('fixed','mixed') for k in report.KS for repeat in range(2)]


def test_summary_uses_same_k_and_validates_repeated_outputs():
    rows=report.summarize(records())
    assert len(rows)==8
    assert all(r['mixed_gain_pct']==pytest.approx(25) for r in rows)
    assert all(r['mixed_faster_prompts']==1 for r in rows)
    bad=records()
    bad[1]['output_sha256']='different'
    with pytest.raises(ValueError,match='trajectory changed'):
        report.summarize(bad)


def test_distribution_normalizes_columns_not_counts_repeats():
    rows,skipped=report.acceptance_distribution(records())
    assert not skipped
    for arm in ('fixed','mixed'):
        for k in report.KS:
            col=[r for r in rows if r['arm']==arm and r['k']==k]
            assert sum(r['count'] for r in col)==1
            assert sum(r['probability'] for r in col)==pytest.approx(1)
    rs=[record('fixed',16),record('mixed',16)]
    rs[0]['round_details'][0]['actual_k']=7
    rs[1]['round_details'][0]['accepted']=0
    _,skipped=report.acceptance_distribution(rs)
    assert skipped=={'ar_or_tail_clipped':1,'eos_adjusted':1}


def test_common_state_comparison_includes_decreasing_acceptance():
    a=[4,6,5,7,6,5,4,3]
    row={arm:dict(accepted_lengths=a.copy(),raw_accepted_lengths=a.copy()) for arm in ('fixed','mixed')}
    row['mixed']['accepted_lengths'][0]=3
    row['mixed']['raw_accepted_lengths'][0]=3
    means,changes,scope=report.paired_stats([row])
    assert scope['used']==1
    assert means[0]['mixed_worse']==1
    p=next(r for r in changes if r['arm']=='fixed' and r['small_k']==8 and r['large_k']==12)
    assert p['larger_worse']==1
    eos=copy.deepcopy(row)
    eos['mixed']['accepted_lengths'][0]=1
    means,_,scope=report.paired_stats([row,eos])
    assert scope['used']==1 and scope['eos_excluded']==1


def test_local_gsm8k_turns_keeps_instruction():
    text='How many?\nPlease reason step by step.'
    rows=prepare.prepare([dict(turns=[text])],1)
    assert rows[0]['prompt'][0]['content']==text
    with pytest.raises(ValueError,match='single-turn'):
        prepare.prepare([dict(turns=['a','b'])],1)


def test_report_cli_draws_two_heatmaps_and_csv(tmp_path):
    for name in ('mixed_holdout','gsm8k'):
        folder=tmp_path/name
        folder.mkdir()
        (folder/'complete.json').write_text('{}')
        (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records()))
    r=subprocess.run([sys.executable,str(HERE/'report_k_sweep.py'),'--root',str(tmp_path)],
                     capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert (tmp_path/'throughput_all.csv').exists()
    for name in ('mixed_holdout','gsm8k'):
        assert (tmp_path/f'{name}_acceptance_heatmap.png').stat().st_size>1000
        assert (tmp_path/f'{name}_acceptance_heatmap.pdf').stat().st_size>1000


def test_real_common_ar_probes_after_multiple_ar_steps():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from model.dflash import DFlashDraftModel
    from adaptive_length.runtime import RoundRuntime
    from adaptive_length.two_stage.cli import paired_measure
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config=Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        head_dim=8,max_position_embeddings=128,attn_implementation='sdpa')
    target=Qwen3ForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad(): target.lm_head.weight.zero_()
    dc=Qwen3Config(**config.to_dict())
    dc.num_hidden_layers,dc.num_target_layers,dc.block_size=1,2,33
    dc.layer_types=['full_attention']
    dc.dflash_config=dict(target_layer_ids=[0],mask_token_id=30)
    dc._attn_implementation='sdpa'
    draft=DFlashDraftModel(dc).eval().requires_grad_(False)
    args=SimpleNamespace(seed=42,max_new_tokens=40,probe_states=3,probe_stride=3,probe_repeats=2)
    with torch.inference_mode():
        rows=bench.collect_paired(target,{'fixed':draft,'mixed':copy.deepcopy(draft)},
            torch.tensor([[1,2,3]]),args,None,RoundRuntime,paired_measure,torch)
    assert [r['generated_position'] for r in rows]==[0,3,6]
    for r in rows:
        assert r['fixed']['accepted_lengths']==report.KS
        assert r['mixed']['accepted_lengths']==report.KS
        assert r['anchor_id']==0
