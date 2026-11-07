"""Two real CPU/Gloo ranks: uneven shards, all four heads, mid-epoch resume."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen3Config,Qwen3ForCausalLM

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adaptive_length.two_stage.distributed_train import partition_prompts,train_models


def dataset():
    rows={}
    for split,groups in [('train',[3,2]),('val',[2])]:
        states=[]
        for group,count in enumerate(groups):
            for i in range(count):
                prefix=list(range(1,4+i))
                states.append(dict(state_id=f'{split}_{group}_{i}',prompt_key=f'{split}_{group}',
                    prefix_ids=prefix,context_length=len(prefix),anchor_id=4,
                    label_accept_k=12,label_speed_k=8,speed_scores=[1.,4.,3.,2.,1.,1.,1.,1.]))
        rows[split]=states
    return rows


def worker(rank,world,root,store):
    torch.set_num_threads(1)
    backend=os.getenv('KV_CLASSIFIER_TEST_BACKEND','gloo')
    if backend=='file_test':
        from file_process_group import register
        register()
    dist.init_process_group(backend,rank=rank,world_size=world,init_method=f'file://{store}')
    try:
        torch.manual_seed(17)
        target=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=8,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=4,
            max_position_embeddings=128,attn_implementation='sdpa')).eval().requires_grad_(False)
        args=SimpleNamespace(output=str(Path(root)/'full'),resume=False,data_sha256='fixture',seed=42,
            accumulate=2,lr=.001,epochs=2,log_every=1,log_seconds=1.,save_every=1)
        manifest=dict(target='tiny',draft='tiny')
        rows=dataset()
        train_models(args,target,manifest,rows,rank,world)
        dist.barrier()
        args.output=str(Path(root)/'resumed')
        train_models(args,target,manifest,rows,rank,world,stop_after_updates=1)
        dist.barrier()
        args.resume=True
        train_models(args,target,manifest,rows,rank,world)
        assert all(p.grad is None for p in target.parameters())
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_whole_prompts_balanced_without_duplicates():
    rows=dataset()['train']
    parts=partition_prompts(rows,2,42)
    assert sorted(map(len,parts))==[2,3]
    assert len({r['state_id'] for part in parts for r in part})==len(rows)
    assert not {r['prompt_key'] for r in parts[0]} & {r['prompt_key'] for r in parts[1]}
    assert partition_prompts(rows,2,42)==parts


def test_two_rank_resume_matches_uninterrupted_weights_and_optimizer(tmp_path):
    mp.spawn(worker,args=(2,str(tmp_path),str(tmp_path/'store')),nprocs=2,join=True)
    full=torch.load(tmp_path/'full/training_state.pt',weights_only=True)
    resumed=torch.load(tmp_path/'resumed/training_state.pt',weights_only=True)
    assert full['next_epoch']==resumed['next_epoch']==2
    assert full['next_position']==resumed['next_position']==0
    assert full['updates']==resumed['updates']==4
    for name in full['models']:
        for key in full['models'][name]:
            torch.testing.assert_close(full['models'][name][key],resumed['models'][name][key],atol=1e-7,rtol=1e-6)
        a,b=full['optimizers'][name]['state'],resumed['optimizers'][name]['state']
        for key in a:
            for field in a[key]:
                torch.testing.assert_close(a[key][field],b[key][field],atol=1e-7,rtol=1e-6)
        assert (tmp_path/f'resumed/{name}.pt').exists()
    progress=json.loads((tmp_path/'resumed/progress.json').read_text())
    assert progress['phase']=='done' and progress['eta_s']==0
    assert (tmp_path/'resumed/completed.json').exists()
