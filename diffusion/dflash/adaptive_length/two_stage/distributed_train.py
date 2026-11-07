"""Equal-step DDP training of four KV classifiers with exact state accounting."""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import K_VALUES
from .classifier import KVClassifier, DFlashLengthClassifier
from .cli import iter_states, load_models, model_identity


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def partition_prompts(rows, world, seed):
    """Balance whole trajectories by state count; never split one prompt cache."""
    groups = {}
    for r in rows:
        groups.setdefault(r['prompt_key'], []).append(r)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    # Stable size ordering leaves seed-dependent ordering within ties.
    groups.sort(key=len, reverse=True)
    parts, loads = [[] for _ in range(world)], [0]*world
    for group in groups:
        rank = min(range(world), key=lambda i: (loads[i], i))
        parts[rank].extend(sorted(group, key=lambda r: r['context_length']))
        loads[rank] += len(group)
    return parts


def tensor_sum(value, world):
    if world > 1:
        dist.all_reduce(value)
    return value


def log_due(rank, world, device, due):
    if world == 1:
        return due
    flag = torch.tensor([int(due) if rank == 0 else 0], device=device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def train_models(args, target, manifest, rows, rank=0, world=1, stop_after_updates=None, draft=None):
    """Dependency injection allows real tiny CPU/Gloo end-to-end tests."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = target.device
    config = target.config
    native_reader = getattr(args, 'reader', 'legacy') == 'dflash'
    if native_reader and draft is None:
        raise ValueError('DFlash-aligned features require the frozen draft checkpoint')
    if native_reader:
        from .dflash_features import iter_dflash_states, FEATURE_SCHEMA
    def states_iterator(subset):
        return iter_dflash_states(target,draft,subset) if native_reader else iter_states(target,subset)
    def classifier_input(model,features):
        unwrapped=model.module if isinstance(model,DDP) else model
        return features[unwrapped.context_mode] if native_reader else features
    identity = dict(manifest=manifest, data_sha256=args.data_sha256, world_size=world,
                    seed=args.seed, accumulate=args.accumulate, lr=args.lr)
    if native_reader:
        identity['reader_schema']=FEATURE_SCHEMA
    saved_path = out/'training_state.pt'
    saved = torch.load(saved_path, map_location='cpu', weights_only=True) if args.resume and saved_path.exists() else None
    if saved and saved['identity'] != identity:
        raise ValueError('Resume requires identical models/data/world-size/seed/accumulation/LR')
    if not saved and (saved_path.exists() or any(out.glob('*.pt'))):
        raise ValueError('Existing classifier weights: use --resume for a v2 checkpoint or a fresh output directory')
    if {r['prompt_key'] for r in rows['train']} & {r['prompt_key'] for r in rows['val']}:
        raise ValueError('Train/validation prompt overlap')
    if not rows['train'] or not rows['val']:
        raise ValueError('Need nonempty classifier train and validation states')
    models, objectives = {}, {}
    for context in ('full_kv', 'window_16'):
        for objective in ('accept', 'speed'):
            name = f'{context}_{objective}'
            torch.manual_seed(args.seed)
            if native_reader:
                model = DFlashLengthClassifier(feature_dim=draft.config.hidden_size,context_mode=context).to(device)
            else:
                model = KVClassifier(anchor_dim=config.hidden_size,
                    head_dim=getattr(config,'head_dim',config.hidden_size//config.num_attention_heads),
                    layers=config.num_hidden_layers,kv_heads=config.num_key_value_heads,
                    context_mode=context).to(device)
            if saved:
                model.load_state_dict(saved['models'][name])
            models[name] = DDP(model, device_ids=[device.index] if device.type=='cuda' else None,
                               broadcast_buffers=False) if world>1 else model
            objectives[name] = objective
    base = lambda m: m.module if isinstance(m,DDP) else m
    optimizers={name:torch.optim.AdamW(model.parameters(),lr=args.lr) for name,model in models.items()}
    if saved:
        for name,opt in optimizers.items():
            opt.load_state_dict(saved['optimizers'][name])
    best=saved['best'] if saved else {name:float('inf') for name in models}
    start_epoch=saved['next_epoch'] if saved else 0
    resume_position=saved['next_position'] if saved else 0
    updates=saved['updates'] if saved else 0
    durations=saved.get('durations',{}) if saved else {}
    started=time.monotonic()
    ntrain,nval=len(rows['train']),len(rows['val'])
    used_keys=sorted({r['prompt_key'] for rs in rows.values() for r in rs})
    last_log=0.
    latest_losses=saved.get('latest_losses',{}) if saved else {}
    latest_validation=saved.get('latest_validation',{}) if saved else {}
    latest_train_accuracy=saved.get('latest_train_accuracy',{}) if saved else {}

    def emit(record):
        nonlocal last_log,latest_losses,latest_validation,latest_train_accuracy
        if rank:
            return
        if 'train_loss' in record:
            latest_losses=record['train_loss']
        if 'validation' in record:
            latest_validation=record['validation']
        if 'train_accuracy' in record:
            latest_train_accuracy=record['train_accuracy']
        record.update(last_train_loss=latest_losses,last_validation=latest_validation,last_train_accuracy=latest_train_accuracy)
        record.update(majority_accuracy=majority_accuracy,
                      reader_schema=FEATURE_SCHEMA if native_reader else "legacy_target_kv",
                      world_size=world,epochs=args.epochs,global_batch=world*args.accumulate,optimizer_updates=updates,
                      run_elapsed_s=round(time.monotonic()-started,2),updated_at=time.time())
        print(json.dumps(record),flush=True)
        atomic_json(record,out/'progress.json')
        last_log=time.monotonic()

    def save(next_epoch,next_position):
        if rank:
            return
        atomic_save(dict(schema=2,identity=identity,next_epoch=next_epoch,next_position=next_position,
            updates=updates,best=best,durations=durations,latest_losses=latest_losses,latest_validation=latest_validation,
            latest_train_accuracy=latest_train_accuracy,
            models={name:base(m).state_dict() for name,m in models.items()},
            optimizers={name:opt.state_dict() for name,opt in optimizers.items()}),saved_path)
        print(json.dumps(dict(event='classifier_resume_checkpoint',epoch=next_epoch+1,
            next_local_state=next_position,optimizer_updates=updates,path=str(saved_path))),flush=True)

    label_counts={split:{objective:dict(Counter(r[f'label_{objective}_k'] for r in records))
        for objective in ('accept','speed')} for split,records in rows.items()}
    majority_accuracy={split:{objective:max(counts.values())/len(rows[split])
        for objective,counts in by_objective.items()} for split,by_objective in label_counts.items()}
    emit(dict(event='classifier_start',phase='setup',train_states=ntrain,val_states=nval,
        label_counts=label_counts,majority_accuracy=majority_accuracy,
        train_prompts=len({r['prompt_key'] for r in rows['train']}),
        resumed=bool(saved),epoch=start_epoch+1,next_local_state=resume_position,
        global_batch=world*args.accumulate))
    if start_epoch>=args.epochs:
        emit(dict(event='classifier_complete',phase='done',epoch=start_epoch,eta_s=0))
        if rank==0:
            atomic_json(dict(epochs=start_epoch,identity=identity),out/'completed.json')
        return
    val_parts=partition_prompts(rows['val'],world,args.seed)
    maxval=max(map(len,val_parts))
    for epoch in range(start_epoch,args.epochs):
        parts=partition_prompts(rows['train'],world,args.seed+epoch)
        lengths=list(map(len,parts))
        maxsteps=max(lengths)
        cursor=resume_position if epoch==start_epoch else 0
        if cursor>maxsteps or (cursor%args.accumulate and cursor!=maxsteps):
            raise ValueError('Invalid checkpoint cursor: expected an optimizer update boundary')
        local_rows=parts[rank]
        iterator=iter(states_iterator(local_rows[cursor:]))
        fallback=None
        phase_start=time.monotonic()
        prior_states=sum(min(cursor,n) for n in lengths)
        # Model KV is reconstructed only for remaining states on resume.
        emit(dict(event='classifier_progress',phase='train',epoch=epoch+1,states=prior_states,
                  total_states=ntrain,percent=100*prior_states/ntrain,eta_s=None))
        window=torch.zeros(2*len(models)+1,device=device)
        active_in_update=0
        for m in models.values():
            m.train()
        for i in range(cursor,maxsteps):
            active=i<len(local_rows)
            if active:
                row,pairs,anchor=next(iterator)
                fallback=(row,pairs,anchor)
            elif fallback is None:
                fallback=next(iter(states_iterator([rows['train'][0]])))
                row,pairs,anchor=fallback
            else:
                row,pairs,anchor=fallback
            boundary=(i+1)%args.accumulate==0 or i+1==maxsteps
            for j,(name,model) in enumerate(models.items()):
                sync=model.no_sync() if isinstance(model,DDP) and not boundary else nullcontext()
                with sync:
                    label=K_VALUES.index(row[f'label_{objectives[name]}_k'])
                    logits=model(classifier_input(model,pairs),anchor)
                    ce=torch.nn.functional.cross_entropy(logits,torch.tensor([label],device=device))
                    loss=ce*(float(active)/args.accumulate)
                    loss.backward()
                if active:
                    window[j]+=ce.detach()
                    window[len(models)+j]+=(logits.detach().argmax(-1)==label).sum()
            if active:
                window[-1]+=1
                active_in_update+=1
            if not boundary:
                continue
            # Correct DDP averaging for padded tail states and the partial last batch.
            population=tensor_sum(torch.tensor(float(active_in_update),device=device),world).item()
            correction=args.accumulate*world/population
            for name,model in models.items():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizers[name].step()
                optimizers[name].zero_grad(set_to_none=True)
            active_in_update=0
            updates+=1
            processed=sum(min(i+1,n) for n in lengths)
            due=(updates%args.log_every==0 or time.monotonic()-last_log>=args.log_seconds or i+1==maxsteps)
            if log_due(rank,world,device,due):
                values=tensor_sum(window.clone(),world).tolist()
                elapsed=time.monotonic()-phase_start
                rate=(processed-prior_states)/elapsed if elapsed else 0
                val_seconds=durations.get('val',nval/rate if rate else 0)
                eta=((ntrain-processed+(args.epochs-epoch-1)*ntrain)/rate
                    +(args.epochs-epoch)*val_seconds) if rate else None
                emit(dict(event='classifier_progress',phase='train',epoch=epoch+1,states=processed,
                    total_states=ntrain,percent=round(100*processed/ntrain,2),
                    overall_train_percent=round(100*(epoch*ntrain+processed)/(args.epochs*ntrain),2),
                    states_per_s=round(rate,3),eta_s=round(eta) if eta is not None else None,
                    eta_scope='remaining classifier training and validation; excludes evaluation',
                    train_loss={name:values[j]/max(values[-1],1) for j,name in enumerate(models)},
                    train_accuracy={name:values[len(models)+j]/max(values[-1],1) for j,name in enumerate(models)},
                    per_rank_states=lengths))
                window.zero_()
            if updates%args.save_every==0 or i+1==maxsteps:
                save(epoch,i+1)
            if stop_after_updates is not None and updates>=stop_after_updates:
                save(epoch,i+1)
                emit(dict(event='classifier_paused',phase='paused',epoch=epoch+1,next_local_state=i+1))
                return
        if cursor<maxsteps:
            durations['train']=ntrain/max(1,ntrain-prior_states)*(time.monotonic()-phase_start)
        val_start=time.monotonic()
        emit(dict(event='classifier_progress',phase='validation',epoch=epoch+1,states=0,total_states=nval,eta_s=None))
        iterator=iter(states_iterator(val_parts[rank]))
        totals=torch.zeros((len(models),4),device=device)
        predictions=torch.zeros((len(models),len(K_VALUES)),device=device)
        for model in models.values():
            model.eval()
        with torch.no_grad():
            for i in range(maxval):
                if i<len(val_parts[rank]):
                    row,pairs,anchor=next(iterator)
                    for j,(name,model) in enumerate(models.items()):
                        logits=base(model)(classifier_input(model,pairs),anchor)  # no DDP collectives on uneven validation shards
                        label=K_VALUES.index(row[f'label_{objectives[name]}_k'])
                        prediction=logits.argmax(-1)
                        predictions[j].index_add_(0,prediction,torch.ones_like(prediction,dtype=predictions.dtype))
                        scores=torch.tensor(row['speed_scores'],device=device)
                        totals[j,0]+=torch.nn.functional.cross_entropy(logits,torch.tensor([label],device=device))
                        totals[j,1]+=(prediction==label).sum()
                        totals[j,2]+=scores[prediction].sum()/scores.max()
                        totals[j,3]+=1
                if i==0 or (i+1)%16==0 or i+1==maxval:
                    due=time.monotonic()-last_log>=args.log_seconds or i+1==maxval
                    if log_due(rank,world,device,due):
                        values=tensor_sum(totals.clone(),world).tolist()
                        predicted=tensor_sum(predictions.clone(),world).tolist()
                        done=int(values[0][3])
                        rate=done/max(time.monotonic()-val_start,1e-9)
                        remaining_train=(args.epochs-epoch-1)*durations.get('train',0)
                        eta=(nval-done)/rate+(args.epochs-epoch-1)*nval/rate+remaining_train
                        emit(dict(event='classifier_progress',phase='validation',epoch=epoch+1,
                            states=done,total_states=nval,percent=round(100*done/nval,2),eta_s=round(eta),
                            validation={name:dict(loss=v[0]/v[3],accuracy=v[1]/v[3],speed_ratio=v[2]/v[3],
                                predicted_k_counts={str(k):int(c) for k,c in zip(K_VALUES,hist)})
                                for name,v,hist in zip(models,values,predicted)}))
        values=tensor_sum(totals,world).tolist()
        predicted=tensor_sum(predictions,world).tolist()
        durations['val']=time.monotonic()-val_start
        for (name,model),v,hist in zip(models.items(),values,predicted):
            if int(v[3])!=nval:
                raise RuntimeError('Validation duplicated or lost states')
            metrics=dict(loss=v[0]/v[3],accuracy=v[1]/v[3],speed_ratio=v[2]/v[3],
                predicted_k_counts={str(k):int(c) for k,c in zip(K_VALUES,hist)})
            if rank==0:
                print(json.dumps(dict(event='classifier_validation',epoch=epoch+1,classifier=name,**metrics)),flush=True)
            if metrics['loss']<best[name]:
                best[name]=metrics['loss']
                if rank==0:
                    atomic_save(dict(settings=base(model).settings,state_dict=base(model).state_dict(),metadata=manifest,
                        objective=objectives[name],epoch=epoch+1,validation=metrics,used_prompt_keys=used_keys),out/f'{name}.pt')
        resume_position=0
        save(epoch+1,0)
    emit(dict(event='classifier_complete',phase='done',epoch=args.epochs,eta_s=0))
    if rank==0:
        atomic_json(dict(epochs=args.epochs,identity=identity),out/'completed.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('target','draft','data','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--accumulate',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--log-every',type=int,default=50,help='Optimizer updates between reports')
    p.add_argument('--log-seconds',type=float,default=15)
    p.add_argument('--save-every',type=int,default=100,help='Optimizer updates between resume checkpoints')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--reader',choices=['legacy','dflash'],default='legacy')
    args=p.parse_args()
    if min(args.epochs,args.accumulate,args.log_every,args.log_seconds,args.save_every)<1:
        p.error('Training/log/checkpoint limits must be positive')
    world=int(os.getenv('WORLD_SIZE','1'))
    rank=int(os.getenv('RANK','0'))
    if not torch.cuda.is_available():
        raise RuntimeError('Classifier CLI requires CUDA; tests inject tiny CPU targets directly')
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK','0')))
    if world>1:
        dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    try:
        root=Path(args.data)
        if rank==0:
            print(json.dumps(dict(event='classifier_setup',phase='checking_data_and_fingerprints',world_size=world)),flush=True)
        manifest=json.loads((root/'manifest.json').read_text())
        check=None
        if rank==0:
            try:
                if manifest['target']!=model_identity(args.target) or manifest['draft']!=model_identity(args.draft):
                    raise ValueError('Frozen model fingerprints differ from labels')
                check=dict(ok=True)
            except Exception as e:
                check=dict(ok=False,error=str(e))
        status=[check]
        if world>1:
            dist.broadcast_object_list(status,src=0)
        if not status[0]['ok']:
            raise ValueError(status[0]['error'])
        digest=hashlib.sha256()
        rows={}
        for split in ('train','val'):
            data=(root/f'{split}.jsonl').read_bytes()
            digest.update(split.encode()+data)
            rows[split]=[json.loads(line) for line in data.decode().splitlines()]
        args.data_sha256=digest.hexdigest()
        if rank==0:
            print(json.dumps(dict(event='classifier_setup',phase='loading_frozen_target',
                train_states=len(rows['train']),val_states=len(rows['val']),world_size=world,
                global_batch=world*args.accumulate)),flush=True)
        target,draft=load_models(args,draft=args.reader=='dflash')
        train_models(args,target,manifest,rows,rank,world,draft=draft)
    finally:
        if world>1:
            dist.destroy_process_group()


if __name__=='__main__':
    main()
