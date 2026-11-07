"""Fit a small TRAIN-only state subset; never writes inference checkpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch
from . import K_VALUES
from .classifier import DFlashLengthClassifier
from .cli import load_models,model_identity
from .dflash_features import iter_dflash_states,FEATURE_SCHEMA


def fit_subset(target,draft,rows,samples=128,steps=1000,batch=32,lr=1e-3,seed=42,log_every=50):
    chosen=list(rows)
    random.Random(seed).shuffle(chosen)
    chosen=chosen[:samples]
    if not chosen:
        raise ValueError('No training states for the fitting probe')
    values,ordered={'full_kv':[],'window_16':[]},[]
    start=time.monotonic()
    for i,(row,features,_) in enumerate(iter_dflash_states(target,draft,chosen),1):
        ordered.append(row)
        for mode in values:
            values[mode].append(features[mode])
        if i==1 or i%16==0 or i==len(chosen):
            print(json.dumps(dict(event='probe_extract',states=i,total=len(chosen))),flush=True)
    values={mode:torch.cat(v).detach() for mode,v in values.items()}
    labels={objective:torch.tensor([K_VALUES.index(r[f'label_{objective}_k']) for r in ordered],device=target.device)
        for objective in ('accept','speed')}
    models,optimizers={},{}
    for mode in values:
        for objective in labels:
            torch.manual_seed(seed)
            name=f'{mode}_{objective}'
            model=DFlashLengthClassifier(draft.config.hidden_size,mode).to(target.device)
            models[name]=(model,mode,objective)
            optimizers[name]=torch.optim.AdamW(model.parameters(),lr=lr)
    generator=torch.Generator(device=target.device).manual_seed(seed)
    metrics={}
    for step in range(1,steps+1):
        indices=torch.randint(len(ordered),(min(batch,len(ordered)),),device=target.device,generator=generator)
        for name,(model,mode,objective) in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(values[mode][indices]),labels[objective][indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizers[name].step()
        if step==1 or step%log_every==0 or step==steps:
            with torch.no_grad():
                for name,(model,mode,objective) in models.items():
                    logits=model(values[mode])
                    metrics[name]=dict(loss=float(torch.nn.functional.cross_entropy(logits,labels[objective])),
                        accuracy=float((logits.argmax(-1)==labels[objective]).float().mean()))
            print(json.dumps(dict(event='probe_fit',step=step,steps=steps,metrics=metrics,
                elapsed_s=time.monotonic()-start)),flush=True)
    if any(p.grad is not None for p in target.parameters()) or any(p.grad is not None for p in draft.parameters()):
        raise RuntimeError('Frozen backbone received gradients in the probe')
    return dict(diagnostic_only=True,reader_schema=FEATURE_SCHEMA,states=len(ordered),steps=steps,
        train_prompt_keys=sorted({r['prompt_key'] for r in ordered}),metrics=metrics,
        majority_accuracy={objective:max(Counter(r[f'label_{objective}_k'] for r in ordered).values())/len(ordered)
            for objective in labels},
        note='Accuracy is on the same small TRAIN subset used for optimization; not held-out performance.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('target','draft','data','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--log-every',type=int,default=50)
    args=p.parse_args()
    if min(args.samples,args.steps,args.batch,args.log_every)<1 or args.lr<=0:
        p.error('Probe limits must be positive')
    root=Path(args.data)
    manifest=json.loads((root/'manifest.json').read_text())
    if model_identity(args.target)!=manifest['target'] or model_identity(args.draft)!=manifest['draft']:
        raise ValueError('Probe models differ from label collection')
    target,draft=load_models(args,draft=True)
    rows=[json.loads(v) for v in (root/'train.jsonl').read_text().splitlines()]
    report=fit_subset(target,draft,rows,args.samples,args.steps,args.batch,args.lr,args.seed,args.log_every)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(f'Diagnostic only: {output}',flush=True)


if __name__=='__main__':
    main()
