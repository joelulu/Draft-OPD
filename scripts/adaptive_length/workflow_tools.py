"""Standard-library workflow status and paired three-arm reporting."""
import argparse
import csv
import json
import time
from pathlib import Path


def write_json(value,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def duration(seconds):
    if seconds is None:
        return 'estimating'
    seconds=max(0,int(seconds))
    return f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'


def status(root,classifiers):
    state=root/'workflow.json'
    if state.exists():
        value=json.loads(state.read_text())
        print(f"Workflow: {value['stage']} ({value['status']})")
    for arm in ('fixed','mixed'):
        print(f"{arm} draft exported: {(root/'models'/arm/'model.safetensors').exists()}")
    for split in ('train','val','test'):
        path=root/'labeled'/f'{split}.jsonl'
        if path.exists():
            with path.open() as stream:
                count=sum(bool(line.strip()) for line in stream)
            print(f'{split} labeled states: {count}')
    progress=classifiers/'progress.json'
    if progress.exists():
        row=json.loads(progress.read_text())
        print(f"Classifier phase: {row['phase']}; epoch {row.get('epoch','?')}/{row.get('epochs','?')}; GPUs {row.get('world_size','?')}")
        if 'states' in row:
            print(f"States: {row['states']}/{row['total_states']} ({row.get('percent',0):.2f}%)")
        if "overall_train_percent" in row:
            print(f"Overall training states: {row['overall_train_percent']:.2f}%")
        print(f"Remaining classifier training + validation: {duration(row.get('eta_s'))}")
        print(f"Last report age: {duration(time.time()-row.get('updated_at',time.time()))}")
        for name,loss in (row.get('train_loss') or row.get('last_train_loss',{})).items():
            print(f'  train {name}: loss={loss:.6f}')
        for name,metrics in (row.get('validation') or row.get('last_validation',{})).items():
            print(f"  val {name}: loss={metrics['loss']:.6f}, accuracy={metrics['accuracy']:.3%}")
        if (classifiers/'training_state.pt').exists():
            print(f'Resume checkpoint: {classifiers/"training_state.pt"}')
    else:
        print('No v2 classifier progress yet; old trainer progress cannot be recovered from the load-only log.')
    for arm in ('baseline','mixed'):
        path=root/'evaluation'/arm/'progress.json'
        if path.exists():
            row=json.loads(path.read_text())
            print(f"Evaluation {arm}: {row['completed']}/{row['total']}, remaining {duration(row.get('eta_s'))}")
    print('Classifier ETA excludes subsequent inference evaluation.')


def summarize(root):
    def read(folder):
        with (folder/'summary_rank000.csv').open() as f:
            summary={r['mode']:r for r in csv.DictReader(f)}
        by_mode={}
        with (folder/'results_rank000.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                by_mode.setdefault(r['mode'],set()).add(r['prompt_key'])
        return summary,by_mode
    baseline,bkeys=read(root/'evaluation/baseline')
    mixed,mkeys=read(root/'evaluation/mixed')
    base=baseline['fixed_16']
    selected=[('fixed_train_fixed16',base,bkeys['fixed_16']),
              ('mixed_train_fixed16',mixed['fixed_16'],mkeys['fixed_16'])]
    selected += [('mixed_train_'+mode,r,mkeys[mode]) for mode,r in mixed.items() if mode.startswith('classifier_')]
    if len(selected)!=6:
        raise ValueError('Expected fixed baseline, mixed fixed16 and all four classifiers')
    speed=float(base['decode_tok_s'])
    output=[]
    for name,row,keys in selected:
        if keys!=bkeys['fixed_16']:
            raise ValueError('Evaluation prompt sets differ')
        output.append(dict(method=name,samples=len(keys),mean_accepted=float(row['mean_accepted']),
            mean_committed=float(row['mean_committed']),decode_tok_s=float(row['decode_tok_s']),
            speedup_vs_baseline=float(row['decode_tok_s'])/speed,
            selector_time_fraction=float(row.get('selector_time_fraction',0))))
    path=root/'final_comparison.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    for row in output:
        print(f"{row['method']:55s} {row['decode_tok_s']:9.2f} tok/s  {row['speedup_vs_baseline']:.3f}x")
    print(f'Results: {path}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['status','mark','summarize'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--classifiers',type=Path)
    p.add_argument('--stage',default='unknown')
    p.add_argument('--status',default='running')
    args=p.parse_args()
    if args.command=='status':
        status(args.root,args.classifiers or args.root/'classifiers_ddp')
    elif args.command=='mark':
        write_json(dict(stage=args.stage,status=args.status,updated_at=time.time()),args.root/'workflow.json')
    else:
        summarize(args.root)


if __name__=='__main__':
    main()
