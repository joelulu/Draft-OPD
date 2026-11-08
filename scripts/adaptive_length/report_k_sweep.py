"""CPU report: per-K throughput, normalized acceptance heatmaps and paired checks."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics

KS = [4,8,12,16,20,24,28,32]


def load_jsonl(path):
    return [json.loads(v) for v in path.read_text().splitlines() if v.strip()]


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(records):
    groups = defaultdict(list)
    trajectories = {}
    for r in records:
        groups[r['arm'],r['k']].append(r)
        key = r['arm'],r['k'],r['prompt_key']
        fingerprint = (r['output_sha256'],r['round_details'])
        if key in trajectories and trajectories[key] != fingerprint:
            raise ValueError('Repeated trajectory changed; do not mix its distribution/timing')
        trajectories[key] = fingerprint
    rows = []
    for k in KS:
        stats = {}
        for arm in ('fixed','mixed'):
            rs = groups[arm,k]
            if not rs:
                raise ValueError(f'Missing {arm} K{k}')
            tokens = sum(r['tokens'] for r in rs)
            seconds = sum(r['decode_s'] for r in rs)
            rounds = sum(r['spec_rounds'] for r in rs)
            stats[arm] = dict(tok_s=tokens/seconds,accepted=sum(r['accepted_tokens'] for r in rs)/max(rounds,1),
                rounds_per_1k=rounds*1000/tokens,mean_output_tokens=statistics.mean(r['tokens'] for r in rs))
        row = dict(k=k,fixed_tok_s=stats['fixed']['tok_s'],mixed_tok_s=stats['mixed']['tok_s'],
            mixed_gain_pct=100*(stats['mixed']['tok_s']/stats['fixed']['tok_s']-1),
            fixed_accepted=stats['fixed']['accepted'],mixed_accepted=stats['mixed']['accepted'],
            fixed_rounds_per_1k=stats['fixed']['rounds_per_1k'],mixed_rounds_per_1k=stats['mixed']['rounds_per_1k'],
            fixed_mean_output_tokens=stats['fixed']['mean_output_tokens'],mixed_mean_output_tokens=stats['mixed']['mean_output_tokens'])
        # Paired per-prompt speed comparisons, retaining all repetitions in totals.
        speeds = {}
        for arm in ('fixed','mixed'):
            by_prompt = defaultdict(list)
            for r in groups[arm,k]: by_prompt[r['prompt_key']].append(r)
            speeds[arm] = {key:sum(v['tokens'] for v in rs)/sum(v['decode_s'] for v in rs) for key,rs in by_prompt.items()}
        if speeds['fixed'].keys() != speeds['mixed'].keys():
            raise ValueError('Fixed/mixed prompts do not match')
        row['mixed_faster_prompts'] = sum(speeds['mixed'][key]>v for key,v in speeds['fixed'].items())
        row['prompts'] = len(speeds['fixed'])
        rows.append(row)
    return rows


def acceptance_distribution(records):
    counts = defaultdict(Counter)
    skipped = Counter()
    for r in records:
        if r['repeat'] != 0: continue
        for rd in r['round_details']:
            if not rd['actual_k'] or rd['actual_k'] != r['k']:
                skipped['ar_or_tail_clipped'] += 1
                continue
            if rd['accepted'] != rd['accepted_raw']:
                skipped['eos_adjusted'] += 1
                continue
            counts[r['arm'],r['k']][rd['accepted']] += 1
    rows = []
    for arm in ('fixed','mixed'):
        for k in KS:
            total = sum(counts[arm,k].values())
            for a in range(33):
                rows.append(dict(arm=arm,k=k,accepted=a,count=counts[arm,k][a],
                    total_rounds=total,probability=counts[arm,k][a]/total if total else None))
    return rows,dict(skipped)


def paired_stats(records):
    clean = []
    skipped = 0
    for r in records:
        # Shared population for both models, exclude if either encounters EOS.
        if any(r[arm]['raw_accepted_lengths'] != r[arm]['accepted_lengths'] for arm in ('fixed','mixed')):
            skipped += 1
        else:
            clean.append(r)
    means = []
    for i,k in enumerate(KS):
        fixed = [r['fixed']['accepted_lengths'][i] for r in clean]
        mixed = [r['mixed']['accepted_lengths'][i] for r in clean]
        means.append(dict(k=k,states=len(clean),fixed_mean=statistics.mean(fixed) if fixed else None,
            mixed_mean=statistics.mean(mixed) if mixed else None,
            mixed_better=sum(m>f for f,m in zip(fixed,mixed)),equal=sum(m==f for f,m in zip(fixed,mixed)),
            mixed_worse=sum(m<f for f,m in zip(fixed,mixed))))
    changes = []
    for arm in ('fixed','mixed'):
        for i,small in enumerate(KS):
            for j in range(i+1,len(KS)):
                pairs = [(r[arm]['accepted_lengths'][i],r[arm]['accepted_lengths'][j]) for r in clean]
                controlled = [(a,b) for a,b in pairs if 0<b<=small]
                changes.append(dict(arm=arm,small_k=small,large_k=KS[j],states=len(pairs),
                    larger_better=sum(b>a for a,b in pairs),equal=sum(b==a for a,b in pairs),
                    larger_worse=sum(b<a for a,b in pairs),
                    capacity_controlled_pairs=len(controlled),
                    larger_helps_despite_enough_capacity=sum(b>a for a,b in controlled)))
    return means,changes,dict(total=len(records),used=len(clean),eos_excluded=skipped)


def plot_distribution(rows, name, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    arrays, totals, means = {},{},{}
    for arm in ('fixed','mixed'):
        array = np.full((33,8),np.nan)
        totals[arm],means[arm] = [],[]
        for i,k in enumerate(KS):
            col = [r for r in rows if r['arm']==arm and r['k']==k]
            n = col[0]['total_rounds']
            totals[arm].append(n)
            means[arm].append(sum(r['accepted']*r['count'] for r in col)/n if n else np.nan)
            if n:
                for r in col:
                    if r['accepted']<=k: array[r['accepted'],i] = 100*r['probability']
        arrays[arm]=array
    values = [v for arr in arrays.values() for v in arr.ravel() if np.isfinite(v)]
    vmax = max(values) if values else 100
    fig,axes = plt.subplots(1,2,figsize=(12,7.8),sharey=True,layout='constrained')
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('#ededed')
    for ax,arm in zip(axes,('fixed','mixed')):
        image=ax.imshow(np.ma.masked_invalid(arrays[arm]),origin='lower',aspect='auto',
                        cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest')
        ax.set_xticks(range(8),[f'{k}\nn={n}' for k,n in zip(KS,totals[arm])])
        ax.set_yticks(range(0,33,2))
        ax.set_xlabel('Fixed candidate count K (excluding anchor)')
        ax.set_title(f'{arm}-trained draft')
        ax.plot(range(8),means[arm],color='#154c79',marker='o',linewidth=1.5,label='Mean accepted')
        ax.legend(loc='upper left',fontsize=8)
    axes[0].set_ylabel('Accepted candidate count (excluding anchor)')
    fig.colorbar(image,ax=axes,label='Rounds within this K column (%)')
    fig.suptitle(f'{name}: acceptance distribution\nFirst repeat only; tail-clipped and EOS-adjusted rounds excluded')
    fig.savefig(path,dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    args=p.parse_args()
    root=Path(args.root)
    all_rows=[]
    for name in ('mixed_holdout','gsm8k'):
        folder=root/name
        if not (folder/'complete.json').exists(): raise ValueError(f'Incomplete: {folder}')
        records=load_jsonl(folder/'results.jsonl')
        rows=summarize(records)
        for r in rows: r['dataset']=name
        write_csv(folder/'throughput.csv',rows)
        all_rows.extend(rows)
        distribution,skipped=acceptance_distribution(records)
        write_csv(folder/'acceptance_distribution.csv',distribution)
        plot_distribution(distribution,name,root/f'{name}_acceptance_heatmap.png')
        print(f'\n===== {name} =====\nK     fixed tok/s   mixed tok/s   mixed gain    mixed faster prompts')
        for r in rows:
            print(f"{r['k']:2d} {r['fixed_tok_s']:13.2f} {r['mixed_tok_s']:13.2f} "
                  f"{r['mixed_gain_pct']:+11.2f}%    {r['mixed_faster_prompts']}/{r['prompts']}")
        print('Mixed faster aggregate K settings:',sum(r['mixed_gain_pct']>0 for r in rows),'/ 8')
        print('Heatmap excluded rounds:',skipped)
        paired_path=folder/'paired_states.jsonl'
        if paired_path.exists():
            means,changes,scope=paired_stats(load_jsonl(paired_path))
            write_csv(folder/'same_state_acceptance.csv',means)
            write_csv(folder/'same_state_k_changes.csv',changes)
            (folder/'same_state_scope.json').write_text(json.dumps(scope,indent=2))
            print('Common AR state acceptance: K fixed_mean mixed_mean mixed_better/equal/worse')
            for r in means:
                f,m=r['fixed_mean'],r['mixed_mean']
                print(f"{r['k']:2d} {f if f is not None else 'N/A'} {m if m is not None else 'N/A'} "
                      f"{r['mixed_better']}/{r['equal']}/{r['mixed_worse']}")
            for arm in ('fixed','mixed'):
                selected=[r for r in changes if r['arm']==arm]
                print(f"{arm}: larger-K hurts in {sum(r['larger_worse'] for r in selected)}/"
                      f"{sum(r['states'] for r in selected)} paired K comparisons")
    write_csv(root/'throughput_all.csv',all_rows)
    print('\nResults:',root)


if __name__=='__main__':
    main()
