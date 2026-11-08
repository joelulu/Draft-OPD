"""Post-hoc changes inside each state's observed best accepted prefix.

No stopping policy or thresholds. Reads existing saved top5 and full-distribution
adjacent-K JS. Missing positions are missing observations, never zero changes.
Path sums of JS are explicitly not JS between initial and final distributions.
"""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics as st

KS=[4,8,12,16,20,24,28,32]


def mean(values):
    values=[v for v in values if v is not None]
    return st.mean(values) if values else None


def rate(values):
    return mean([int(v) for v in values if v is not None])


def fmt(value,digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def csv_write(path,rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_position(row,small,large,position):
    pair=row['pairs'][f'{small}:{large}']
    item=pair['positions'][position-1]
    if item['position']!=position:
        raise ValueError('Unaligned position record')
    return item


def position_history(row,position,current_k):
    available=[k for k in KS if position<=k<=current_k]
    if not available:
        return None
    points=[row['trials'][str(k)]['positions'][position-1] for k in available]
    edges=[pair_position(row,a,b,position) for a,b in zip(available,available[1:])]
    ids=[p['top5_ids'][0] for p in points]
    flips=sum(not p['top1_equal'] for p in edges)
    current=points[-1]
    history=points[:-1]
    current_set=set(current['top5_ids'])
    return dict(position=position,current_k=current_k,first_observed_k=available[0],
        observed_k_count=len(available),transition_count=len(edges),
        observed_ks=json.dumps(available),top1_ids=json.dumps(ids),
        first_top1_id=ids[0],current_top1_id=ids[-1],
        top1_flip_count=flips,ever_top1_changed=flips>0 if edges else None,
        net_top1_changed=ids[0]!=ids[-1] if edges else None,
        changed_then_returned=flips>0 and ids[0]==ids[-1] if edges else None,
        unique_top1_count=len(set(ids)),
        modal_top1_agreement=max(Counter(ids).values())/len(ids) if edges else None,
        current_vs_history_top1_change=rate(p['top5_ids'][0]!=ids[-1] for p in history),
        current_vs_history_top5_overlap=mean(len(current_set & set(p['top5_ids']))/len(current_set) for p in history),
        current_vs_first_top5_overlap=len(current_set & set(points[0]['top5_ids']))/len(current_set) if edges else None,
        current_vs_history_peak_delta=mean(abs(current['top5_probs'][0]-p['top5_probs'][0]) for p in history),
        current_vs_history_margin_delta=mean(abs(current['top1_top2_margin']-p['top1_top2_margin']) for p in history),
        top5_set_change_count=sum(p['top5_overlap']<1-1e-9 for p in edges),
        top5_order_change_count=sum(not p['top5_order_equal'] for p in edges),
        path_js_sum=sum(p['js'] for p in edges) if edges else None,
        path_js_mean=mean(p['js'] for p in edges),
        path_js_max=max((p['js'] for p in edges),default=None),
        path_top5_overlap_mean=mean(p['top5_overlap'] for p in edges),
        path_peak_delta_sum=sum(p['peak_delta'] for p in edges) if edges else None,
        path_margin_delta_sum=sum(p['margin_delta'] for p in edges) if edges else None,
        reference_prob_range=(max(p['reference_prob'] for p in points)-min(p['reference_prob'] for p in points))
            if all(p['reference_prob'] is not None for p in points) else None,
        current_reference_hit=points[-1]['reference_hit'],
        current_reference_prob=points[-1]['reference_prob'],
        current_reference_margin=points[-1]['reference_margin'],
        current_top1_prob=points[-1]['top5_probs'][0],
        current_top1_top2_margin=points[-1]['top1_top2_margin'])


def region_summary(row,current_k,end,histories):
    visible=[h for h in histories if h['position']<=end]
    tracked=[h for h in visible if h['transition_count']>0]
    previous=KS[KS.index(current_k)-1] if current_k>4 else None
    last=[pair_position(row,previous,current_k,p) for p in range(1,min(end,previous)+1)] if previous else []
    total=sum(h['transition_count'] for h in tracked)
    js_sum=sum(h['path_js_sum'] for h in tracked)
    flips=sum(h['top1_flip_count'] for h in tracked)
    return dict(requested_region_end=end,visible_positions=len(visible),
        positions_with_history=len(tracked),position_transition_observations=total,
        # Both weighted definitions are kept: early positions have more edges.
        path_js_edge_weighted=js_sum/total if total else None,
        path_js_position_weighted=mean(h['path_js_mean'] for h in tracked),
        mean_position_js_path_sum=mean(h['path_js_sum'] for h in tracked),
        path_top1_flip_rate=flips/total if total else None,
        path_top5_overlap=(sum(h['path_top5_overlap_mean']*h['transition_count'] for h in tracked)/total) if total else None,
        path_top5_set_change_rate=sum(h['top5_set_change_count'] for h in tracked)/total if total else None,
        current_vs_history_top1_change=(sum(h['current_vs_history_top1_change']*h['transition_count'] for h in tracked)/total) if total else None,
        current_vs_history_top5_overlap=(sum(h['current_vs_history_top5_overlap']*h['transition_count'] for h in tracked)/total) if total else None,
        mean_position_peak_path_sum=mean(h['path_peak_delta_sum'] for h in tracked),
        mean_position_margin_path_sum=mean(h['path_margin_delta_sum'] for h in tracked),
        mean_reference_prob_range=mean(h['reference_prob_range'] for h in tracked),
        ever_changed_position_rate=rate(h['ever_top1_changed'] for h in tracked),
        net_changed_position_rate=rate(h['net_top1_changed'] for h in tracked),
        changed_then_returned_rate=rate(h['changed_then_returned'] for h in tracked),
        last_edge_positions=len(last),last_edge_js=mean(p['js'] for p in last),
        last_edge_top1_change=rate(not p['top1_equal'] for p in last),
        last_edge_top5_overlap=mean(p['top5_overlap'] for p in last),
        last_edge_peak_delta=mean(p['peak_delta'] for p in last),
        reference_hit_rate=rate(h['current_reference_hit'] for h in visible),
        reference_prob_mean=mean(h['current_reference_prob'] for h in visible),
        top1_prob_mean=mean(h['current_top1_prob'] for h in visible))


def analyze_state(row):
    accepted={k:row['trials'][str(k)]['accepted'] for k in KS}
    maximum=max(accepted.values())
    first_best=next(k for k in KS if accepted[k]==maximum)
    identity={k:row[k] for k in ('arm','prompt_key','prompt_index','split','generated_position')}
    identity.update(observed_max_accept=maximum,smallest_best_k=first_best,
        eos_adjusted=any(t['eos_adjusted'] for t in row['trials'].values()))
    positions,regions,edges=[] ,[],[]
    for current in KS:
        histories=[position_history(row,p,current) for p in range(1,current+1)]
        for h in histories:
            positions.append(dict(identity,accepted_at_k=accepted[current],inside_best_accepted=h['position']<=maximum,
                                  at_best_first_reject=h['position']==maximum+1,**h))
        previous=KS[KS.index(current)-1] if current>4 else 0
        endpoints={'best_accepted':maximum,'previous_prefix':previous,
            'best_boundary':maximum+1}
        endpoints.update({f'prefix_{end}':end for end in (4,8,12,16,20,24,28,32)})
        for region,end in endpoints.items():
            if region=='best_boundary':
                # A single boundary position, not a cumulative prefix.
                boundary=[h for h in histories if h['position']==end]
                summary=region_summary(row,current,end,histories)
                pos=boundary[0] if boundary else None
                summary={k:None for k in summary}
                summary.update(requested_region_end=end,visible_positions=len(boundary),
                    positions_with_history=int(pos is not None and pos['transition_count']>0),
                    position_transition_observations=pos['transition_count'] if pos else 0,
                    path_js_edge_weighted=pos['path_js_mean'] if pos else None,
                    path_js_position_weighted=pos['path_js_mean'] if pos else None,
                    mean_position_js_path_sum=pos['path_js_sum'] if pos else None,
                    reference_prob_mean=pos['current_reference_prob'] if pos else None,
                    reference_hit_rate=int(pos['current_reference_hit']) if pos and pos['current_reference_hit'] is not None else None,
                    top1_prob_mean=pos['current_top1_prob'] if pos else None)
            else:
                summary=region_summary(row,current,end,histories)
            regions.append(dict(identity,current_k=current,accepted_at_k=accepted[current],region=region,**summary))
    for i,large in enumerate(KS[1:],1):
        small=KS[i-1]
        for p in row['pairs'][f'{small}:{large}']['positions']:
            edges.append(dict(identity,small_k=small,large_k=large,
                small_accept=accepted[small],large_accept=accepted[large],
                inside_best_accepted=p['position']<=maximum,
                best_first_reject=p['position']==maximum+1,
                became_reference_correct=(row['trials'][str(small)]['positions'][p['position']-1]['reference_hit'] is False
                    and row['trials'][str(large)]['positions'][p['position']-1]['reference_hit'] is True),
                lost_reference_correct=(row['trials'][str(small)]['positions'][p['position']-1]['reference_hit'] is True
                    and row['trials'][str(large)]['positions'][p['position']-1]['reference_hit'] is False),**p))
    return positions,regions,edges


def markdown_example(row,regions,legend):
    maximum=max(t['accepted'] for t in row['trials'].values())
    lines=[f"# {row['arm']} prompt={row['prompt_index']} state={row['generated_position']}",
        f"Observed max accepted={maximum}; this is post-hoc, not a deployable stopping signal.",
        f"EOS-adjusted trials: {any(t['eos_adjusted'] for t in row['trials'].values())}",
        '', '| K | accepted | visible in 1..A* | history positions | cumulative mean JS | last-edge JS | history top1 flip rate | history top5 overlap | reference hit rate |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in regions:
        if r['region']=='best_accepted':
            lines.append(f"| {r['current_k']} | {r['accepted_at_k']} | {r['visible_positions']} | {r['positions_with_history']} | "
                f"{fmt(r['path_js_edge_weighted'])} | {fmt(r['last_edge_js'])} | {fmt(r['path_top1_flip_rate'])} | "
                f"{fmt(r['path_top5_overlap'])} | {fmt(r['reference_hit_rate'])} |")
    lines.extend(['','Blank cells below mean the position did not exist at that K.',
        '', '| position | '+' | '.join(f'K{k}' for k in KS)+' |',
        '|---:|'+'---|'*len(KS)])
    for position in range(1,min(maximum+1,32)+1):
        cells=[]
        for k in KS:
            if position>k:
                cells.append('—')
                continue
            p=row['trials'][str(k)]['positions'][position-1]
            token=p['top5_ids'][0]
            text=legend.get(str(token),str(token)).replace('|','\\|').replace('\n','\\n').replace('\r','\\r')
            hit='✓' if p['reference_hit'] is True else ('×' if p['reference_hit'] is False else '?')
            cells.append(f'{hit} {token}:{text} ({p["top5_probs"][0]:.3f})')
        lines.append(f'| {position} | '+' | '.join(cells)+' |')
    lines.extend(['','✓/× compares top1 with common AR reference, not a claim about post-rejection verification condition.',
        'Path JS is a sum/mean of adjacent transitions; it is not endpoint JS.'])
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='Existing logit_stability_TIMESTAMP root or one dataset directory')
    p.add_argument('--output')
    p.add_argument('--dataset',help='Optional dataset subdirectory to process')
    p.add_argument('--prompt-index',type=int,default=13,help='Detailed example prompt index')
    p.add_argument('--example-states',type=int,nargs='+',default=[0,192])
    args=p.parse_args()
    root=Path(args.root)
    folders=[root] if (root/'states.jsonl').exists() else sorted(p.parent for p in root.glob('*/states.jsonl'))
    if args.dataset: folders=[p for p in folders if p.name==args.dataset]
    if not folders: raise FileNotFoundError('No matching states.jsonl files')
    out=Path(args.output) if args.output else root/('accepted_region_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    all_positions,all_regions,all_edges=[],[],[]
    for folder in folders:
        legend=json.loads((folder/'token_legend.json').read_text())
        seen=set()
        for line in (folder/'states.jsonl').open():
            if not line.strip(): continue
            row=json.loads(line)
            key=row['arm'],row['prompt_key'],row['generated_position']
            if key in seen: raise ValueError('Duplicate state/model')
            seen.add(key)
            positions,regions,edges=analyze_state(row)
            for group in (positions,regions,edges):
                for r in group: r['dataset']=folder.name
            all_positions.extend(positions)
            all_regions.extend(regions)
            all_edges.extend(edges)
            if row['prompt_index']==args.prompt_index and row['generated_position'] in args.example_states:
                name=f'{folder.name}_{row["arm"]}_prompt{args.prompt_index}_state{row["generated_position"]}.md'
                (out/name).write_text(markdown_example(row,regions,legend))
                maximum=max(t['accepted'] for t in row['trials'].values())
                print(f'\n{folder.name} {row["arm"]} prompt={args.prompt_index} state={row["generated_position"]} A*={maximum}')
                print('K accepted visible history_n path_JS last_JS top1_flip_rate top5_overlap')
                for r in regions:
                    if r['region']=='best_accepted':
                        print(f"{r['current_k']:2d} {r['accepted_at_k']:8d} {r['visible_positions']:7d} "
                              f"{r['positions_with_history']:9d} {fmt(r['path_js_edge_weighted']):>8} "
                              f"{fmt(r['last_edge_js']):>8} {fmt(r['path_top1_flip_rate']):>10} {fmt(r['path_top5_overlap']):>10}")
    csv_write(out/'position_histories.csv',all_positions)
    csv_write(out/'cumulative_regions.csv',all_regions)
    csv_write(out/'adjacent_position_changes.csv',all_edges)
    groups=defaultdict(list)
    for r in all_regions:
        if r['eos_adjusted']: continue
        a=r['observed_max_accept']
        bucket='0' if a==0 else ('1-4' if a<=4 else '5-8' if a<=8 else '9-12' if a<=12 else '13-16' if a<=16 else '17-32')
        if r['region'] in ('best_accepted','previous_prefix','best_boundary'):
            groups[r['dataset'],r['arm'],bucket,r['current_k'],r['region']].append(r)
    summaries=[]
    for (dataset,arm,bucket,k,region),rs in sorted(groups.items()):
        summaries.append(dict(dataset=dataset,arm=arm,accept_bucket=bucket,current_k=k,region=region,
            states=len(rs),mean_observed_max_accept=mean(r['observed_max_accept'] for r in rs),
            mean_visible_positions=mean(r['visible_positions'] for r in rs),
            mean_history_positions=mean(r['positions_with_history'] for r in rs),
            state_mean_path_js=mean(r['path_js_edge_weighted'] for r in rs),
            state_mean_last_edge_js=mean(r['last_edge_js'] for r in rs),
            state_mean_path_top1_flip=mean(r['path_top1_flip_rate'] for r in rs),
            state_mean_path_top5_overlap=mean(r['path_top5_overlap'] for r in rs),
            state_mean_current_vs_history_top1_change=mean(r['current_vs_history_top1_change'] for r in rs),
            state_mean_current_vs_history_top5_overlap=mean(r['current_vs_history_top5_overlap'] for r in rs),
            state_mean_reference_hit=mean(r['reference_hit_rate'] for r in rs)))
    csv_write(out/'group_summary.csv',summaries)
    json_meta=dict(source=[str(p) for p in folders],positions=len(all_positions),regions=len(all_regions),
        eos_adjusted_regions_excluded_from_group_summary=sum(r['eos_adjusted'] for r in all_regions),
        scope='Pure post-hoc descriptive analysis, using observed maximum acceptance across tested K. '
              'Different K have different position coverage. No missing position is treated as stable. '
              'Path JS is not endpoint JS; full nonadjacent JS cannot be reconstructed from saved top5. '
              'Group means are state-weighted, with within-prompt dependence; no significance claim. '
              'No stopping policy or threshold selection.')
    (out/'scope.json').write_text(json.dumps(json_meta,indent=2))
    print('\nResults:',out)


if __name__=='__main__':
    main()
