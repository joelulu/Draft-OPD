"""Same-state cross-K logits, calibration-only stopping rules, and online replay.

K excludes anchor. K32 is a reference, not truth. Teacher verification supplies
actual acceptance; common target-AR references supply comparable position hits.
No full-vocabulary logits are written to disk. No predictor is trained.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics as st
import time

from oracle_upper_bound import KS, emitted_count, json_write, signature, replay


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rows_from(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def load_inputs(args, torch, tokenizer, read_prompts):
    selected, seen = [], set()
    for key, prompt in read_prompts(args.data):
        if key in seen:
            continue
        seen.add(key)
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
        if ids.shape[1] > args.max_prompt:
            raise ValueError('Prompt too long; increase max-prompt, do not silently substitute')
        selected.append((key, prompt, ids.to(args.device)))
        if len(selected) == args.samples:
            break
    if len(selected) != args.samples:
        raise ValueError('Not enough unique prompts')
    return selected


def setup(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model.dflash import DFlashDraftModel
    from adaptive_length.pipeline import read_prompts
    from adaptive_length.runtime import RoundRuntime, clock
    from adaptive_length.two_stage.cli import model_identity
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for collection and online evaluation')
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda', 0)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    inputs = load_inputs(args, torch, tokenizer, read_prompts)
    fingerprints = {name:model_identity(path) for name,path in
        [('target',args.target),('fixed',args.fixed_draft),('mixed',args.mixed_draft)]}
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to(args.device).eval().requires_grad_(False)
    drafts = {arm:DFlashDraftModel.from_pretrained(path, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to(args.device).eval().requires_grad_(False)
        for arm,path in [('fixed',args.fixed_draft),('mixed',args.mixed_draft)]}
    return torch, tokenizer, inputs, fingerprints, target, drafts, RoundRuntime, clock


def greedy_top5(values,torch):
    # BF16 logits can tie. Repeated argmax breaks ties by token ID just like
    # the actual greedy proposal; torch.topk's tie ordering is unspecified.
    work=values.clone()
    indices=[]
    for _ in range(min(5,values.shape[-1])):
        index=work.argmax(-1,keepdim=True)
        indices.append(index)
        work.scatter_(-1,index,float('-inf'))
    return torch.cat(indices,-1)


def logit_features(logits, torch):
    values = logits.detach().float().squeeze(0)
    logp = values.log_softmax(-1)
    top_ids = greedy_top5(values,torch)
    top_values = values.gather(-1,top_ids)
    top_probs = logp.gather(-1, top_ids).exp()
    entropy = -(logp.exp()*logp).sum(-1)
    margins = top_values[:,0]-top_values[:,1]
    return dict(logp=logp, ids=top_ids, logits=top_values, probs=top_probs,
                entropy=entropy, margins=margins)


def compare_features(small, large, torch):
    n = small['logp'].shape[0]
    lp, lq = small['logp'], large['logp'][:n]
    p, q = lp.exp(), lq.exp()
    lm = ((p+q)*.5).clamp_min(torch.finfo(p.dtype).tiny).log()
    js = (.5*((p*(lp-lm)).sum(-1)+(q*(lq-lm)).sum(-1))).clamp_min(0)
    ids, other = small['ids'], large['ids'][:n]
    overlap = (ids[:,:,None] == other[:,None,:]).any(-1).float().mean(-1)
    same1 = ids[:,0] == other[:,0]
    same5 = (ids == other).all(-1)
    peak = (small['probs'][:,0]-large['probs'][:n,0]).abs()
    margin = (small['margins']-large['margins'][:n]).abs()
    positions = [dict(position=i+1, js=float(js[i]), top1_equal=bool(same1[i]),
        top5_overlap=float(overlap[i]), top5_order_equal=bool(same5[i]),
        peak_delta=float(peak[i]), margin_delta=float(margin[i])) for i in range(n)]
    return dict(mean_js=float(js.mean()), max_js=float(js.max()),
        last4_js=float(js[-4:].mean()), top1_change=float((~same1).float().mean()),
        top5_distance=float(1-overlap.mean()), peak_delta=float(peak.mean()),
        positions=positions)


def collect_state(runtime, reference, repeats, rng, eos, clock, torch):
    from adaptive_length.two_stage.dflash_features import materialize_context
    start = runtime.prefix.shape[1]
    # Project confirmed context once, shared by all K and explicitly costed.
    t0 = clock()
    cached = runtime.draft_cache.get_seq_length()
    materialize_context(runtime.draft, runtime.draft_cache,
        runtime.context[:,cached-runtime.context_start:], cached)
    prep_ms = (clock()-t0)*1000
    for k in KS:
        block = runtime.propose(k)
        runtime.verify(block)
    features, trials = {}, {}
    for repeat in range(repeats):
        order = KS.copy()
        rng.shuffle(order)
        for k in order:
            t0 = clock()
            block, logits = runtime.propose(k, return_logits=True)
            t1 = clock()
            result = runtime.verify(block)
            t2 = clock()
            ids = block[0].tolist()
            a = result[0]
            if k in trials:
                if trials[k]['block_ids'] != ids or trials[k]['accepted_raw'] != a:
                    raise RuntimeError('Repeated proposal/acceptance differs at fixed state')
                lp=features[k]['logp']
                lq=logits.detach().float().squeeze(0).log_softmax(-1)
                p,q=lp.exp(),lq.exp()
                lm=((p+q)*.5).clamp_min(torch.finfo(p.dtype).tiny).log()
                js=(.5*((p*(lp-lm)).sum(-1)+(q*(lq-lm)).sum(-1))).clamp_min(0)
                trials[k]['repeat_js_mean'].append(float(js.mean()))
                trials[k]['repeat_js_max'].append(float(js.max()))
            else:
                f = logit_features(logits, torch)
                features[k] = f
                pos = []
                posterior = result[1][0,:k].tolist()
                for i in range(k):
                    ref = reference[i] if i < len(reference) else None
                    ref_prob = float(f['logp'][i,ref].exp()) if ref is not None else None
                    ref_margin = (float(logits[0,i,ref])-float(f['logits'][i,1 if int(f['ids'][i,0])==ref else 0])) if ref is not None else None
                    pos.append(dict(position=i+1, top5_ids=f['ids'][i].tolist(),
                        top5_logits=f['logits'][i].tolist(), top5_probs=f['probs'][i].tolist(),
                        entropy=float(f['entropy'][i]), top1_top2_margin=float(f['margins'][i]),
                        reference_id=ref, reference_prob=ref_prob, reference_margin=ref_margin,
                        reference_hit=ids[i+1]==ref if ref is not None else None,
                        verify_target_id=posterior[i], verify_hit=ids[i+1]==posterior[i]))
                emitted = emitted_count(ids,a,k+1,eos)
                trials[k] = dict(k=k, accepted_raw=a, accepted=emitted-1,
                    eos_adjusted=emitted-1!=a, block_ids=ids, positions=pos,
                    draft_ms=[], verify_ms=[],repeat_js_mean=[],repeat_js_max=[])
                reference_accept=0
                for p in pos:
                    if p['reference_hit'] is not True: break
                    reference_accept+=1
                trials[k].update(reference_prefix_accept=reference_accept,
                    reference_available=min(k,len(reference)))
            trials[k]['draft_ms'].append((t1-t0)*1000)
            trials[k]['verify_ms'].append((t2-t1)*1000)
            if runtime.target_cache.get_seq_length()!=start or runtime.draft_cache.get_seq_length()!=start:
                raise RuntimeError('Trial cache boundary polluted')
    pairs = {}
    comparisons = {(KS[i],KS[i+1]) for i in range(7)} | {(k,32) for k in KS[:-1]}
    for small, large in sorted(comparisons):
        pair = compare_features(features[small],features[large],torch)
        pair.update(small_k=small, large_k=large,
            adjacent=KS.index(large)==KS.index(small)+1,
            accept_gain=trials[large]['accepted']-trials[small]['accepted'])
        for pos in pair['positions']:
            i = pos['position']
            pos.update(small_region='accepted' if i<=trials[small]['accepted'] else
                ('first_reject' if i==trials[small]['accepted']+1 else 'after_reject'),
                reference_region32='accepted' if i<=trials[32]['accepted'] else
                ('first_reject' if i==trials[32]['accepted']+1 else 'after_reject'))
        pairs[f'{small}:{large}'] = pair
    return dict(context_prep_ms=prep_ms, trials={str(k):v for k,v in trials.items()}, pairs=pairs)


def reference_sequence(target, draft, ids, limit, eos, Runtime):
    walker = Runtime(target,draft,ids)
    n = ids.shape[1]
    for _ in range(limit):
        walker.ar_step()
        if eos is not None and int(walker.prefix[0,-1])==eos:
            break
    return walker.prefix[0,n:].tolist()


def collect(args):
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists():
        raise ValueError('Use a fresh output directory')
    torch,tok,inputs,fingerprints,target,drafts,Runtime,clock = setup(args)
    split_at = len(inputs)//2
    if split_at<1 or split_at==len(inputs): raise ValueError('Need >=2 prompts for separate calibration/test')
    prompts = [dict(prompt_key=key,prompt=prompt,index=i+1,
        split='cal' if i<split_at else 'test') for i,(key,prompt,_) in enumerate(inputs)]
    json_write(out/'metadata.json',dict(dataset=args.dataset,k_values=KS,seed=args.seed,
        max_new_tokens=args.max_new_tokens,max_prompt=args.max_prompt,samples=args.samples,
        states=args.states,stride=args.stride,repeats=args.repeats,
        fingerprints=fingerprints,prompts=prompts,
        reference='common target-AR trajectory; K32 is only a comparison reference',
        timing='draft/verify medians exclude feature analysis; same-state rule speed proxies omit decision overhead'))
    legend = {}
    with torch.inference_mode():
        for i,(key,prompt,ids) in enumerate(inputs):
            limit = min(args.max_new_tokens,(args.states-1)*args.stride+33)
            reference = reference_sequence(target,drafts['fixed'],ids,limit,tok.eos_token_id,Runtime)
            for position in range(0,args.states*args.stride,args.stride):
                if position+33>args.max_new_tokens or position>=len(reference): break
                if tok.eos_token_id is not None and reference[position]==tok.eos_token_id: break
                prefix = torch.cat((ids,ids.new_tensor([reference[:position]])),dim=1)
                order = ['fixed','mixed']
                random.Random(args.seed+i*100+position).shuffle(order)
                for arm in order:
                    state = Runtime(target,drafts[arm],prefix)
                    state.anchor = ids.new_tensor([[reference[position]]])
                    row = collect_state(state,reference[position+1:position+33],args.repeats,
                        random.Random(args.seed+i*100+position),tok.eos_token_id,clock,torch)
                    row.update(arm=arm,prompt_key=key,prompt_index=i+1,split=prompts[i]['split'],
                        generated_position=position,context_length=prefix.shape[1],
                        prefix_sha256=signature(prefix[0].tolist()),anchor_id=reference[position])
                    for trial in row['trials'].values():
                        for p in trial['positions']:
                            for token in p['top5_ids']:
                                if str(token) not in legend: legend[str(token)]=tok.decode([token])
                    with (out/'states.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                    json_write(out/'token_legend.json',legend)
                    print(f"[{args.dataset}] prompt={i+1}/{len(inputs)} state={position} "
                          f"{arm} {row['split']} acceptance="+
                          ' '.join(f"{row['trials'][str(k)]['accepted']}({k})" for k in KS),flush=True)
                    del state,row
    json_write(out/'collect_complete.json',dict(prompts=len(inputs)))


def candidate_rules():
    thresholds = dict(top1_change=[0,.05,.1,.2],top5_distance=[0,.05,.1,.2],
                      mean_js=[1e-5,1e-4,1e-3,1e-2],peak_delta=[.005,.01,.02,.05])
    return [dict(metric=m,threshold=t,patience=p) for m,ts in thresholds.items() for t in ts for p in (1,2)]


def choose_from_state(row,rule):
    stable = 0
    for i,k in enumerate(KS[1:],1):
        change = row['pairs'][f'{KS[i-1]}:{k}'][rule['metric']]
        stable = stable+1 if change<=rule['threshold'] else 0
        if stable>=rule['patience']: return k,KS[:i+1]
    return 32,KS.copy()


def rule_summary(rows,rule):
    oracle_committed = selected_committed = total_ms = 0.
    calls = future_gain = early_stop = 0
    selections = Counter()
    for row in rows:
        k,tried = choose_from_state(row,rule)
        trials = row['trials']
        accepted = trials[str(k)]['accepted']
        best = max(t['accepted'] for t in trials.values())
        selected_committed += accepted+1
        oracle_committed += best+1
        total_ms += row['context_prep_ms']+sum(st.median(trials[str(t)]['draft_ms']) for t in tried)+st.median(trials[str(k)]['verify_ms'])
        calls += len(tried)
        remaining = max(trials[str(t)]['accepted'] for t in KS if t>k) if k<32 else accepted
        gain = max(0,remaining-accepted)
        future_gain += gain
        early_stop += gain>0
        selections[k] += 1
    n=len(rows)
    return dict(states=n,commit_retention=selected_committed/oracle_committed if n else None,
        proxy_tok_s=selected_committed/total_ms*1000 if total_ms else None,
        mean_draft_calls=calls/n if n else None,
        harmful_stop_rate=early_stop/n if n else None,
        mean_forgone_future_accept=future_gain/n if n else None,
        selected_k_counts=json.dumps(dict(selections)),
        proxy_note='same-state medians; excludes stopping metric/CPU overhead, not end-to-end throughput')


def analyze(args):
    out=Path(args.output)
    if not (out/'collect_complete.json').exists(): raise ValueError('Collection incomplete')
    metadata=json.loads((out/'metadata.json').read_text())
    rows=rows_from(out/'states.jsonl')
    legend=json.loads((out/'token_legend.json').read_text())
    # Check shared state identities and prompt-level split integrity.
    by_state=defaultdict(dict)
    prompt_splits={}
    for r in rows:
        if prompt_splits.setdefault(r['prompt_key'],r['split'])!=r['split']: raise ValueError('Prompt leakage')
        key=r['prompt_key'],r['generated_position']
        if r['arm'] in by_state[key]: raise ValueError('Duplicate model/state')
        by_state[key][r['arm']]=r
    for group in by_state.values():
        if set(group)!= {'fixed','mixed'}: raise ValueError('Unpaired models')
        if any(group['fixed'][k]!=group['mixed'][k] for k in ('prefix_sha256','anchor_id','split')):
            raise ValueError('Mismatched prefixes or anchors')
    position_rows,pair_rows,top1_rows,top5_rows=[],[],[],[]
    for row in rows:
        identity={k:row[k] for k in ('arm','prompt_key','split','generated_position')}
        for k in KS:
            trial=row['trials'][str(k)]
            grid=dict(identity,k=k,accepted=trial['accepted'])
            grid5=dict(identity,k=k,accepted=trial['accepted'])
            for i in range(32):
                p=trial['positions'][i] if i<k else None
                grid[f'p{i+1:02d}']=f"{p['top5_ids'][0]}:{legend[str(p['top5_ids'][0])]}" if p else ''
                grid5[f'p{i+1:02d}']=json.dumps([
                    dict(id=t,text=legend[str(t)],prob=prob)
                    for t,prob in zip(p['top5_ids'],p['top5_probs'])],ensure_ascii=False) if p else ''
            top1_rows.append(grid)
            top5_rows.append(grid5)
            for pos in trial['positions']:
                item=dict(identity,k=k,accepted=trial['accepted'],**pos)
                for field in ('top5_ids','top5_logits','top5_probs'): item[field]=json.dumps(item[field])
                item['top5_text']=json.dumps([legend[str(t)] for t in pos['top5_ids']],ensure_ascii=False)
                position_rows.append(item)
        for pair in row['pairs'].values():
            for pos in pair['positions']:
                pair_rows.append(dict(identity,small_k=pair['small_k'],large_k=pair['large_k'],
                    adjacent=pair['adjacent'],accept_gain=pair['accept_gain'],**pos))
    write_csv(out/'top5_markers.csv',position_rows)
    write_csv(out/'top1_grid_32.csv',top1_rows)
    write_csv(out/'top5_grid_32.csv',top5_rows)
    write_csv(out/'position_changes.csv',pair_rows)
    summary=[]
    groups=defaultdict(list)
    for r in pair_rows: groups[r['arm'],r['small_k'],r['large_k'],r['small_region']].append(r)
    for (arm,small,large,region),rs in groups.items():
        summary.append(dict(arm=arm,small_k=small,large_k=large,small_region=region,positions=len(rs),
            mean_js=st.mean(r['js'] for r in rs),top1_change=st.mean(not r['top1_equal'] for r in rs),
            top5_overlap=st.mean(r['top5_overlap'] for r in rs),peak_delta=st.mean(r['peak_delta'] for r in rs)))
    write_csv(out/'change_summary.csv',summary)
    k_curves=[]
    for arm in ('fixed','mixed'):
        arm_rows=[r for r in rows if r['arm']==arm]
        for k in KS:
            pairs=[r['pairs'][f'{k}:32'] for r in arm_rows] if k<32 else []
            item=dict(arm=arm,k=k,states=len(arm_rows),
                mean_accept=st.mean(r['trials'][str(k)]['accepted'] for r in arm_rows),
                top1_change_vs32=st.mean(p['top1_change'] for p in pairs) if pairs else 0.,
                top5_overlap_vs32=st.mean(1-p['top5_distance'] for p in pairs) if pairs else 1.,
                mean_js_vs32=st.mean(p['mean_js'] for p in pairs) if pairs else 0.,
                same_k_repeat_js_max=max((v for r in arm_rows for v in r['trials'][str(k)]['repeat_js_max']),default=0.))
            k_curves.append(item)
        print(f'\n{arm} pooled descriptive curve (K32 is reference, not ground truth):')
        print('K  accepted  top1_change_vs32  top5_overlap_vs32   mean_JS')
        for r in k_curves:
            if r['arm']==arm:
                print(f"{r['k']:2d} {r['mean_accept']:8.3f} {r['top1_change_vs32']:17.2%} "
                      f"{r['top5_overlap_vs32']:19.2%} {r['mean_js_vs32']:10.6f}")
    write_csv(out/'k_curves.csv',k_curves)
    clean=[r for r in rows if not any(t['eos_adjusted'] for t in r['trials'].values())]
    trials=[]
    choices={}
    for arm in ('fixed','mixed'):
        cal=[r for r in clean if r['arm']==arm and r['split']=='cal']
        test=[r for r in clean if r['arm']==arm and r['split']=='test']
        if not cal or not test: raise ValueError(f'No eligible cal/test states for {arm}')
        candidates=[]
        for rule in candidate_rules():
            stat=rule_summary(cal,rule)
            candidates.append(dict(rule=rule,metrics=stat))
            trials.append(dict(arm=arm,split='cal',**rule,**stat))
        eligible=[r for r in candidates if r['metrics']['commit_retention']>=.95]
        best=max(eligible,key=lambda r:r['metrics']['proxy_tok_s']) if eligible else max(candidates,
            key=lambda r:(r['metrics']['commit_retention'],r['metrics']['proxy_tok_s']))
        # Test statistics are computed only AFTER fixing the calibration choice.
        test_stat=rule_summary(test,best['rule'])
        choices[arm]=dict(rule=best['rule'],calibration=best['metrics'],test=test_stat,
            calibration_retention_target_met=bool(eligible))
        print(f"{arm}: rule={best['rule']} cal retention={best['metrics']['commit_retention']:.2%} "
              f"test retention={test_stat['commit_retention']:.2%} "
              f"test calls={test_stat['mean_draft_calls']:.2f} harmful stops={test_stat['harmful_stop_rate']:.2%}",flush=True)
    write_csv(out/'calibration_rules.csv',trials)
    json_write(out/'chosen_rules.json',dict(fingerprints=metadata['fingerprints'],choices=choices,
        cal_prompt_keys=sorted(k for k,v in prompt_splits.items() if v=='cal'),
        test_prompt_keys=sorted(k for k,v in prompt_splits.items() if v=='test'),
        scope='rules selected on calibration prompts only; online full decode is authoritative'))
    # Conditional analysis: does low change predict no future gain? Calibration
    # and test are separate, so findings can be checked without threshold tuning.
    curves=[]
    for arm in ('fixed','mixed'):
        for split in ('cal','test'):
            for metric,thresholds in dict(top1_change=[0,.05,.1,.2],top5_distance=[0,.05,.1,.2],
                mean_js=[1e-5,1e-4,1e-3,1e-2],peak_delta=[.005,.01,.02,.05]).items():
                for threshold in thresholds:
                    matches=[]
                    for row in clean:
                        if row['arm']!=arm or row['split']!=split: continue
                        for i,k in enumerate(KS[1:-1],1):
                            if row['pairs'][f'{KS[i-1]}:{k}'][metric]<=threshold:
                                a=row['trials'][str(k)]['accepted']
                                matches.append(max(0,max(row['trials'][str(t)]['accepted'] for t in KS[i+1:])-a))
                    curves.append(dict(arm=arm,split=split,metric=metric,threshold=threshold,
                        stable_comparisons=len(matches),later_still_improves_rate=st.mean(v>0 for v in matches) if matches else None,
                        mean_later_gain=st.mean(matches) if matches else None))
    write_csv(out/'stability_vs_future_gain.csv',curves)
    json_write(out/'analysis_complete.json',dict(states=len(rows),clean_states=len(clean)))
    print('Reports:',out,flush=True)


def rule_feature(logits,block,metric,torch):
    if metric=='top1_change': return block[0,1:].detach()
    if metric=='top5_distance': return greedy_top5(logits[0].detach().float(),torch)
    if metric=='mean_js': return logits[0].detach().float().log_softmax(-1)
    if metric=='peak_delta':
        x=logits[0].detach().float()
        return (x.max(-1).values-x.logsumexp(-1)).exp()
    raise ValueError(metric)


def feature_change(previous,current,metric,torch):
    current=current[:previous.shape[0]]
    if metric=='top1_change': return float((previous!=current).float().mean())
    if metric=='top5_distance': return float(1-(previous[:,:,None]==current[:,None,:]).any(-1).float().mean())
    if metric=='peak_delta': return float((previous-current).abs().mean())
    if metric=='mean_js':
        p,q=previous.exp(),current.exp()
        lm=((p+q)*.5).clamp_min(torch.finfo(p.dtype).tiny).log()
        return float((.5*((p*(previous-lm)).sum(-1)+(q*(current-lm)).sum(-1))).clamp_min(0).mean())
    raise ValueError(metric)


def adaptive_decode(target,draft,ids,rule,args,eos,Runtime,clock,torch):
    state=Runtime(target,draft,ids)
    prompt_len=ids.shape[1]
    rounds=[]
    draft_calls=candidate_work=0
    decision_ms=0.
    begin=clock()
    while state.prefix.shape[1]-prompt_len<args.max_new_tokens:
        remaining=args.max_new_tokens-(state.prefix.shape[1]-prompt_len)
        if remaining==1:
            count=state.ar_step()
            rounds.append(dict(k=0,accepted=0,committed=count,tried=[]))
        else:
            previous=None
            last_actual=0
            stable=0
            tried=[]
            for nominal in KS:
                actual=min(nominal,remaining-1)
                if actual==last_actual: break
                block,logits=state.propose(actual,return_logits=True)
                tried.append(actual)
                draft_calls+=1
                candidate_work+=actual
                last_actual=actual
                if actual<nominal or nominal==32: break
                t0=time.perf_counter()
                current=rule_feature(logits,block,rule['metric'],torch)
                if previous is not None:
                    change=feature_change(previous,current,rule['metric'],torch)
                    stable=stable+1 if change<=rule['threshold'] else 0
                previous=current
                decision_ms+=(time.perf_counter()-t0)*1000
                if stable>=rule['patience']: break
            result=state.verify(block,keep_cache=True)
            count=state.advance(block,result,cache_kept=True)
            rounds.append(dict(k=last_actual,accepted=result[0],committed=count,tried=tried))
        generated=state.prefix[0,prompt_len:]
        if eos is not None and bool((generated==eos).any()): break
    seconds=clock()-begin
    generated=state.prefix[0,prompt_len:prompt_len+args.max_new_tokens].tolist()
    if eos is not None and eos in generated: generated=generated[:generated.index(eos)+1]
    excess=sum(r['committed'] for r in rounds)-len(generated)
    if excess:
        rounds[-1]['committed']-=excess
        rounds[-1]['accepted']=max(0,rounds[-1]['committed']-1)
    return dict(tokens=len(generated),decode_s=seconds,decode_tok_s=len(generated)/seconds,
        rounds=len(rounds),accepted_tokens=sum(r['accepted'] for r in rounds if r['k']),
        spec_rounds=sum(r['k']>0 for r in rounds),draft_calls=draft_calls,candidate_work=candidate_work,
        selected_k_counts=dict(Counter(r['k'] for r in rounds if r['k'])),
        decision_wall_ms=decision_ms,round_details=rounds,output_sha256=signature(generated))


def evaluate(args):
    out=Path(args.output)
    metadata=json.loads((out/'metadata.json').read_text())
    chosen=json.loads((out/'chosen_rules.json').read_text())
    destination=out/'online'
    destination.mkdir(exist_ok=False)
    torch,tok,inputs,fingerprints,target,drafts,Runtime,clock=setup(args)
    if fingerprints!=chosen['fingerprints']: raise ValueError('Frozen weights changed')
    if any(getattr(args,k)!=metadata[k] for k in ('seed','max_new_tokens','max_prompt','samples')):
        raise ValueError('Online evaluation settings differ from collection')
    if [key for key,_,_ in inputs]!=[r['prompt_key'] for r in metadata['prompts']]: raise ValueError('Prompt selection changed')
    test_keys=set(chosen['test_prompt_keys'])
    if test_keys & set(chosen['cal_prompt_keys']): raise ValueError('Prompt leakage')
    records=[]
    with torch.inference_mode():
        for i,(key,prompt,ids) in enumerate(inputs):
            if key not in test_keys: continue
            for draft in drafts.values():
                warm=Runtime(target,draft,ids)
                for k in KS:
                    block,logits=warm.propose(k,return_logits=True)
                    warm.verify(block)
                    for metric in ('top1_change','top5_distance','mean_js','peak_delta'):
                        f=rule_feature(logits,block,metric,torch)
                        feature_change(f,f,metric,torch)
                del warm
            for repeat in range(args.eval_repeats):
                jobs=[(arm,mode) for arm in drafts for mode in ('fixed_16','fixed_32','stability')]
                random.Random(args.seed+i*100+repeat).shuffle(jobs)
                for arm,mode in jobs:
                    if mode=='stability':
                        r=adaptive_decode(target,drafts[arm],ids,chosen['choices'][arm]['rule'],
                            args,tok.eos_token_id,Runtime,clock,torch)
                    else:
                        r=replay(target,drafts[arm],ids,None,int(mode.split('_')[1]),None,None,args,
                            tok.eos_token_id,Runtime,clock,torch)
                        r.update(draft_calls=r['spec_rounds'],candidate_work=r['candidate_tokens'],decision_wall_ms=0.)
                    r.update(arm=arm,mode=mode,prompt_key=key,repeat=repeat)
                    records.append(r)
                    with (destination/'results.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
                    print(f"online test prompt={i+1} repeat={repeat+1} {arm} {mode} {r['decode_tok_s']:.2f} tok/s",flush=True)
    summary=[]
    for arm in drafts:
        baseline=[r for r in records if r['arm']==arm and r['mode']=='fixed_16']
        baseline_speed=sum(r['tokens'] for r in baseline)/sum(r['decode_s'] for r in baseline)
        for mode in ('fixed_16','fixed_32','stability'):
            rs=[r for r in records if r['arm']==arm and r['mode']==mode]
            tokens=sum(r['tokens'] for r in rs)
            seconds=sum(r['decode_s'] for r in rs)
            n=sum(r['spec_rounds'] for r in rs)
            summary.append(dict(arm=arm,mode=mode,prompts=len({r['prompt_key'] for r in rs}),
                tok_s=tokens/seconds,gain_vs_own16_pct=100*(tokens/seconds/baseline_speed-1),
                accepted=sum(r['accepted_tokens'] for r in rs)/max(n,1),
                draft_calls_per_verify=sum(r['draft_calls'] for r in rs)/max(n,1),
                candidate_work_per_output=sum(r['candidate_work'] for r in rs)/tokens))
    write_csv(destination/'comparison.csv',summary)
    for r in summary: print(json.dumps(r),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['collect','analyze','eval'])
    p.add_argument('--output',required=True)
    for name in ('target','fixed-draft','mixed-draft','data'):
        p.add_argument('--'+name)
    p.add_argument('--dataset',default='unknown')
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--states',type=int,default=4)
    p.add_argument('--stride',type=int,default=64)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--max-prompt',type=int,default=512)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--eval-repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if min(args.samples,args.states,args.stride,args.max_new_tokens,args.max_prompt,args.repeats,args.eval_repeats)<1:
        p.error('Positive limits required')
    if args.command!='analyze' and not all(getattr(args,k) for k in ('target','fixed_draft','mixed_draft','data')):
        p.error('collect/eval need target, both drafts and data')
    if args.command=='collect' and (args.samples<2 or args.max_new_tokens<33):
        p.error('Need >=2 prompts and max-new-tokens >=33')
    {'collect':collect,'analyze':analyze,'eval':evaluate}[args.command](args)


if __name__=='__main__':
    main()
