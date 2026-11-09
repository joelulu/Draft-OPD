"""Frozen fixed-K16 trajectory: locate first errors and audit one-token oracle repair."""
import argparse
import csv
import json
from pathlib import Path

THRESHOLDS=(0.1,0.2,0.4)


def choose(probs, threshold):
    # Positions are zero-based internally, returned one-based for reports.
    return next((i+1 for i in range(len(probs)-1)
        if probs[i][0]-probs[i][1]<=threshold
        and any(p[0]>0.8 for p in probs[i+1:])),None)


def category(position, accepted, width=16):
    if position is None: return 'miss' if accepted<width else 'no_selection_all_correct'
    if accepted==width or position<accepted+1: return 'false_positive_correct'
    return 'hit_first_error' if position==accepted+1 else 'after_first_error'


def summarize(rows):
    output=[]
    # Common EOS exclusion across every method keeps comparisons paired.
    eligible=[r for r in rows if not r['any_eos']]
    for threshold in THRESHOLDS:
        trials=[r['rules'][str(threshold)] for r in eligible]
        n=len(eligible); selected=sum(t['position'] is not None for t in trials)
        hits=sum(t['category']=='hit_first_error' for t in trials)
        errors=sum(r['accepted']<16 for r in eligible)
        cover=sum(r['accepted']<15 for r in eligible)
        base=sum(r['accepted'] for r in eligible)
        repaired=sum(t['accepted_repaired'] for t in trials)
        output.append(dict(threshold=threshold,blocks_total=len(rows),blocks_non_eos=n,
            eos_blocks_excluded=len(rows)-n,selected=selected,hit_first_error=hits,
            false_positive_correct=sum(t['category']=='false_positive_correct' for t in trials),
            after_first_error=sum(t['category']=='after_first_error' for t in trials),
            missed_errors=sum(t['position'] is None and r['accepted']<16 for t,r in zip(trials,eligible)),
            errors=errors,uncoverable_errors=errors-cover,
            precision=hits/selected if selected else None,
            recall_all_errors=hits/errors if errors else None,
            recall_coverable_errors=hits/cover if cover else None,
            mean_accepted=base/n if n else None,
            mean_accepted_repaired=repaired/n if n else None,
            mean_gain=(repaired-base)/n if n else None,
            improved_blocks=sum(t['accepted_repaired']>r['accepted'] for t,r in zip(trials,eligible)),
            mean_perfect_first_error_repair=sum(r['perfect_accepted'] for r in eligible)/n if n else None))
    return output


def dump_csv(path, rows):
    if not rows: return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True);p.add_argument('--fixed-draft',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cross-k-repair',action='store_true',help='Add K24 and draft-only cross-K corrections')
    p.add_argument('--samples',type=int,default=16);p.add_argument('--max-new-tokens',type=int,default=512)
    args=p.parse_args()
    if args.samples<1 or args.max_new_tokens<17: p.error('Need positive samples and at least 17 output tokens')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from model.dflash import DFlashDraftModel
    from adaptive_length.runtime import RoundRuntime
    from prepare_gsm8k_oracle import prepare
    torch.manual_seed(42);torch.cuda.set_device(0)
    tok=AutoTokenizer.from_pretrained(args.target)
    prompts=prepare((json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()),args.samples)
    (out/'prompts.json').write_text(json.dumps(prompts,ensure_ascii=False,indent=2))
    (out/'metadata.json').write_text(json.dumps(dict(**vars(args),thresholds=THRESHOLDS,
        rule_version='any_later_top1_gt_0.8',following_top1_threshold=0.8,
        following_comparison='strictly_greater',following_positions='any later candidate in K16 block',selection='earliest eligible K16 position; cross-K draft-only repair when enabled',
        scope='paired states on original K16 trajectory; oracle and optional draft-only one-token repairs, not a speed benchmark'),indent=2))
    target=AutoModelForCausalLM.from_pretrained(args.target,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    draft=DFlashDraftModel.from_pretrained(args.fixed_draft,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    from cross_k_repair import select_tokens,summarize_cross
    eos=tok.eos_token_id
    def details(logits):
        values=logits[0].float();lp=values.log_softmax(-1);work=values.clone();ids=[]
        for _ in range(2):
            ix=work.argmax(-1,keepdim=True);ids.append(ix);work.scatter_(-1,ix,float('-inf'))
        ix=torch.cat(ids,-1)
        return ix.tolist(),lp.gather(-1,ix).exp().tolist()
    def has_eos(block,result):
        return eos is not None and eos in block[0,:result[0]+1].tolist()
    all_rows=[];flat=[];prompt_stats=[]
    with (out/'rounds.jsonl').open('w') as stream,torch.inference_mode():
        for pi,prompt in enumerate(prompts,1):
            text=tok.apply_chat_template(prompt['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(text,add_special_tokens=False,return_tensors='pt').cuda()
            rt=RoundRuntime(target,draft,ids);generated=0;round_index=0;tail=0
            while generated<args.max_new_tokens:
                if eos is not None and int(rt.anchor.item())==eos: break
                if args.max_new_tokens-generated<17:
                    rt.ar_step();generated+=1;tail+=1;continue
                round_index+=1
                b8,l8=rt.propose(8,return_logits=True)
                b16,l16=rt.propose(16,return_logits=True)
                if args.cross_k_repair:
                    b24,l24=rt.propose(24,return_logits=True)
                    ids24,pr24=details(l24)
                    distributions={k:l[0].float().softmax(-1) for k,l in [(8,l8),(16,l16),(24,l24)]}
                base=rt.verify(b16)
                ids8,pr8=details(l8);ids16,pr16=details(l16)
                record=dict(prompt_index=pi,source_index=prompt['source_index'],round=round_index,
                    generated_position=generated,context_length=rt.prefix.shape[1],
                    block_ids=b16[0].tolist(),accepted=base[0],first_error=base[0]+1 if base[0]<16 else None,
                    top2_ids_k8=ids8,top2_probs_k8=pr8,top2_ids_k16=ids16,top2_probs_k16=pr16,rules={})
                any_eos=has_eos(b16,base)
                for threshold in THRESHOLDS:
                    position=choose(pr16,threshold)
                    repair=b16.clone();repaired=base;new_token=None
                    if position is not None:
                        new_token=int(base[1][0,position-1]);repair[0,position]=new_token
                        repaired=rt.verify(repair)
                    any_eos=any_eos or has_eos(repair,repaired)
                    trial=dict(position=position,category=category(position,base[0]),
                        replacement_target_id=new_token,accepted_repaired=repaired[0],
                        gain=repaired[0]-base[0],
                        correct_id_in_k16_top2=(new_token in ids16[position-1]) if position else None,
                        correct_id_in_k8_top2=(new_token in ids8[position-1]) if position and position<=8 else None)
                    record['rules'][str(threshold)]=trial
                    flat.append(dict(prompt_index=pi,round=round_index,threshold=threshold,accepted=base[0],**trial))
                perfect=b16.clone();perfect_result=base
                if base[0]<16:
                    perfect[0,base[0]+1]=base[1][0,base[0]]
                    perfect_result=rt.verify(perfect)
                any_eos=any_eos or has_eos(perfect,perfect_result)
                record.update(perfect_accepted=perfect_result[0],any_eos=any_eos)
                if args.cross_k_repair:
                    record.update(top2_ids_k24=ids24,top2_probs_k24=pr24,cross={})
                    cross_eos=any_eos
                    for threshold in THRESHOLDS:
                        position=record['rules'][str(threshold)]['position']
                        original=int(b16[0,position]) if position else None
                        target_id=int(base[1][0,position-1]) if position else None
                        diagnostics=None
                        if position:
                            top={k:ids[position-1] for k,ids in [(8,ids8),(16,ids16),(24,ids24)] if position<=k}
                            union=sorted(set(t for ids in top.values() for t in ids))
                            probabilities={k:dict(zip(union,dist[position-1,union].tolist())) for k,dist in distributions.items() if k in top}
                            replacements=select_tokens(probabilities,top,original)
                            diagnostics=dict(top2_by_k=top,candidate_probabilities=probabilities)
                        else:
                            replacements={name:None for name in ('vote','mean_probability','contrast16_8','contrast24_16')}
                        methods={}
                        for method,token in replacements.items():
                            changed=position is not None and token!=original
                            repair=b16.clone();result=base
                            if changed:
                                repair[0,position]=token
                                result=rt.verify(repair)
                            cross_eos=cross_eos or has_eos(repair,result)
                            methods[method]=dict(position=position,original_token=original,replacement_token=token,
                                changed=changed,matches_target_at_position=token==target_id if position else False,
                                accepted_repaired=result[0],gain=result[0]-base[0])
                        record['cross'][str(threshold)]=methods
                        record.setdefault('cross_diagnostics',{})[str(threshold)]=diagnostics
                    record['cross_any_eos']=cross_eos
                for selected in flat[-len(THRESHOLDS):]: selected['eos_excluded']=any_eos
                all_rows.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
                print(json.dumps(dict(prompt=pi,round=round_index,accepted=base[0],
                    first_error=record['first_error'],rules=record['rules'],perfect=perfect_result[0],eos_excluded=any_eos)),flush=True)
                # All interventions are discarded. Re-verify original block and commit only it.
                committed=rt.verify(b16,keep_cache=True)
                if committed[0]!=base[0]: raise RuntimeError('Original acceptance changed after interventions')
                count=rt.advance(b16,committed,cache_kept=True);generated+=count
                if has_eos(b16,base): break
            prompt_stats.append(dict(prompt_index=pi,blocks=round_index,generated_tokens=generated,ar_tail_steps=tail))
            print(f'完成 prompt {pi}/{len(prompts)}，累计 {len(all_rows)} blocks',flush=True)
            if args.cross_k_repair:
                (out/'cross_summary.json').write_text(json.dumps(summarize_cross(all_rows,THRESHOLDS),indent=2))
                dump_csv(out/'cross_summary.csv',summarize_cross(all_rows,THRESHOLDS))
            (out/'summary.json').write_text(json.dumps(summarize(all_rows),indent=2))
    dump_csv(out/'summary.csv',summarize(all_rows));dump_csv(out/'selections.csv',flat);dump_csv(out/'prompt_stats.csv',prompt_stats)
    print(json.dumps(summarize(all_rows),indent=2),flush=True)
    if args.cross_k_repair:
        print('跨 K 实际修正统计：',flush=True)
        print(json.dumps(summarize_cross(all_rows,THRESHOLDS),indent=2),flush=True)
    print(f'Results: {out}',flush=True)


if __name__=='__main__': main()
