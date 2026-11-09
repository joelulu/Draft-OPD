"""One-state position x K top5 matrix, with actual target-AR probabilities.

Draft IDs/probabilities come from saved logits. Target's own probabilities were
not stored, so reconstruct the exact common AR state and run target only. No
draft weights or speculative evaluation are needed. Target after the rejection
boundary is explicitly common AR, not conditioned on each K's wrong proposals.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path

KS=[4,8,12,16,20,24,28,32]


def signature(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def model_identity(path):
    """Same byte fingerprint used by logit_stability collection."""
    root=Path(path).resolve()
    files=sorted(p for p in root.iterdir() if p.name=='config.json' or p.suffix in ('.safetensors','.bin'))
    if len(files)<2: raise ValueError('Expected local target config and weights')
    digest=hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def select_rows(folder,prompt_index,state,arm):
    wanted={'fixed','mixed'} if arm=='both' else {arm}
    rows={}
    with (folder/'states.jsonl').open() as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line)
            if row['prompt_index']==prompt_index and row['generated_position']==state and row['arm'] in wanted:
                if row['arm'] in rows: raise ValueError('Duplicate model/state')
                rows[row['arm']]=row
    if set(rows)!=wanted: raise ValueError(f'State/model missing: found {sorted(rows)}, expected {sorted(wanted)}')
    first=next(iter(rows.values()))
    for row in rows.values():
        if any(row[k]!=first[k] for k in ('prompt_key','prefix_sha256','anchor_id','context_length')):
            raise ValueError('Model states are not aligned')
        for k in KS:
            positions=row['trials'][str(k)]['positions']
            if len(positions)!=k: raise ValueError('Wrong number of candidate positions')
            for i,p in enumerate(positions,1):
                if p['position']!=i or not len(p['top5_ids'])==len(p['top5_probs'])==5:
                    raise ValueError('Misaligned top5 data')
                if any(type(t) is not int or t<0 for t in p['top5_ids']):
                    raise ValueError('Invalid token ID')
                if len(set(p['top5_ids']))!=5 or any(not math.isfinite(v) or not 0<=v<=1 for v in p['top5_probs']):
                    raise ValueError('Invalid top5 IDs/probabilities')
    return rows


def reconstruct_reference(model,input_ids,state,eos,torch,expected,positions=32):
    from transformers import DynamicCache
    cache=DynamicCache()
    prefix=input_ids.clone()
    with torch.inference_mode():
        output=model(input_ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
        anchor=output.logits[:,-1:].argmax(-1)
        for _ in range(state):
            if eos is not None and int(anchor.item())==eos:
                raise ValueError('Reference ended before requested state')
            output=model(anchor,past_key_values=cache,use_cache=True,logits_to_keep=1)
            prefix=torch.cat((prefix,anchor),dim=1)
            anchor=output.logits[:,-1:].argmax(-1)
        if signature(prefix[0].tolist())!=expected['prefix_sha256'] or int(anchor.item())!=expected['anchor_id']:
            raise ValueError('Reconstructed prefix/anchor differs from saved experiment; do not mix target references')
        result=[]
        for position in range(1,positions+1):
            if eos is not None and int(anchor.item())==eos:
                break
            output=model(anchor,past_key_values=cache,use_cache=True,logits_to_keep=1)
            logits=output.logits[0,-1].float()
            token=int(logits.argmax().item())
            prob=float(logits.log_softmax(-1)[token].exp().item())
            result.append(dict(position=position,token_id=token,probability=prob))
            anchor=anchor.new_tensor([[token]])
    return result,prefix


def verify_saved_blocks(model,prefix,rows,torch):
    """Restore the collector's full-prefill target cache and verify saved blocks."""
    from transformers import DynamicCache
    cache=DynamicCache()
    start=prefix.shape[1]
    result={}
    with torch.inference_mode():
        model(prefix,past_key_values=cache,use_cache=True,output_hidden_states=True,logits_to_keep=1)
        for arm,row in rows.items():
            result[arm]={}
            for k in KS:
                trial=row['trials'][str(k)]
                block=prefix.new_tensor([trial['block_ids']])
                if block.shape[1]!=k+1 or int(block[0,0])!=row['anchor_id']:
                    raise ValueError('Saved block/anchor malformed')
                cache.crop(start)
                out=model(block,position_ids=torch.arange(start,start+k+1,device=prefix.device)[None],
                    past_key_values=cache,use_cache=True,output_hidden_states=True)
                logits=out.logits[0,:k].float()
                ids=logits.argmax(-1)
                probabilities=logits.log_softmax(-1).gather(-1,ids[:,None]).exp()[:,0]
                entries=[]
                for i in range(k):
                    token=int(ids[i])
                    if token!=trial['positions'][i]['verify_target_id']:
                        raise ValueError('Target verification differs from saved experiment; abort instead of mixing')
                    entries.append(dict(position=i+1,token_id=token,probability=float(probabilities[i])))
                result[arm][str(k)]=entries
                cache.crop(start)
    return result


def validate_reference(reference,rows,fingerprint):
    first=next(iter(rows.values()))
    for field in ('prompt_key','prefix_sha256','anchor_id','generated_position'):
        if reference.get(field)!=first[field]: raise ValueError(f'Target reference mismatch: {field}')
    if reference.get('target_fingerprint')!=fingerprint: raise ValueError('Target fingerprint mismatch')
    refs={p['position']:p for p in reference['positions']}
    if sorted(refs)!=list(range(1,len(refs)+1)): raise ValueError('Reference positions not contiguous')
    for p in refs.values():
        if type(p['token_id']) is not int or p['token_id']<0 or not math.isfinite(p['probability']) or not 0<=p['probability']<=1:
            raise ValueError('Invalid target probability')
    for row in rows.values():
        verification=reference.get('verification_by_arm',{}).get(row['arm'],{})
        for k,t in row['trials'].items():
            verified=verification.get(k)
            if verified is None or len(verified)!=int(k): raise ValueError('Missing matching per-K target verification probabilities')
            for p,v in zip(t['positions'],verified):
                if v['position']!=p['position'] or v['token_id']!=p['verify_target_id']:
                    raise ValueError('Per-K target verification reference mismatch')
                if not math.isfinite(v['probability']) or not 0<=v['probability']<=1:
                    raise ValueError('Invalid target verification probability')
                old=p.get('reference_id')
                if old is not None and (p['position'] not in refs or refs[p['position']]['token_id']!=old):
                    raise ValueError('Target AR token differs from stored reference; abort instead of mixing')


def prob_text(prob):
    return f'{prob:.6f}' if prob>=1e-5 else f'{prob:.3e}'


def export(rows,reference,out,positions=32):
    out.mkdir(parents=True,exist_ok=False)
    refs={p['position']:p for p in reference['positions']}
    (out/'target_reference.json').write_text(json.dumps(reference,indent=2))
    sections=[]
    all_long=[]
    for arm in ('fixed','mixed'):
        if arm not in rows: continue
        row=rows[arm]
        maximum=max(t['accepted'] for t in row['trials'].values())
        wide=[]
        table=[]
        header='<tr><th class="position">Position</th><th class="target">Target AR<br>ID / probability</th>'
        header+=''.join(f'<th>K{k}<br><small>accepted={row["trials"][str(k)]["accepted"]}</small></th>' for k in KS)+'</tr>'
        for position in range(1,positions+1):
            target=refs.get(position)
            target_id=target['token_id'] if target else None
            target_prob=target['probability'] if target else None
            record=dict(position=position,target_id=target_id,target_probability=target_prob)
            label=f'{target_id}<br><strong>{prob_text(target_prob)}</strong>' if target else '—<br><small>after EOS</small>'
            cells=[f'<td class="position">{position}</td>',f'<td class="target">{label}</td>']
            previous=None
            for k in KS:
                if position>k:
                    record[f'K{k}']=''
                    cells.append('<td class="missing">—</td>')
                    continue
                p=row['trials'][str(k)]['positions'][position-1]
                verified=reference['verification_by_arm'][arm][str(k)][position-1]
                record[f'K{k}']=f"Tverify: {verified['token_id']} @ {verified['probability']:.9g}\n"+'\n'.join(
                    f'D{rank}: {token} @ {prob:.9g}' for rank,(token,prob) in enumerate(zip(p['top5_ids'],p['top5_probs']),1))
                current=p['top5_ids'][0]
                changed=previous is not None and current!=previous
                previous=current
                cell=[f'<div class="verify">T: <b>{verified["token_id"]}</b> <span class="prob">{prob_text(verified["probability"])}</span></div>']
                for rank,(token,prob) in enumerate(zip(p['top5_ids'],p['top5_probs']),1):
                    cls='match' if target_id is not None and token==target_id else ''
                    cell.append(f'<div class="{cls}"><span class="rank">{rank}</span> '
                                f'<b>{token}</b> <span class="prob">{prob_text(prob)}</span></div>')
                    all_long.append(dict(arm=arm,position=position,k=k,rank=rank,draft_token_id=token,
                        draft_probability=prob,target_ar_token_id=target_id,target_ar_probability=target_prob,
                        target_verify_token_id=verified['token_id'],target_verify_probability=verified['probability'],
                        matches_target_ar=token==target_id if target_id is not None else None,
                        matches_target_verify=token==verified['token_id']))
                cells.append(f'<td class="{"changed" if changed else ""}">'+''.join(cell)+'</td>')
            wide.append(record)
            table.append(f'<tr class="{"boundary" if position==maximum+1 else ""}">'+''.join(cells)+'</tr>')
        with (out/f'matrix_{arm}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(wide[0]))
            w.writeheader()
            w.writerows(wide)
        sections.append(f'<section><h2>{arm}-trained draft · observed A*={maximum}</h2>'
            '<div class="scroll"><table><thead>'+header+'</thead><tbody>'+''.join(table)+'</tbody></table></div></section>')
    with (out/'top5_long.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_long[0]))
        w.writeheader()
        w.writerows(all_long)
    first=next(iter(rows.values()))
    title=f"Top5 evolution · prompt {first['prompt_index']} · generated state {first['generated_position']}"
    page='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Top5 evolution</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;color:#172b40;background:#f6f8fb}h1{font-size:23px}h2{font-size:19px}
p{line-height:1.6;max-width:1100px}.scroll{overflow:auto;max-height:78vh;border:1px solid #cbd5df;background:white}
table{border-collapse:separate;border-spacing:0;font-family:ui-monospace,monospace;font-size:12px}
th,td{border-right:1px solid #dae1e8;border-bottom:1px solid #dae1e8;padding:8px;min-width:142px;vertical-align:top}
th{position:sticky;top:0;background:#eaf0f7;z-index:4;text-align:left}td div{white-space:nowrap;line-height:1.8}
.position{position:sticky;left:0;min-width:60px;background:#f0f4f8;z-index:2}.target{position:sticky;left:77px;min-width:122px;background:#edf4ff;z-index:2}
th.position,th.target{z-index:6}.missing{background:#f1f3f5;color:#8a95a1;text-align:center}.rank{color:#7c8995}
.prob{float:right;margin-left:12px;color:#46556a}.match{background:#dcf5e4;color:#14572c}.changed{box-shadow:inset 0 0 0 2px #f2a33a}
.verify{color:#174b88;border-bottom:1px dashed #aac1dc;margin-bottom:3px}
.boundary td{border-top:2px solid #537ab0}section{margin-top:24px}small{font-weight:normal}
</style><h1>'''+html.escape(title)+'''</h1>
<p>Each K cell: T = target verification top1 ID / its own probability, followed by five draft ranks with token ID / probability.
Probabilities use the full vocabulary (top5 probabilities are not renormalized).
Green = matches the common target-AR token. Orange outline = top1 changed versus the preceding available K at this position.
Gray = this position was not generated. Blue row border = observed A*+1 for this draft.</p>
<p>The left Target AR column is one common autoregressive reference with its own probabilities. Each cell's T uses that K's
draft-conditioned verification prefix and may differ after an earlier draft error. This is a position × K evolution matrix,
not a Markov transition probability matrix.</p>'''
    (out/'matrix.html').write_text(page+''.join(sections)+'</html>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--dataset',default='gsm8k')
    p.add_argument('--prompt-index',type=int,default=13)
    p.add_argument('--state',type=int,default=0)
    p.add_argument('--arm',choices=['fixed','mixed','both'],default='both')
    p.add_argument('--positions',type=int,default=32)
    p.add_argument('--target',default='/home/jovyan/LMM/lmm_model/Qwen3-4B')
    p.add_argument('--target-reference',help='Reuse a previously generated matching target_reference.json; CPU only')
    p.add_argument('--output')
    args=p.parse_args()
    if not 1<=args.positions<=32 or args.state<0: p.error('positions must be1..32 and state>=0')
    root=Path(args.root)
    folder=root if (root/'states.jsonl').exists() else root/args.dataset
    metadata=json.loads((folder/'metadata.json').read_text())
    rows=select_rows(folder,args.prompt_index,args.state,args.arm)
    first=next(iter(rows.values()))
    fingerprint=metadata['fingerprints']['target']
    if args.target_reference:
        reference=json.loads(Path(args.target_reference).read_text())
    else:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if not torch.cuda.is_available(): raise RuntimeError('Need one GPU to supplement target probabilities, or reuse --target-reference')
        torch.cuda.set_device(0)
        if model_identity(args.target)!=fingerprint: raise ValueError('Target weights differ from collection')
        prompt=next(r['prompt'] for r in metadata['prompts'] if r['prompt_key']==first['prompt_key'])
        tokenizer=AutoTokenizer.from_pretrained(args.target)
        text=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(text,add_special_tokens=False,return_tensors='pt').cuda()
        target=AutoModelForCausalLM.from_pretrained(args.target,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').cuda().eval().requires_grad_(False)
        print('Reconstructing common target AR context and probabilities...',flush=True)
        probabilities,prefix=reconstruct_reference(target,ids,args.state,tokenizer.eos_token_id,torch,first)
        reference={field:first[field] for field in ('prompt_key','prefix_sha256','anchor_id','generated_position')}
        reference.update(target_fingerprint=fingerprint,positions=probabilities,
            verification_by_arm=verify_saved_blocks(target,prefix,rows,torch),
            scope='common target AR next-candidate top1 and own full-softmax probability; excludes anchor')
    validate_reference(reference,rows,fingerprint)
    out=Path(args.output) if args.output else folder/(f'top5_matrix_prompt{args.prompt_index}_state{args.state}_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    export(rows,reference,out,args.positions)
    print(f'Open: {out}/matrix.html')
    print(f'CSV: {out}/matrix_fixed.csv and/or matrix_mixed.csv; long form: top5_long.csv')


if __name__=='__main__':
    main()
