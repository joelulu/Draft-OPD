"""Frozen native DFlash layer-0 anchor-only features, before K is selected.

Uses actual draft context KV, QK norms, RoPE, GQA, O projection, residuals
and MLP. No future MASKs: later-layer/full-block queries depend on K.
"""
import torch
from transformers import DynamicCache
from model.dflash import rotate_half
from model.utils import extract_context_feature
from .classifier import cache_pairs

FEATURE_SCHEMA='dflash_layer0_anchor_only_v1'


def check_draft(draft):
    if draft.training:
        raise ValueError('Native length features require an eval-mode frozen draft')
    rope=getattr(draft.config,'rope_scaling',None) or {}
    rope_type=str(rope.get('rope_type',rope.get('type','default')))
    if 'dynamic' in rope_type or 'longrope' in rope_type:
        raise NotImplementedError('K-dependent/dynamic RoPE needs separate native-reader validation')


@torch.no_grad()
def materialize_context(draft,cache,target_hidden,start):
    """Append only confirmed context, using the exact native context projections."""
    check_draft(draft)
    if target_hidden.shape[0]!=1 or start!=cache.get_seq_length():
        raise ValueError('Draft context cache must be aligned to the confirmed prefix')
    count=target_hidden.shape[1]
    if count==0:
        return
    projected=draft.hidden_norm(draft.fc(target_hidden))
    positions=torch.arange(start,start+count,device=projected.device)[None]
    cos,sin=draft.rotary_emb(projected,positions)
    for i,layer in enumerate(draft.layers):
        if i and cache.get_seq_length(i)!=start:
            raise ValueError('Draft context cache layers have different lengths')
        attn=layer.self_attn
        keys=attn.k_proj(projected).view(1,count,-1,attn.head_dim)
        keys=attn.k_norm(keys).transpose(1,2)
        values=attn.v_proj(projected).view(1,count,-1,attn.head_dim).transpose(1,2)
        keys=keys*cos[:,None]+rotate_half(keys)*sin[:,None]
        cache.update(keys,values,i,{'cos':cos,'sin':sin})


class _ReadOnlyPrefix:
    """Native attention appends anchor KV here, without changing the real cache."""
    def __init__(self,keys,values):
        self.keys,self.values=keys,values
    def update(self,keys,values,layer_idx,cache_kwargs=None):
        if layer_idx!=0 or keys.shape[-2]!=1:
            raise ValueError('Expected a single native layer-0 anchor query')
        return torch.cat((self.keys,keys),-2),torch.cat((self.values,values),-2)


@torch.no_grad()
def read_anchor(draft,cache,anchor_embedding,position,context_mode):
    """Native first decoder layer, historical context + anchor, zero MASK tokens."""
    check_draft(draft)
    if context_mode not in ('full_kv','window_16'):
        raise ValueError('Unknown native context mode')
    if cache.get_seq_length()!=position or anchor_embedding.shape[:2]!=(1,1):
        raise ValueError('Anchor position must follow the confirmed context cache')
    keys,values=cache_pairs(cache)[0]
    if keys.shape[-2]!=position:
        raise ValueError('Native reader currently requires an untruncated absolute-position cache')
    if context_mode=='window_16':
        keys,values=keys[...,-16:,:],values[...,-16:,:]
    # Cropping KV never resets positions or changes GQA head correspondence.
    context_positions=torch.arange(position-keys.shape[-2],position,device=keys.device)
    key_positions=torch.cat((context_positions,context_positions.new_tensor([position])))
    window=draft.layers[0].self_attn.sliding_window
    allowed=torch.ones_like(key_positions,dtype=torch.bool)
    if window is not None:
        allowed=key_positions>position-int(window)
    attention_mask=allowed[None,None,None,:]
    positions=key_positions.new_tensor([[position]])
    rotary=draft.rotary_emb(anchor_embedding,positions)
    result=draft.layers[0](hidden_states=anchor_embedding,
        target_hidden=anchor_embedding.new_empty((1,0,draft.config.hidden_size)),
        position_ids=positions,position_embeddings=rotary,
        attention_mask=attention_mask,past_key_value=_ReadOnlyPrefix(keys,values),use_cache=True)
    return result[:,0].detach()


def iter_dflash_states(target,draft,rows):
    """Reconstruct both native caches from each original labeled trajectory."""
    grouped={}
    for row in rows:
        grouped.setdefault(row['prompt_key'],[]).append(row)
    for group in grouped.values():
        target_cache,draft_cache=DynamicCache(),DynamicCache()
        prefix=[]
        for row in sorted(group,key=lambda r:r['context_length']):
            ids=row['prefix_ids']
            if ids[:len(prefix)]!=prefix or len(ids)<=len(prefix):
                raise ValueError('Native features require a monotonic confirmed-prefix trajectory')
            with torch.no_grad():
                outputs=target(torch.tensor([ids[len(prefix):]],device=target.device),
                    past_key_values=target_cache,use_cache=True,output_hidden_states=True,logits_to_keep=1)
                context=extract_context_feature(outputs.hidden_states,draft.target_layer_ids)
                materialize_context(draft,draft_cache,context,len(prefix))
                del outputs,context
                embedding=target.get_input_embeddings()(torch.tensor([[row['anchor_id']]],device=target.device))
                features={mode:read_anchor(draft,draft_cache,embedding,len(ids),mode)
                    for mode in ('full_kv','window_16')}
            prefix=ids
            yield row,features,None
