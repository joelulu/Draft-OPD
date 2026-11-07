"""Read all target KV layers/heads, with either full or last-16 context."""
import math
import torch
from torch import nn
from . import K_VALUES


def cache_pairs(cache):
    if hasattr(cache, 'layers'):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    else:
        pairs = list(zip(cache.key_cache, cache.value_cache))
    if not pairs or any(k is None or v is None or k.shape[-2] == 0 for k, v in pairs):
        raise ValueError('Classifier requires a nonempty initialized target KV cache')
    return pairs


class KVClassifier(nn.Module):
    def __init__(self, anchor_dim, head_dim, layers, kv_heads, context_mode, hidden=128):
        super().__init__()
        if context_mode not in ('full_kv', 'window_16'):
            raise ValueError('Unknown context mode')
        self.settings = dict(anchor_dim=anchor_dim, head_dim=head_dim, layers=layers,
                             kv_heads=kv_heads, context_mode=context_mode, hidden=hidden)
        self.context_mode = context_mode
        self.query = nn.Linear(anchor_dim, head_dim)
        self.query_bias = nn.Parameter(torch.zeros(layers, kv_heads, head_dim))
        self.anchor = nn.Linear(anchor_dim, hidden)
        self.head = nn.Sequential(nn.Linear(head_dim+hidden+1, hidden), nn.SiLU(), nn.Linear(hidden, len(K_VALUES)))

    def forward(self, pairs, anchor):
        if len(pairs) != self.settings['layers'] or anchor.shape[0] != 1:
            raise ValueError('Classifier supports B=1 and the configured layer count')
        q = self.query(anchor.float())
        contexts = []
        for i, (keys, values) in enumerate(pairs):
            if keys.shape[1] != self.settings['kv_heads'] or keys.shape[-1] != self.settings['head_dim']:
                raise ValueError('Target KV geometry differs from classifier checkpoint')
            if self.context_mode == 'window_16':
                keys, values = keys[..., -16:, :], values[..., -16:, :]
            # Each frozen KV head is read by a K-independent anchor query.
            query = q[:, None, None, :] + self.query_bias[i][None, :, None, :]
            scores = (keys.detach().float()*query).sum(-1)/math.sqrt(keys.shape[-1])
            contexts.append((scores.softmax(-1)[..., None]*values.detach().float()).sum(-2).mean(1))
        context = torch.stack(contexts).mean(0)
        length = anchor.new_tensor([[math.log1p(pairs[0][0].shape[-2])]], dtype=torch.float32)
        return self.head(torch.cat((context, self.anchor(anchor.float()), length), -1))

    @torch.no_grad()
    def choose(self, runtime):
        embedding = runtime.target.get_input_embeddings()(runtime.anchor[:, 0])
        logits = self(cache_pairs(runtime.target_cache), embedding)
        return K_VALUES[int(logits.argmax(-1).item())]


class DFlashLengthClassifier(nn.Module):
    """Only this small head trains; all native DFlash feature operations freeze."""
    def __init__(self,feature_dim,context_mode,hidden=128):
        super().__init__()
        if context_mode not in ('full_kv','window_16'):
            raise ValueError('Unknown native context mode')
        self.settings=dict(kind='dflash_layer0_anchor_only_v1',feature_dim=feature_dim,
                           context_mode=context_mode,hidden=hidden)
        self.context_mode=context_mode
        self.norm=nn.LayerNorm(feature_dim)
        self.head=nn.Sequential(nn.Linear(feature_dim,hidden),nn.SiLU(),nn.Linear(hidden,len(K_VALUES)))

    def forward(self,features,unused_anchor=None):
        return self.head(self.norm(features.detach().float()))

    @torch.no_grad()
    def choose(self,runtime):
        logits=self(runtime.dflash_length_features(self.context_mode))
        return K_VALUES[int(logits.argmax(-1).item())]


def build_classifier(settings):
    settings=dict(settings)
    kind=settings.pop('kind','legacy_target_kv')
    if kind=='legacy_target_kv':
        return KVClassifier(**settings)
    if kind=='dflash_layer0_anchor_only_v1':
        return DFlashLengthClassifier(**settings)
    raise ValueError(f'Unsupported classifier feature schema: {kind}')
