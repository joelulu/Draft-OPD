"""B=1 greedy reference runtime; lengths exclude anchor, no draft-hidden carriers."""
import time
import torch
from transformers import DynamicCache
from model.utils import extract_context_feature


def clock():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class RoundRuntime:
    def __init__(self, target, draft, input_ids):
        if input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Only a single nonempty, unpadded prompt is supported.")
        self.target, self.draft = target, draft
        self.target_cache, self.draft_cache = DynamicCache(), DynamicCache()
        self.prefix = input_ids.clone()
        with torch.no_grad():
            out = target(input_ids, past_key_values=self.target_cache, use_cache=True,
                         output_hidden_states=True, logits_to_keep=1)
            self.anchor = out.logits[:, -1:].argmax(-1)
            self.context = extract_context_feature(out.hidden_states, draft.target_layer_ids)
        self.context_start = 0

    def features(self):
        # Pivot predicts anchor. Anchor's own target hidden has not been computed.
        pivot = self.draft.hidden_norm(self.draft.fc(self.context[:, -1]))
        anchor = self.target.get_input_embeddings()(self.anchor[:, 0])
        return pivot.float(), anchor.float()

    def propose(self, k, *, return_logits=False, forward_model=None):
        if not 1 <= k <= 32:
            raise ValueError("k excludes anchor and must be in [1,32].")
        start = self.prefix.shape[1]
        cached = self.draft_cache.get_seq_length()
        if cached < self.context_start or cached > start:
            raise RuntimeError("Draft cache is not aligned with confirmed target context.")
        block = torch.full((1, k + 1), self.draft.mask_token_id, device=self.prefix.device, dtype=torch.long)
        block[:, :1] = self.anchor
        context = self.context[:, cached - self.context_start:]
        positions = torch.arange(cached, start + k + 1, device=block.device)[None]
        hidden = (forward_model if forward_model is not None else self.draft)(
            noise_embedding=self.target.get_input_embeddings()(block), target_hidden=context,
            position_ids=positions, past_key_values=self.draft_cache,
            use_cache=True, is_causal=False)
        logits = self.target.lm_head(hidden[:, 1:])
        block[:, 1:] = logits.detach().argmax(-1)
        self.draft_cache.crop(start)
        return (block, logits) if return_logits else block

    def verify(self, block, *, keep_cache=False, return_logits=False):
        start = self.prefix.shape[1]
        self.target_cache.crop(start)
        positions = torch.arange(start, start + block.shape[1], device=block.device)[None]
        with torch.no_grad():
            out = self.target(block, position_ids=positions, past_key_values=self.target_cache,
                              use_cache=True, output_hidden_states=True)
        posterior = out.logits.argmax(-1)
        accepted = int((block[:, 1:] == posterior[:, :-1]).cumprod(-1).sum().item())
        if not keep_cache:
            self.target_cache.crop(start)
        result = (accepted, posterior, extract_context_feature(out.hidden_states, self.draft.target_layer_ids))
        return (result, out.logits[:, :-1]) if return_logits else result

    def advance(self, block, result, *, cache_kept=False):
        accepted, posterior, context = result
        count = accepted + 1
        start = self.prefix.shape[1]
        if cache_kept:
            self.target_cache.crop(start + count)
        else:
            self.target_cache.crop(start)
            with torch.no_grad():
                self.target(block[:, :count], past_key_values=self.target_cache,
                            use_cache=True, logits_to_keep=1)
        self.prefix = torch.cat((self.prefix, block[:, :count]), dim=1)
        self.anchor = posterior[:, accepted:accepted + 1]
        self.context = context[:, :count].detach()
        self.context_start = start
        return count

    def step(self, k):
        block = self.propose(k)
        result = self.verify(block, keep_cache=True)
        return self.advance(block, result, cache_kept=True)

    def ar_step(self):
        start = self.prefix.shape[1]
        out = self.target(self.anchor, past_key_values=self.target_cache,
                          use_cache=True, output_hidden_states=True, logits_to_keep=1)
        self.prefix = torch.cat((self.prefix, self.anchor), dim=1)
        self.anchor = out.logits[:, -1:].argmax(-1)
        self.context = extract_context_feature(out.hidden_states, self.draft.target_layer_ids)
        self.context_start = start
        return 1
