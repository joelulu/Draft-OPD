"""Small head using only features available before drafting."""
import torch
from torch import nn
import torch.nn.functional as F


def validate_lengths(lengths):
    lengths = tuple(int(k) for k in lengths)
    if not lengths or lengths != tuple(sorted(set(lengths))) or not all(1 <= k <= 32 for k in lengths):
        raise ValueError("Candidate lengths must be unique, increasing integers in [1, 32].")
    return lengths


class LengthHead(nn.Module):
    def __init__(self, hidden_size, lengths=(4, 8, 12, 16, 24, 32), projection=128, features="both"):
        super().__init__()
        self.lengths = validate_lengths(lengths)
        if features not in ("both", "anchor", "pivot"):
            raise ValueError(features)
        self.settings = dict(hidden_size=hidden_size, lengths=list(lengths), projection=projection, features=features)
        self.features = features
        self.pivot = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, projection), nn.SiLU())
        self.anchor = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, projection), nn.SiLU())
        self.mlp = nn.Sequential(nn.Linear(projection * (2 if features == "both" else 1), projection),
                                 nn.SiLU(), nn.Linear(projection, len(lengths)))
        self.register_buffer("capacities", torch.tensor(lengths, dtype=torch.float32))

    def forward(self, pivot, anchor):
        pieces = []
        if self.features != "anchor":
            pieces.append(self.pivot(pivot.float()))
        if self.features != "pivot":
            pieces.append(self.anchor(anchor.float()))
        return torch.sigmoid(self.mlp(torch.cat(pieces, dim=-1))) * self.capacities

    def loss(self, pivot, anchor, accepted):
        # Equal weight per action; the target is realized greedy prefix acceptance.
        return F.mse_loss(self(pivot, anchor), accepted.float())


def select_lengths(expected_acceptance, costs_ms, lengths, remaining=None):
    """Return action indices. Only select widths that fit the remaining output budget."""
    costs = torch.as_tensor(costs_ms, device=expected_acceptance.device, dtype=torch.float32)
    if costs.ndim != 1 or costs.numel() != len(lengths) or expected_acceptance.shape[-1] != len(lengths):
        raise ValueError("Expected one prediction and one cost per candidate length.")
    if not torch.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Every action needs a finite positive draft+verify cost.")
    scores = (expected_acceptance + 1) / costs
    if remaining is not None:
        allowed = torch.as_tensor(lengths, device=scores.device) + 1 <= remaining
        scores = scores.masked_fill(~allowed, -torch.inf)
        if not allowed.any():
            raise ValueError("No configured length fits; use an AR tail step.")
    return scores.argmax(dim=-1)
