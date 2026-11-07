"""Pure label/cost logic; usable without torch or a GPU."""
import math
import statistics
from collections import defaultdict
from . import K_VALUES


def make_labels(accepted, costs_ms, ks=K_VALUES):
    if len(accepted) != len(ks) or len(costs_ms) != len(ks):
        raise ValueError('Expected one acceptance and cost per K')
    if list(ks) != sorted(set(ks)) or not ks:
        raise ValueError('K must be unique and increasing')
    for k, a, cost in zip(ks, accepted, costs_ms):
        if a != int(a) or not 0 <= a <= k or not math.isfinite(cost) or cost <= 0:
            raise ValueError('Invalid acceptance/cost')
    scores = [(a+1)/c for a, c in zip(accepted, costs_ms)]
    ia = max(range(len(ks)), key=lambda i: accepted[i])
    iv = max(range(len(ks)), key=lambda i: scores[i])
    return dict(label_accept_k=ks[ia], label_speed_k=ks[iv], speed_scores=scores,
                cost_ms=list(costs_ms))


def context_bin(length):
    # Multiplicative bins avoid a single calibration for all context lengths.
    return max(1, int(length)).bit_length()-1


def calibrate(rows):
    buckets = defaultdict(lambda: [[] for _ in K_VALUES])
    for r in rows:
        if r['split'] != 'train':
            raise ValueError('Cost calibration must use training states only')
        for i, repeats in enumerate(r['total_times_ms']):
            buckets[context_bin(r['context_length'])][i].extend(repeats)
    if not buckets:
        raise ValueError('No training states for calibration')
    return {str(b): [statistics.median(v) for v in values] for b, values in buckets.items()}


def costs_for(table, length):
    b = context_bin(length)
    nearest = min(map(int, table), key=lambda v: (abs(v-b), v))
    return table[str(nearest)]
