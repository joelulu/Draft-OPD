"""Exact candidate-token budgets for fixed/mixed-length OPD comparisons."""


def width_plan(lengths, budget, rng):
    """Randomly tile a per-rank update budget, without off-grid short blocks.

    Sampling is uniform among feasible widths at each draw; end-of-budget choices
    are constrained. The realized width histogram must therefore be reported.
    """
    if budget <= 0:
        raise ValueError("Candidate-token budget must be positive.")
    lengths = tuple(sorted(set(lengths)))
    if not lengths or min(lengths) < 1 or max(lengths) > 32:
        raise ValueError("Widths must be in [1,32].")
    reachable = [True] + [False] * budget
    for n in range(1, budget + 1):
        reachable[n] = any(k <= n and reachable[n-k] for k in lengths)
    if not reachable[budget]:
        raise ValueError(f"Budget {budget} cannot be tiled by candidate lengths {lengths}.")
    plan, remaining = [], budget
    while remaining:
        k = rng.choice([k for k in lengths if k <= remaining and reachable[remaining-k]])
        plan.append(k)
        remaining -= k
    return plan
