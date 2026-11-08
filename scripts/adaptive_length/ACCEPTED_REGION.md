# Post-hoc accepted-region evolution (no stopping policy)

Uses existing `logit_stability_TIMESTAMP/{dataset}/states.jsonl`, no GPU, no new
collection, no threshold selection, no online evaluation. The region endpoint
is A*=max accepted across the eight measured Ks at a state, not acceptance at
K32. This is an observed post-hoc maximum, not a known universal true horizon.

```bash
python scripts/adaptive_length/analyze_accepted_region.py \
  --root runs/fixed_vs_dynamic_20261107_051744/logit_stability_20261108_155129 \
  --prompt-index 13 --example-states 0 192
```

Processes both datasets/drafts. Optional `--dataset gsm8k` limits to one. Console
and Markdown examples show prompt13 at states0/192. All states are analyzed.

For A*=8, region `best_accepted` covers positions1..8; K4 contributes only1..4.
K8 and larger contribute1..8. K4's ungenerated5..8 are absent, never counted as
unchanged or incorrect. In addition to best-accepted range, stores fixed prefix
ranges4/8/12/.../32, the single position A*+1, and `previous_prefix` (K16 means
prefix1..12). This lets the user inspect history through K16 over first12 even
when A* differs. A*=0 has an empty best-accepted region, not falsely stable.

Histories accumulate every available adjacent-K comparison through current K.
At K16 over positions1..12:

- positions1..4 have K4->8->12->16: three transitions each;
- positions5..8 have K8->12->16: two transitions each;
- positions9..12 have K12->16: one transition each.

There are24 position-transition observations, not36. New positions with a single
observation have no stability estimate. Reports both edge-weighted and
position-weighted JS, number of observed positions/transitions, top1 flips,
unordered top5 overlap, peak/margin changes, common-AR reference probabilities
and hits. Net change versus first available K is separate from cumulative flips,
so A->B->A is correctly identified as oscillation rather than stable.
Also compares current predictions with **every** earlier available K at that
position: current-vs-history top1 disagreement, top5 overlap, peak and margin
delta, versus-first-K overlap and modal agreement. E.g. at K16, position1
compares current with K4/K8/K12; position10 compares with K12 only. This is
separate from adjacent-path metrics and can be computed exactly from top5.

JS comes from saved full-distribution adjacent comparisons. A JS path sum or
mean is **not** endpoint JS and is not called that. Full nonadjacent distribution
JS cannot be recovered from top5 alone. Different K have different coverage;
use fixed prefix4 once comparing all Ks, or explicit denominators when using
expanding prefix histories. K32 is not the ground-truth distribution.

Fresh report directory under the source root:

- `cumulative_regions.csv`: each state/currentK/region, actual A(K), A*, first
  best K, coverage, cumulative and last-edge metrics, top1/top5 changes.
- `position_histories.csv`: each state/currentK/position, earliest observed K,
  all observed top1 IDs, flip counts, oscillation, JS paths, confidence/reference.
- `adjacent_position_changes.csv`: same-state adjacent changes with A* range
  tags and reference-correct repairs/breaks, preserving boundary information.
- `group_summary.csv`: dataset/model/A* bucket/currentK/region state-weighted
  summaries. EOS-adjusted states are excluded here but retained/flagged in raw
  reports. Prompt correlation means these are descriptive, not significance tests.
- Example `.md` files: per-K acceptance/history table and per-position top1
  token text/probability/reference-hit grid. The original full top5 remains in
  the source `top5_grid_32.csv` and `states.jsonl`.
- `scope.json`: provenance and metric definitions.

Tests: `python scripts/adaptive_length/test_accepted_region.py`.
