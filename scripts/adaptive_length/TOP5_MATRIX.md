# One-state position x K matrix

```bash
EVAL_GPU=7 bash scripts/adaptive_length/run_top5_matrix.sh \
  --root runs/fixed_vs_dynamic_20261107_051744/logit_stability_20261108_155129 \
  --dataset gsm8k --prompt-index 13 --state 0 --arm both
```

Draft predictions are reused from saved `states.jsonl`; no draft model is loaded
and no online strategy is run. Existing data contain **draft** probabilities
and target token IDs, not target's own probabilities. This command loads only
the frozen target on one GPU, checks its byte fingerprint, reconstructs the
common AR prefix/anchor, and supplements real target probabilities. It also
re-verifies saved draft blocks (no new drafting) to obtain each K's target
verification probabilities. Every stored target ID must match, otherwise abort.

The HTML matrix has rows=positions1..32, columns=K4..32 step4. Each K cell shows:

```text
T: target verification token ID   probability
1  draft top1 token ID           probability
2  draft top2 token ID           probability
3  draft top3 token ID           probability
4  draft top4 token ID           probability
5  draft top5 token ID           probability
```

The left sticky column additionally shows **common target AR** top1 ID and its
own probability for that position, constant across Ks/models. After an earlier
draft error, per-K verification is conditioned on that draft's wrong prefix,
whereas AR follows target's own correct trajectory. They are distinct and both
are labeled explicitly. All probabilities use the full vocabulary; the five
displayed draft probabilities are not renormalized. The collected draft
`reference_prob` is never substituted for target's own probability.

Gray means position not generated at that K, not zero probability. Green marks
draft candidates matching common AR token; orange outlines a changed top1
relative to preceding available K at the same position. A*+1 boundary is marked
per draft. K32 is not asserted to be correct. This is an evolution matrix, not
a Markov transition probability matrix.

Files in fresh `top5_matrix_prompt13_state0_TIMESTAMP/` under the dataset:

- `matrix.html`: fixed/mixed sections, horizontally scrollable with sticky
  position/target headers; download and open in a browser.
- `matrix_fixed.csv`, `matrix_mixed.csv`: same wide matrices for a spreadsheet.
- `top5_long.csv`: numeric position/K/rank/token/probability and both target
  references, suitable for precise statistics.
- `target_reference.json`: validated target probabilities/provenance, reusable
  with `--target-reference /path/to/target_reference.json` without loading GPU.

`--state 192` selects the other observed state, `--arm fixed` or `--arm mixed`
exports one model; `--positions 12` displays only first12. Cached reference must
include requested arms and matching prompt/prefix/anchor/fingerprint. AR data
after EOS are absent; per-K conditional verification can still be displayed,
as with the original diagnostic. Target-AR probabilities reflect sequential
generation, per-K verification probabilities reflect full block verification;
small floating-point differences are possible even for a shared prefix.

CPU tests: `python scripts/adaptive_length/test_top5_matrix.py`. The real-target
test requires torch/transformers but uses tiny random weights, no downloads.

## 十个样本的中文折叠报告

```bash
EVAL_GPU=7 bash scripts/adaptive_length/run_top5_matrix.sh \
  --root runs/fixed_vs_dynamic_20261107_051744/logit_stability_20261108_155129 \
  --dataset gsm8k --state 192 --samples 10 --arm both
```

按 prompt 编号选取在指定 state 同时保存 fixed/mixed 的前十条不同 prompt，
不按接受长度或改善幅度筛选。每条 prompt 的一个 KV+anchor 状态算一个样本。
若不足十条直接报错，不静默减少样本数。target 仅加载一次，
每个样本单独校验上下文和验证 token，保留原始 CSV 与 target_reference.json。
总览 `matrix.html` 包含十个可折叠样本、二十张表，以及中文介绍和图例。
不重新训练或重新生成 draft。需要在已有原始数据和 target 权重的服务器运行。
