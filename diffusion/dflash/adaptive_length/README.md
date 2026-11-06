# Qwen3-4B：单次起草前的长度选择

实现：多长度 Draft-OPD 参考训练 → 冻结 Draft 后采集配对标签 → 训练长度头 → 动态宽度 greedy 评测。
不使用跨块 Draft hidden；已有 verl/SGLang 的入口不变。新增的是独立 PyTorch 研究入口，尚未接入服务框架调度。

## 输入、长度和目标

- `K={4,8,12,16,24,32}` 表示候选数，**不含 anchor**，实际前向宽度是 `K+1`。
- pivot：最后一个已被 Target 计算的位置，提取 `target_layer_ids` 多层 hidden，复用 Draft 的 `fc + hidden_norm` 融合。
- anchor：pivot 预测的 token，只查 Target embedding，不读取尚不存在的 anchor Target hidden。
- 每路 `LayerNorm → Linear(128) → SiLU`，拼接后 `256 → 128 → 6`；输出各长度的预期连续接受数。
- 策略最大化 `(预期接受数+1)/(Draft起草+Target验证成本)`。支持 `both / anchor / pivot` 输入消融。
- 每种 K 都在同一前缀实际前向、验证；不会用一次长度 32 的截断结果生成其他长度标签。

## 第一轮：先冻结原有 Draft 检查长度是否可预测

```bash
cd ~/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/Draft-OPD
git fetch origin
git switch research/qwen3-4b-adaptive-length
git pull --ff-only origin research/qwen3-4b-adaptive-length

export RESULT_ROOT="$PWD/runs/adaptive_length_smoke_$(date +%Y%m%d_%H%M%S)"
GPU=7 COLLECT_SAMPLES=32 EVAL_SAMPLES=8 MAX_NEW_TOKENS=128 HEAD_EPOCHS=10 \
  bash scripts/adaptive_length/run_pipeline.sh all
```

默认复用用户服务器路径：

| 项目 | 默认路径 |
|---|---|
| Python 环境 | `.venv-draftopd` |
| CUDA | `/home/jovyan/LLM/cuda-12.9` |
| Target | `/home/jovyan/LMM/lmm_model/Qwen3-4B` |
| Draft | `/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16` |
| 采集和训练数据 | `data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl` |
| 独立测试数据，若存在 | `data/opd_val_32.jsonl` |

可以覆盖 `PYTHON_BIN / CUDA_ROOT / MAIN_MODEL_PATH / DRAFT_MODEL_PATH / TRAIN_JSONL / VAL_JSONL`。
超出 `MAX_PROMPT=512` 的 prompt 会跳过，保留完整聊天模板。默认不开 thinking。

`all` **只运行冻结 Draft 的采集、长度头训练、评测，不自动重训 Draft**。
分步可运行 `collect / train-head / eval`，保持同一个 RESULT_ROOT。
已有标签目录拒绝覆盖，新采集必须换 RESULT_ROOT。增加样本量建议 `COLLECT_SAMPLES=128`。

## 第二轮：多长度 OPD 训练

```bash
export RESULT_ROOT="$PWD/runs/multilength_opd_$(date +%Y%m%d_%H%M%S)"
# DDP：每卡一份冻结 Target + 一份可训练 Draft，不是 FSDP。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
  TRAIN_STEPS=1000 SAVE_FREQ=250 LR=3e-4 MAX_NEW_TOKENS=256 \
  bash scripts/adaptive_length/run_pipeline.sh train-draft

# 新权重必须重新采集并训练长度头。
export DRAFT_MODEL_PATH="$RESULT_ROOT/draft/step_001000"
export RESULT_ROOT="$RESULT_ROOT/length_head"
CUDA_VISIBLE_DEVICES=7 NPROC=1 COLLECT_SAMPLES=128 \
  bash scripts/adaptive_length/run_pipeline.sh all
```

训练每次随机采样一个实际 K，执行预测和 Target 验证；接受前缀使用 forward KL，拒绝及后缀使用 reverse KL，按有效候选数归一化。
Target 在真实候选 token 诱导的状态上提供分布，不人工制造固定比例错误。
这是复用 Draft-OPD 原理的独立 greedy 参考训练器，**不等同于原 verl 的全部损失、采样、批处理及优化配置**。
不支持从断点恢复优化器状态，仅导出标准 `save_pretrained` Draft 权重。

训练仅使用按内容哈希划分的 80% prompt。每次更新后重建历史，防止沿用旧权重投影的 Draft KV；
这会增加 prefill 成本，优先保证实验逻辑正确，不用于声称训练效率优于 verl。
8 卡版本固定每卡更新次数并通过 DDP 同步梯度，每卡 prompt/候选可不同。
先用单卡 `TRAIN_STEPS=10 TRAIN_SAMPLES=8` 验证服务器兼容性，再扩大。

标签采集也支持 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8`：每卡独立 B=1，输出各 rank 的 shard。
这不是 B=8 的吞吐评测。训练长度头要等采集进程全部完成。

## 输出和公平性

- `paired/labels_rank*.pt`：pivot、anchor、每个 K 的接受数与同步实测成本、prompt 哈希、模型路径与层号。
- `head.pt`：长度头、成本表、输入配置、验证指标、全部采集 prompt 哈希。
- `eval_v2/results_rank000.jsonl`：逐 prompt 的 AR、所有固定 K、动态 K 的速度、动作和逐 token 匹配结果。
- `eval_v2/summary_rank000.csv`：聚合 token / 聚合 decode 时间；prefill 单列，decode 包含长度头、选择、起草和验证。
- `draft/step_*/`：多长度训练导出的 Draft。

同一 prompt 的所有 anchor 按内容哈希分到同一训练/验证组，重复 prompt 不跨组。
长度头的成本表只由训练组估计；验证组用于选 checkpoint。
最终解码测试自动排除**全部采集 prompt**，包括长度头验证组，避免以验证集代替独立测试。
若无独立 VAL_JSONL，则从原数据的哈希保留组挑选未采集的 prompt。

配对采集在每个 prompt 首轮预热各宽度，随机动作顺序；默认重复计时 2 次取中位数。
成本使用训练样本的静态中位数，未按上下文/负载分桶，可通过 Python CLI `--costs` 提供独立成本表。
配对估计用于筛查可预测性；真正速度以独立解码测试为准，不承诺一定超过最佳固定 K。
默认记录 greedy 输出匹配情况但继续评测；设置 STRICT_MATCH=1 可在不匹配时中止。
仅 B=1，temperature=0；尚不能推断高并发收益。
模型身份目前以路径和层配置记录，勿原地更换同路径权重后复用旧标签/长度头。

## 消融与验证

```bash
source .venv-draftopd/bin/activate
export PYTHONPATH="$PWD/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
python -m adaptive_length.pipeline train-head --labels "$RESULT_ROOT/paired" \
  --output "$RESULT_ROOT/head_anchor.pt" --features anchor
python -m adaptive_length.pipeline train-head --labels "$RESULT_ROOT/paired" \
  --output "$RESULT_ROOT/head_pivot.pt" --features pivot
python -m pytest -q diffusion/dflash/tests/test_adaptive_length.py
```

CPU 测试使用小型随机 Qwen3/DFlash，覆盖真实缓存回退、全接受/拒绝、变长 greedy 一致性、特征因果位置、
长度收益和成本选择、输出预算、三种输入消融、checkpoint、按 prompt 分组以及 OPD 梯度方向。
本仓库 DFlash 基于 Transformers 4.x，测试环境 `transformers==4.57.1`；5.x 的 cache API 有不兼容变更。

## Evaluation v2: shared tail clipping

The launcher now defaults to `STRICT_MATCH=0`: token matches remain recorded but
never stop an exploratory benchmark. Set `STRICT_MATCH=1` for explicit strict checks.
All fixed/adaptive policies first choose a nominal K and then run
`actual_K=min(K, remaining_output_tokens-1)`. AR is used only for the last single
token. Adaptive selection does not pretend to have acceptance/cost labels for
untrained tail widths. Both requested and actual widths are saved.

Fixed baselines automatically include the checkpoint's `block_size-1` (when in
[1,32]), e.g. native `fixed_15` for a b16 model. Existing six-action heads work
unchanged; no recollection or head retraining is required to add this baseline.

Keep the existing RESULT_ROOT containing head.pt and rerun `eval`. New outputs
are under `eval_v2/`, preserving old `eval/` results. `HEAD_PATH` and `EVAL_OUTPUT`
can override those paths. The launcher also accepts a custom `PYTHON_BIN` to reuse
a virtual environment in the parent server checkout.

`summary_rank000.csv` adds speculative rounds, AR steps, tail rounds, mean accepted
candidates, mean committed tokens (including anchor), and non-tail acceptance.
`rounds_rank000.jsonl` contains the per-round counts, requested K, actual K and
remaining budget. Reported acceptance excludes tokens beyond EOS; raw acceptance
is retained in per-round data. The non-tail metric excludes budget-clipped rounds.

`PROFILE_ROUNDS=1` enables optional GPU event totals `head_ms/draft_ms/verify_ms/ar_ms`.
Events resolve after the final decode sync; no extra per-stage synchronization is
inserted. CUDA event spans are diagnostic timeline measurements, not pure kernel
compute time. Total decode includes host control, timing and accounting overhead.
Use `PROFILE_ROUNDS=0` (default) for primary speed comparisons; stage columns are
blank in that mode. Compare profiling runs only against other profiling runs.
Do not combine v1 and v2 timing results into one comparison.

## Main experiment: fixed-vs-mixed OPD training (no length head)

Run `bash scripts/adaptive_length/compare_training.sh all` with a fresh RESULT_ROOT.
The script runs two sequential training arms from the same checkpoint, then evaluates
**initial / fixed-trained / mixed-trained** at all fixed inference widths, including
native `block_size-1`. It does not collect length-head labels, train a head or run
an adaptive policy. Both arms use the same FP32 trainable parameters/AdamW states,
BF16 CUDA autocast, loss, LR, seed, prompt split and per-rank prompt schedule.

The fixed width is read from the original config (b16 usually means 15 candidates).
Mixed widths are `{4,8,12,16,24,32}`. Default `CANDIDATE_TOKENS_PER_UPDATE=240`
means **240 supervised candidate positions per rank per optimizer update** in both
arms. Variable numbers of blocks are accumulated with loss weight `K/240`.
The mixed plan samples feasible widths until exactly 240 positions are filled;
remaining-budget constraints affect its sampling histogram, which is logged.
DDP syncs gradients only on the last microbatch, with buffer broadcasts disabled
so unequal numbers of microbatches per rank do not create collective mismatches.

Default is 200 updates, 256 training prompts before rank sharding and 64 held-out
prompts for evaluation, max output 256. For 8 ranks, total supervised positions
per arm are `200 * 240 * 8 = 384000`. This matches supervision and update budgets,
**not FLOPs or wall time**. Actual compute time and width histograms are recorded.
Each update uses the same prompt in both arms; draft-induced prefixes and the
number of anchors can differ. EOS/output-limit restarts continue on that prompt.

Single-GPU smoke (same server model/data paths and parent venv are supported):

```bash
export RESULT_ROOT="$PWD/runs/fixed_vs_mixed_smoke_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=7 NPROC=1 TRAIN_STEPS=2 TRAIN_SAMPLES=8 \
EVAL_SAMPLES=4 MAX_NEW_TOKENS=64 \
  bash scripts/adaptive_length/compare_training.sh all
```

After smoke passes, use a fresh RESULT_ROOT and e.g.
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 TRAIN_STEPS=200 TRAIN_SAMPLES=256
EVAL_GPU=7 EVAL_SAMPLES=64 MAX_NEW_TOKENS=256`.
Train arms run sequentially, each using all 8 cards; evaluation is single-GPU B=1.
This is the standalone reference OPD trainer, not the original verl/FSDP stack.
Recomputing prefixes each microbatch prioritizes correctness over training speed.

`comparison.csv` combines all three models × fixed inference widths, including
acceptance/throughput. `budget_check.json` checks candidate counts, steps, prompt
schedule, initialization and optimizer controls; approximate GPU-hours are also
reported. Per-rank training JSON/JSONL and final `save_pretrained` checkpoints are
kept in `fixed/draft/` and `mixed/draft/`. Training overwrites are rejected; optimizer
resume is not implemented. `eval`/`summarize` stages reuse an existing comparison
root and its stored step count. Evaluation always uses the data file's content-hash
holdout, so the old length-head validation file does not limit testing to 32 prompts.

## Native verl OPD experiment

For original verl/FSDP/SGLang OPD with mixed **response training** widths and
native rejection replay, see [native OPD notes](../../../scripts/adaptive_length/NATIVE_OPD.md).
The launcher is `scripts/adaptive_length/native_opd.sh fixed|mixed`. This is distinct
from the standalone reference training comparison above. Rollout stays native-width;
this extension does not claim fully multi-width on-policy rollout.
