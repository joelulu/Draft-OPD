# Qwen3-4B：起草前长度预测（第一阶段）

本入口完成冻结 Target/Draft 的标签采集、长度头训练和动态宽度解码评测。
原有 verl/SGLang OPD 训练入口不变。本阶段**没有重训多长度 Draft，也没有跨块 Draft hidden 传递**。
先回答“现有 Draft 的合适长度能否从起草前特征预测”，再决定是否投入多长度 OPD 训练。

## 输入与预测目标

- `K ∈ {4,8,12,16,24,32}` 是候选数，**不含 anchor**；实际 Draft 输入宽度为 `K+1`。
- pivot：最后一个已经过 Target 的位置，取现有 `target_layer_ids` 多层 hidden，复用 Draft 的 `fc + hidden_norm`。
- anchor：该 pivot 预测出的 token，仅查 Target embedding，**不读取 anchor 自己的 Target hidden**。
- 两路各 `LayerNorm → Linear(128) → SiLU`，拼接后 `256 → 128 → 6`。
- 6 个输出分别回归各实际长度下的连续接受候选数，限制在 `[0,K]`；不强制不同 K 的预测单调。
- 推理选 `(预测接受数+1)/(实测 Draft+Target 验证耗时)` 最大的 K。成本表取训练 prompt 的中位数。
- 支持 `both/anchor/pivot` 消融；CPU 可以训练长度头。

标签在**同一前缀、同一 anchor**上分别实际起草、验证每个 K，绝不由一次 32 的预测截断生成其他长度标签。
每个动作从相同 cache 起点计时；首轮先预热各宽度；顺序随机，重复计时取中位数。
轨迹随机选择一个 K 推进，提交的都是 Target 验证后的 token。

## 服务器一键运行

```bash
cd ~/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/Draft-OPD
git fetch origin
git switch research/qwen3-4b-adaptive-length
git pull --ff-only origin research/qwen3-4b-adaptive-length

# 首轮小规模检查；目前是单 GPU、B=1 的参考实现。
export RESULT_ROOT="$PWD/runs/adaptive_length_smoke_$(date +%Y%m%d_%H%M%S)"
GPU=7 COLLECT_SAMPLES=16 EVAL_SAMPLES=8 MAX_NEW_TOKENS=128 HEAD_EPOCHS=10 \
  bash diffusion/dflash/adaptive_length/run_pipeline.sh all
```

默认复用：

- `.venv-draftopd`
- CUDA `/home/jovyan/LLM/cuda-12.9`
- Target `/home/jovyan/LMM/lmm_model/Qwen3-4B`
- Draft `/home/jovyan/LMM/lmm_model/Qwen3-4B-DFlash-b16`
- 训练数据 `data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl`
- 测试数据优先 `data/opd_val_32.jsonl`；不存在时使用训练文件中采集区间之后的 prompt。

可通过 `MAIN_MODEL_PATH / DRAFT_MODEL_PATH / TRAIN_JSONL / VAL_JSONL / PYTHON_BIN` 覆盖路径。
超出 `MAX_PROMPT` 的输入跳过，不截坏聊天模板。采集输出每个 prompt 一个 shard；新任务使用新目录，防止覆盖。
默认采集 128 条、评测 32 条、输出上限 256。实际采集数量可能因过长 prompt 被过滤而减少。

完整小实验：

```bash
export RESULT_ROOT="$PWD/runs/adaptive_length_$(date +%Y%m%d_%H%M%S)"
GPU=7 COLLECT_SAMPLES=128 EVAL_SAMPLES=32 \
  bash diffusion/dflash/adaptive_length/run_pipeline.sh all
```

分步执行时保持同一个 `RESULT_ROOT`，使用 `collect / train / eval` 替代 `all`。
长度头消融可复用标签：

```bash
export PYTHONPATH="$PWD/diffusion/dflash${PYTHONPATH:+:$PYTHONPATH}"
source .venv-draftopd/bin/activate
python -m adaptive_length.cli train --input "$RESULT_ROOT/paired" \
  --output "$RESULT_ROOT/head_anchor.pt" --features anchor
python -m adaptive_length.cli train --input "$RESULT_ROOT/paired" \
  --output "$RESULT_ROOT/head_pivot.pt" --features pivot
```

## 输出与判读

- `paired/metadata.json`、`manifest.json`、`prompt_*.pt`：模型信息、特征、6 种宽度的接受数和耗时。
- `head.pt`、`head.metrics.json`：最佳验证 loss 的长度头、训练成本表和固定长度基线。
- `eval/results.jsonl`、`eval/summary.json`：AR、训练集选出的固定 K、动态 K，以及各个固定 K 的实测结果。

标签按 prompt 内容哈希分训练/验证，重复问题不会跨组。最终解码测试拒绝使用采集阶段出现的 prompt。
`paired_estimated_tok_s` 是配对标签上的估计，`hindsight_estimated_tok_s` 使用事后答案，是诊断参考而非可部署策略。
实际效果以 `eval/summary.json` 为准：聚合 token/聚合 decode 时间，计入长度头、选择同步、起草与验证开销；prefill 不计入。
每个方法与 greedy AR 逐 token 对照，任何不匹配都会在保存结果后报错。EOS 与输出预算统一处理。

成本表目前是 B=1、同硬件/同模型的静态中位数，没有按上下文长度分桶。
因此尚不能据此推断高并发收益，也不能将实际按不同 K 前向视为已经完成多长度训练。
更换 Draft 权重后须重新采集和训练长度头。checkpoint 记录路径和层配置，不对大权重文件计算校验和；勿原地替换同路径权重后复用旧头。

## CPU 验证

在已安装 torch、transformers==4.57.1、datasets、pytest 的环境中：

```bash
python -m pytest -q diffusion/dflash/tests/test_adaptive_length.py
```

覆盖长度收益/成本选择、预算约束、三个输入消融、checkpoint 恢复、按 prompt 分组，以及小型真实 Qwen3/DFlash 的 cache 回滚和变长 greedy 一致性（包括全接受）。
