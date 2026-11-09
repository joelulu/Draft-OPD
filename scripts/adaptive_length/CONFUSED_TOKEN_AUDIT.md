# fixed 困惑 token 定位与理想修正

运行 `RESULT_ROOT=已有实验目录 EVAL_GPU=7 bash scripts/adaptive_length/run_confused_token_audit.sh`。
默认使用本地 DeepSpec-main/eval_datasets/gsm8k.jsonl 的前16条不同问题，
fixed 导出模型，K16 原始轨迹，生成上限512，K8只辅助记录。无需训练或下载数据。

三档概率差0.1/0.2/0.4；后面任意一个位置的top1概率严格大于0.8；不要求紧邻或连续。
每块选择最早满足位置（1～15）；第16位没有后续位置，不能被选中。
metadata 中 rule_version=any_later_top1_gt_0.8，用于区分旧规则。
同位置替换为原draft前缀条件下target的greedy token，然后重新验证；其余draft token不变。
全部干预丢弃，只推进原始K16块。另有直接定位真实首错并修正一次的参照。
对任何一种干预在提交范围内包含EOS的状态，全部方法配对排除；详细JSON保留raw接受数。
剩余不足17输出token时走AR，不计入block统计。末尾AR token数见prompt_stats.csv。

输出 summary.csv/json：精确率、全首错召回率、可覆盖首错召回率、误报/漏检数量，
以及修正前后平均接受数。selections.csv逐档逐块；rounds.jsonl含K8/K16 top2编号和概率、
选中位置、替换token、重新验证接受数；prompts.json保存实际问题。

这是离线理想修正收益检查，使用target答案，不是已实现的draft修正策略或吞吐评测。

## 跨 K 实际修正（免训练）

在原命令末尾加 `--cross-k-repair`。仍用 fixed 和相同16条GSM8K，
同一状态计算K8/K16/K24；用K16概率规则选位置，再独立测试四种方法：

- vote：可用档位top1投票，平票优先保留K16，之后按token编号。
- mean_probability：在可用档位top2的并集内，选择完整词表概率的算术平均最大者。
- contrast16_8：在K16 top2内最大化2log(p16)-log(p8)，第9～15位保持原token。
- contrast24_16：在K24 top2内最大化2log(p24)-log(p16)。

对比系数预设alpha=1，不根据16条测试数据调参；大K不保证更正确。
这些是待检验规则，不是LoopCD原算法的复现。位置超过8时投票/平均只用K16与K24。
候选token在其他档位的概率是实际完整softmax概率，不把top2重新归一化。
四种替换均不读取target答案；target只在选定后评价，理想修正另作参照。

输出cross_summary.csv/json，每档阈值×每种方法，共12行：修改次数、
命中首错及真正修复数、改善/损害block数、原始与修正后接受数、净收益。
rounds.jsonl的cross/cross_diagnostics保留每次替换及候选跨K概率。
跨K表对任一方法/理想修正提交范围遇EOS的状态共同排除，保证配对；
故其样本数可能不同于旧summary表，按cross_summary自己的基线比较。
实验包含额外起草和离线多次验证，不报告部署吞吐量；原始K16轨迹不受修正影响。
