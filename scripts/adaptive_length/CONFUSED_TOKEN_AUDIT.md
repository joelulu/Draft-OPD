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
