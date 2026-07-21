# Residual HintFlow

主线改为 baseline-first residual agent：turn 0 先生成与 baseline 完全相同的
完整候选；后续候选只能通过保守 selector 替换 incumbent，最终不再默认采用最后
一次回复。每题最多 7 次 OSS 调用。

## GPU 约束

本机命令只允许物理 GPU3。训练脚本会覆盖
`CUDA_VISIBLE_DEVICES=3`；serve 脚本会校验 API 进程树对应的 GPU UUID。

## 固定数据划分

```bash
python HintFlow/make_residual_splits.py --dev-size 512
```

- `checkpoints/residual_splits/train.jsonl`：采数/训练
- `checkpoints/residual_splits/dev.jsonl`：迭代与 gate
- `checkpoints/residual_splits/final.jsonl`：最终里程碑，禁止日常调参
- `manifest.json`：按规范化 problem hash 固定且默认不可覆盖

## 候选上界

```bash
python HintFlow/eval_residual.py \
  --data-file checkpoints/residual_splits/dev.jsonl \
  --limit 512 --workers 4 --max-solver-calls 7 \
  --policy-mode fixed --selector-mode keep \
  --out-dir checkpoints/eval_residual_fixed_dev
```

先检查 `any_candidate_oracle_em - baseline_em >= 0.08`。候选池不过线时不要
训练 selector。

## Counterfactual turn feedback

```bash
python HintFlow/collect_residual_feedback.py \
  --data-file checkpoints/residual_splits/train.jsonl \
  --limit 512 --samples-per-action 2 \
  --out checkpoints/residual_feedback_collection/train_turns.jsonl

python HintFlow/collect_residual_feedback.py \
  --data-file checkpoints/residual_splits/dev.jsonl \
  --limit 128 --samples-per-action 2 \
  --out checkpoints/residual_feedback_collection/dev_turns.jsonl

python HintFlow/export_residual_feedback.py \
  --counterfactual checkpoints/residual_feedback_collection/train_turns.jsonl,checkpoints/residual_feedback_collection/dev_turns.jsonl \
  --legacy-trees checkpoints/hintflow_trees_2k/trees.jsonl,checkpoints/hintflow_trees_v2/trees.jsonl \
  --out-dir checkpoints/residual_feedback_dataset
```

导出任务相互独立：

- `correctness`：CORRECT / INCORRECT
- `selection`：KEEP / REPLACE（tie 为低权重 KEEP）
- `action`：STOP / VERIFY_REPAIR / ALTERNATE_SOLVE / TARGETED_CHECK
- `diagnosis`：不泄露 gold 的错误类型、证据和 repair hint

Action 标签使用同 state、matched seed 的一步反事实：

`Q(s,a)=E[max(EM(incumbent), EM(candidate_a))]`，
再减去非 STOP 调用成本。

## 训练与闭环

```bash
DATA_DIR=checkpoints/residual_feedback_dataset \
bash scripts/train_residual_feedback_gpu3.sh

# 或跑一个小型端到端 smoke
bash scripts/run_residual_smoke_gpu3.sh
```

每轮 checkpoint 使用不可变目录：
`checkpoints/residual_cycles/<cycle>/`。完整 gate：

```bash
CYCLE=my_cycle DATA_DIR=checkpoints/residual_feedback_dataset \
bash scripts/run_residual_cycle_gpu3.sh
```

Promotion 必须同时通过：

- candidate oracle headroom ≥ 8pp
- strict selector accuracy ≥ 70%
- correctness balanced accuracy ≥ 70%
- free-generation valid rate ≥ 90%
- baseline correct retention ≥ 98%
- dev paired ΔEM ≥ 5pp，bootstrap 95% CI 下界 > 0
- 请求 error rate = 0

拒绝或异常时恢复上一 promoted model；final set 只有显式设置
`FINAL_EVAL=1` 才运行。

## 当前 promoted 结果

Promoted model：`checkpoints/residual_cycles/residual_v3/merged`。

- dev64：baseline 59.38% → residual 65.63%，**+6.25pp**；
  95% bootstrap CI `[+1.56,+12.50]pp`；4 recover / 0 harm。
- frozen candidate dev34：baseline 73.53% → learned selector 82.35%，
  **+8.82pp**；3 recover / 0 harm。
- final128 公平 8k baseline：55.47% → residual 56.25%，
  **+0.78pp**；1 recover / 0 harm，baseline-correct retention 100%。
- 4k-token ablation：43.75% → 46.88%，+3.13pp；该结果不能与历史
  8k baseline 直接横比。
- 公平 final128 平均 2.73 次 OSS 调用。

开发集通过完整 promotion gate；公平 final128 仍保持 0 harm，但 candidate oracle
headroom 仅 1.56pp，说明强 8k baseline 下的主要瓶颈已从 selector 转为候选生成。
下一轮应优先设计能解决 baseline-wrong 难题的分支，而不是放松 selector 的保守
替换门槛。
