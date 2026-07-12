# PAgent

用可训练的小模型（Qwen3-4B LoRA）为冻结的大模型（GPT-OSS-20B）生成提示，在 DAPO-Math 上提升 Exact Match（EM）。只训练小模型，solver 全程冻结。

当前最佳本地权重：`checkpoints/blind_ff_sft_17k_merged`（Blind FF-SFT 17K full）。

---

## 1. 模型部署

### 角色与路径

| 角色 | 模型 | 本地路径 / 服务名 |
|------|------|-------------------|
| Solver（冻结） | GPT-OSS-20B | `/home/ivaning/models/gpt-oss-20b`，served-name `gpt-oss-20b` |
| Router 基座 | Qwen3-4B | `/home/ivaning/prompt-r1r/Prompt-R1/Qwen/Qwen3-4B` |
| 最佳 Router（merge） | Blind FF-SFT 17K | `checkpoints/blind_ff_sft_17k_merged`，served-name `qwen3-4b-blind-ff-17k` |
| Adapter（可再 merge） | 同上 LoRA | `checkpoints/blind_ff_sft_17k_adapter` |

默认端口（`config.py`）：

- OSS answerer：`8006–8009`（OpenAI-compatible `/v1`）
- Router：`8083`（离散 router 默认）或 eval 脚本常用 `8086`

### 启动 OSS（solver）

高吞吐（约 69GB/卡，`util=0.70`）：

```bash
bash scripts/serve_oss_4gpu.sh
```

省显存（约 36GB/卡，`util=0.36`，`max_model_len=8192`）：

```bash
bash scripts/serve_oss_compact.sh
# 默认 GPU 0/2/3 → :8006/:8008/:8009
```

停止 OSS：

```bash
pkill -f 'vllm.entrypoints.openai.api_server.*gpt-oss-20b'
```

检查：

```bash
for p in 8006 8007 8008 8009; do
  curl -sf http://127.0.0.1:$p/v1/models >/dev/null && echo "$p up" || echo "$p down"
done
```

### 启动 Router（Qwen3-4B merged）

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model checkpoints/blind_ff_sft_17k_merged \
  --port 8086 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 8192 \
  --served-model-name qwen3-4b-blind-ff-17k
```

### 数据准备

```bash
python run.py prepare --train-size 17000 --val-size 256
# 写出 data/train.jsonl, data/val.jsonl
```

---

## 2. 离散 Action Router（复现）

固定 `ACTION_SPACE`（6 个模板 hint），小模型只输出 action key。

### 2.1 Oracle labeling（穷举 6 actions）

```bash
# 需 OSS 已启动
bash scripts/run_label_2048_tmux.sh
# 或手动：
python run.py label \
  --data-file data/train.jsonl \
  --limit 2048 \
  --out-dir checkpoints/label_2048 \
  --workers 32 \
  --protocol native
```

**结果（2048 题）**

| 统计 | 数值 |
|------|------|
| rollouts | 12288（2048×6） |
| signal 题（可用于监督） | 759（37.1%） |
| all-correct / all-wrong | 797 / 492 |
| 标签文件 | `checkpoints/label_2048/labels.jsonl` |

### 2.2 SFT / DPO / GRPO

```bash
# SFT（离散 action）
python run.py sft-train \
  --labels-file checkpoints/label_2048/labels.jsonl \
  --out-dir checkpoints/sft_adapter \
  --epochs 3 --batch-size 8 --lr 2e-5
python run.py sft-merge

# DPO
python run.py dpo-build-pairs   # → checkpoints/label_2048/dpo_pairs.jsonl
python run.py dpo-train --out-dir checkpoints/dpo_adapter
python run.py dpo-merge

# GRPO（端到端在线）
python run.py train --mode quick --gpu 1
python run.py merge
```

Serve merged router 后评测（离散 mode=`router`）：

```bash
python eval_repeat.py \
  --repeats 4 --limit 128 \
  --router-mode router \
  --router-url http://127.0.0.1:8083/v1 \
  --router-model qwen3-4b-router \
  --out-root checkpoints/eval_native_128_repeat
```

### 2.3 主要结果

| 方法 | 评测 | Router EM | vs live baseline | 结论 |
|------|------|-----------|------------------|------|
| GRPO action router（论文表 shared baseline） | 128×4 | **44.34%** | **+1.76pp**（相对 42.58%） | 提升很小 |
| SFT on oracle actions | 128 单次 | 42.97% | **−1.56pp** | 失败 |
| DPO | 128 单次 | 43.75% | **−0.78pp** | 失败 |

旧 merged 权重已上传 HF（本地已删）：  
`ivaning0919/pagent-router-{grpo,sft,dpo}-merged`

---

## 3. Free-form（模版 hint 监督）

把离散 oracle action 转成对应的**固定模板文本 hint**，再 SFT 小模型直接生成 free-form hint。

```bash
# 从 action labels 构建 ff labels
python run.py ff-sft-build \
  --labels-file checkpoints/label_2048/labels.jsonl \
  --out-file checkpoints/label_2048/ff_labels.jsonl
# 759 条；其中 410 条为空 hint（baseline 最优）

# SFT + merge
python run.py ff-sft-train \
  --labels-file checkpoints/label_2048/ff_labels.jsonl \
  --out-dir checkpoints/ff_sft_adapter \
  --epochs 3 --batch-size 8 --lr 2e-5
python run.py ff-sft-merge \
  --adapter-dir checkpoints/ff_sft_adapter \
  --merged-dir checkpoints/ff_sft_merged

# 128×4 eval（需先 serve merged 模型）
python eval_repeat.py \
  --repeats 4 --limit 128 \
  --router-mode ff_router \
  --router-url http://127.0.0.1:8086/v1 \
  --router-model qwen3-4b-ff \
  --out-root checkpoints/eval_ff_sft_128_repeat \
  --eval-workers 32
```

**结果（128×4，native）**

| Metric | Mean | Std |
|--------|------|-----|
| Live baseline EM | 45.31% | ±1.2pp（本 session） |
| FF SFT router EM | **50.20%** | ±1.4pp |
| Paired Δ | **+4.88pp** | ±2.2pp |
| 相对 shared baseline 42.58% | **+7.62pp** | — |

McNemar（4 轮合计）：router-only 51 / baseline-only 26，**p=0.006**。  
结论：自由文本 hint 明显强于离散 action router。

HF：`ivaning0919/pagent-ff-sft-merged`

---

## 4. Blind Hint Labeling（核心数据管线）

Hint **只看题面、不看 gold**；仅当 baseline 错且带 hint 后变对（flip）才写入 SFT label。

流程：baseline → 对 baseline 错题采 K 条 blind hint → OSS 重试 → 保留 flip。

### 4.1 2048（小规模）

```bash
python run.py oracle-hint \
  --data-file data/train.jsonl \
  --limit 2048 \
  --out-dir checkpoints/blind_hint_2048 \
  --k 6 --hint-temp 0.8 \
  --workers 40 --protocol native
```

| 统计 | 数值 |
|------|------|
| 题数 | 2048 |
| baseline 错 | 620 |
| flip labels | **448** |
| 覆盖题数 | 227 |
| 产物 | `checkpoints/blind_hint_2048/oracle_labels.jsonl` |

### 4.2 17K（主数据）

```bash
bash scripts/run_blind_hint_17k_tmux.sh
# 或（OSS 已起）：
python run.py prepare --train-size 17000 --val-size 256
python run.py oracle-hint \
  --data-file data/train.jsonl \
  --limit 17000 \
  --out-dir checkpoints/blind_hint_17k \
  --k 6 --hint-temp 0.8 \
  --workers 40 --protocol native \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1
```

| 统计 | 数值 |
|------|------|
| 题数 | 17000 |
| baseline 错 | 5206 |
| flip labels | **3687** |
| 覆盖题数 | **1809** |
| 产物 | `checkpoints/blind_hint_17k/oracle_labels.jsonl` |

### 4.3 按题去重（消融用）

多 candidate 题对每条 hint 再测 3 次，保留 flip rate 最高的一条：

```bash
python run.py ff-dedup-blind \
  --labels-file checkpoints/blind_hint_17k/oracle_labels.jsonl \
  --out-file checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl \
  --repeats 3 --workers 32
# → 1809 条（一题一条）
```

---

## 5. Blind Free-form SFT（主结果线）

用 blind flip labels 做 free-form SFT：小模型生成自然语言 hint → 冻结 OSS 作答。

### 5.1 训练（17K full，当前最佳）

```bash
bash scripts/run_blind_ff_sft_17k_tmux.sh
# 等价于：
python run.py ff-sft-train \
  --labels-file checkpoints/blind_hint_17k/oracle_labels.jsonl \
  --out-dir checkpoints/blind_ff_sft_17k_adapter \
  --gpu 2 --epochs 3 --batch-size 4 --lr 2e-5
python run.py ff-sft-merge \
  --adapter-dir checkpoints/blind_ff_sft_17k_adapter \
  --merged-dir checkpoints/blind_ff_sft_17k_merged
```

Dedup 变体：

```bash
bash scripts/run_blind_ff_sft_17k_dedup_tmux.sh
```

### 5.2 OOD 评测（128×4 paired）

```bash
bash scripts/run_blind_ff_sft_17k_eval_tmux.sh
# 或：
python eval_repeat.py \
  --repeats 4 --limit 128 \
  --router-mode ff_router \
  --router-url http://127.0.0.1:8086/v1 \
  --router-model qwen3-4b-blind-ff-17k \
  --out-root checkpoints/eval_blind_ff_sft_17k_128_repeat \
  --eval-workers 32
```

**主结果（native，128×4）**

| 模型 | Labels | Router EM | Paired Δ | 4 轮全正 |
|------|--------|-----------|----------|----------|
| Blind FF SFT v1 (2048) | 448 | 48.44% ±2.2 | +4.49pp | Yes |
| Blind FF SFT v2 | — | 51.17% ±1.7 | +9.18pp | Yes |
| Blind FF SFT v3 | — | 51.56% ±2.5 | +8.59pp | Yes |
| Blind FF GRPO（v1 init） | RL | 46.88% ±2.3 | +2.93pp | **No** |
| **Blind FF SFT 17K full** | **3687** | **51.56% ±1.5** | **+9.57pp** | **Yes** |
| Blind FF SFT 17K dedup | 1809 | 48.44% ±2.0 | +5.27pp | Yes |

相对 **shared live baseline 42.58%**（论文对比表）：

| Method | EM | Δ |
|--------|-----|---|
| Live baseline | 42.58% | — |
| Action router | 44.34% | +1.76pp |
| FF SFT (template) | 50.20% | +7.62pp |
| **Blind 17K full** | **51.56%** | **+8.98pp** |
| Blind 17K dedup | 48.44% | +5.86pp |

17K full McNemar（4 轮合计）：router-only 68 / baseline-only 19，**p&lt;0.001**。

产物：`checkpoints/eval_blind_ff_sft_17k_128_repeat/aggregate.json`

### 5.3 同分布评测（训练 label 覆盖的 1809 题）

题集：`oracle_labels_dedup.jsonl` 的 unique question id（1809）。  
每个模型跑：`live_baseline` / `ff_router` / `oracle_hint`（重放训练 label hint）。

```bash
bash scripts/run_ff_idist_eval_tmux.sh
# 或单模型：
python eval_idist.py \
  --labels-file checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl \
  --out-dir checkpoints/eval_idist_blind_ff_17k_full \
  --router-model qwen3-4b-blind-ff-17k \
  --router-url http://127.0.0.1:8086/v1 \
  --workers 32
```

**结果（1809 题）**

| 模型 | Baseline | Router | Oracle hint | Router−Baseline | Router−Oracle |
|------|----------|--------|-------------|-----------------|---------------|
| **17K full** | 30.68% | **55.94%** | 61.58% | **+25.26pp** | −5.64pp |
| 17K dedup | 29.96% | 53.51% | 61.47% | +23.55pp | −7.96pp |

说明：这 1809 题来自 blind labeling 时 baseline 做错的题，故 baseline EM 低于全量 ~42%。  
Full 同分布仍优于 Dedup（+2.4pp），且离 oracle 上界约 5.6pp。

产物：
- `checkpoints/eval_idist_blind_ff_17k_full/summary.json`
- `checkpoints/eval_idist_blind_ff_17k_dedup/summary.json`

---

## 6. 数据 / 训练消融与负结果

| 实验 | 结果 | 结论 |
|------|------|------|
| 离散 action SFT / DPO | −1.6pp / −0.8pp | 固定模板 action space 不够用 |
| 离散 GRPO action router | +1.76pp（相对 shared baseline） | 有信号但天花板低 |
| Template FF SFT | +4.9pp paired / +7.6pp vs shared | 自由文本 hint 有效 |
| Blind 2048 → 17K 扩大数据 | OOD Δ ~+4.5 → **+9.6pp** | **数据规模关键** |
| 17K 按题去重再 SFT | OOD −3.1pp vs full；idist −2.4pp | **多 hint/题是增强**，去重有害 |
| Blind GRPO（SFT 后 RL） | +2.9pp，有一轮为负 | **暂不建议 RL** |
| 同分布 vs OOD（17K full） | 55.94% vs 51.56% | 有记忆，但 OOD 仍显著为正 |

**当前建议**：主模型用 **Blind FF-SFT 17K full**；dedup 与 GRPO 路线可停。

---

## 常用命令速查

```bash
# OSS
bash scripts/serve_oss_compact.sh          # 省显存
bash scripts/serve_oss_4gpu.sh             # 高吞吐

# Blind label → SFT → eval
bash scripts/run_blind_hint_17k_tmux.sh
bash scripts/run_blind_ff_sft_17k_tmux.sh
bash scripts/run_blind_ff_sft_17k_eval_tmux.sh
bash scripts/run_ff_idist_eval_tmux.sh

# Dedup 消融
python run.py ff-dedup-blind --repeats 3
bash scripts/run_blind_ff_sft_17k_dedup_tmux.sh
bash scripts/run_blind_ff_sft_17k_dedup_eval_tmux.sh
```
