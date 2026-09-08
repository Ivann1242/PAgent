# PAgent

用可训练的小模型（Qwen3-4B LoRA）为冻结的大模型生成提示，在 DAPO-Math 上提升 Exact Match（EM）。只训练小模型，solver 全程冻结。

两条时代：

1. **OSS-20B + Blind FF**（§1–6）：冻结 GPT-OSS-20B，4B 生成一条 free-form hint。最佳权重 `checkpoints/blind_ff_sft_17k_merged`。强 solver 上 KEEP/REPLACE 很稳，但绝对 EM 已经很高，协议空间小。
2. **Qwen3-14B + HintFlow**（§7–8）：solver 换成 Qwen3-14B（nothink），小模型换成 `qwen3-4b-blind-ff-grpo-clean-iid-vp-step200`。当前最佳协议是 **Draft-Plan-Select v4**：holdout512 上相对自己的裸 draft **+4.9pp**，配对略强于 HF1，明显强于固定 5 步的 HF5。

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

**OSS-20B 线建议**：主模型用 **Blind FF-SFT 17K full**；dedup 与早期 GRPO 可停。14B 协议线见 §7–8。

---

## 7. 换到 Qwen3-14B：数据、训练、以及「hint 本身够不够」

OSS-20B 的 holdout512 @20k 已经是 78.7% → HF1 81.6%（rec 16 / harm 1）。MATH500 同样只有 +1.2pp。强 solver 上，hint 的边际收益被压扁。后面把 solver 换成 **Qwen3-14B nothink @8k**（holdout512 裸解约 53.5%），把协议空间重新打开。

### 7.1 14B 上的 Blind label 必须重解析

对 14B 重跑 oracle labeling 后，`### Final Answer` 一类格式让旧抽取器漏判。清洗结果：

| | 行数 | 题数 |
|--|------|------|
| 原始 14B labels | 13145 | 5020 |
| 丢掉「baseline 其实已经对」 | 5357 | — |
| 丢掉「hint 重解析并不对」 | 87 | — |
| **clean**（baseline=0 且 hinted=1） | **7701** | **3262** |
| 其中真难题 / 污染题 | — | 3288 / 1732 |

产物：`checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean.jsonl`。  
**发现**：不修抽取器会把大量「格式 recover」当成 hint flip，后面 SFT/GRPO 会学到噪声。

### 7.2 学到的 hint ≈ 通用 CoT，远低于 oracle

在 clean-128（这些题的 14B baseline 按定义是 0，oracle hint 是 100%）：

| Challenger | EM |
|------------|-----|
| 学到的 router hint | 43.0% |
| 固定 CoT（「Think step by step…」） | **43.8%** |
| Oracle 训练 hint | **100%** |

73 道「oracle 对、router 错」里，**69 道标签是 `similar_hint_but_still_fail`**：router 和 oracle 说的是同一类方法，Jaccard 经常 0.4–0.6，14B 还是算错。真正缺公式 / 缺数值结构的很少。

OOD 也站在同一边。AIME24 @8k、强制 REPLACE：

| Challenger | EM vs 裸 36.7% |
|------------|----------------|
| GRPO Blind FF | 36.7%（0） |
| 固定 CoT | **43.3%（+6.7pp）** |

**发现**：当前 4B 并没有学到「那条能翻转 14B 的具体 hint」。它学到的是更稳的解题口气。oracle 上界还在，但那是题级、几乎不可泛化的提示，不是 4B 现在能复制的对象。协议必须假设 **hint 经常只是另一份噪声求解**，收益来自「怎么比、怎么选、怎么少害」，不是来自一条神 hint。

### 7.3 Virtual-positive GRPO

在 clean IID 上做 virtual-positive GRPO（`checkpoints/blind_ff_grpo_clean_iid_vp_adapter`，评测常用 step200）。思路：组内全错时仍给一条虚拟正样本，避免「全 0 advantage」。idist train500（baseline=0）：router **48.2%** / oracle 100%。这个权重是后面所有 14B 协议共用的 4B，不是又一个更强 hinter。

同一套 4B，换协议比换权重更敏感：14B AIME24 @8k 强制 REPLACE 时 Blind FF 不涨、固定 CoT 涨；HF1 保守 KEEP/REPLACE 在 OSS AIME24 @20k（4 种子）只剩均值 +2.5pp、总 rec 4 / harm 1。**发现**：权重把 4B 定在「会写 hint」这一档之后，增益由比选规则决定。

### 7.4 旧协议在 OSS 上其实已经说明了「少害」

这些是换 14B 之前的对照，用来解释后面为什么 Draft 盯 harm：

| 协议 | 集合 | Δ | rec / harm | 要点 |
|------|------|---|------------|------|
| **HF1** KEEP/REPLACE | OSS idist 1809 @8k | **+14.5pp** | 267 / **5** | 难题集上 selector 极稳 |
| **HF1** | OSS holdout512 @20k | +2.9pp | 16 / 1 | 强 baseline 上增益变小 |
| **Gate**（hint→4B 决定用不用→一次求解） | OSS idist 配对 989 | +3.3pp | 181 / **148** | 没有 incumbent，害和救几乎打平 |
| **Residual V3** | OSS final128 @8k | +0.8pp | 1 / **0** | 0 harm 做得到；候选上界只剩 1.6pp |
| **HF1** GRPO-VP step40 | OSS AIME24 @20k ×4 | +2.5pp | 4 / 1 | 强 OOD 上几乎不 REPLACE |
| **HF1** | OSS MATH500 @20k | +1.2pp | 7 / 1 | 同样被强 baseline 压扁 |

Residual 的结论仍然成立：**8k 强 baseline 下，瓶颈从 selector 换成了「怎么生出另一条对的解答」**。Gate 说明「先决定用不用 hint、再只解一次」留不住已有正确答案。

---

## 8. HintFlow 协议族（Qwen3-14B + GRPO 4B）

共同设置（除非另注）：solver `qwen3-14b` @ 8k，小模型 `qwen3-4b-blind-ff-grpo-clean-iid-vp-step200`，数据 `data/dapo_holdout_512.jsonl`，temperature 0。

### 8.1 动机：协议税

HF1 是一次 Blind FF + 保守 KEEP/REPLACE。HF5 把同一套路固定成 5 步增量推理。轨迹上很清楚：**固定步数、固定话轮、保守 selector 会把门关死**，算力用在重复写已经写过的答案上。

HF5 selector 按**步**计的 BASELINE 率：

| 设置 | BASELINE | HINTED | holdout512 EM |
|------|----------|--------|---------------|
| iid / holdout 2k×5 | 65–67% | 33–35% | 48.6%（2k×5） |
| GRPO router + 4B-bare selector | ~81% | ~19% | — |
| 同 router 兼 selector，8k×5 | **96.5%** | **3.5%** | **44.2%** |

末步尤其极端（约 90–94% 留 BASELINE）：两边已经写出同一条 `Answer:`，selector 按平局不换。8k×5 几乎等于「付了 5 倍生成，协议却拒绝干预」。

Interact（scout / plan / alt_solve / critique / revise）EM 更高（holdout512 **58.2%**），但话轮固定、14B 调用更多。Draft 的目标是：**动态步数、draft 只用来规划，最终答案来自逐步重建**，算力仍落在「大约两份 14B 解答」这一档。

### 8.2 家族对照（holdout512，14B @ 8k）

| 协议 | 最终 EM | 相对自己的裸 14B | recover / harm | 14B 调用量级 |
|------|---------|------------------|----------------|--------------|
| 裸 14B（HF1 baseline） | 53.5% | — | — | 1×8k |
| **HF1** Blind FF + KEEP/REPLACE | 55.5% | +2.0pp | 50 / 40 | 2×8k |
| HF5 固定 5 步（2k×5） | 48.6% | 负 | — | 10×2k |
| HF5 固定 5 步（8k×5） | 44.2% | 负 | — | 10×8k |
| Interact 5-act | **58.2%** | +4.7pp | 57 / 33 | ~5 个话轮 |
| **Draft v4** | **56.8%** | **+4.9pp** | **48 / 23** | 1×8k + ~7.5×2k |

Draft 官方 512（失败题记 0）：裸 draft 52.0% → **56.8%**。和 HF1 成功题配对（506 题）：Draft **57.5%** vs HF1 **56.1%**（**+1.4pp**），Draft 独对 51、HF1 独对 44。

领先不大，但来源清楚：**不是多救，是少害**（harm 23 vs HF1 的 40）。Interact 的绝对 EM 仍更高，预算也更贵。

HF1 的 challenger EM（53.7%）几乎等于裸 baseline（53.5%）。协议赚到的 +2.0pp 几乎全是 selector：用对的那一边，而不是 hint 本身更强。Interact 的 incumbent 来源：scout 512、alt_solve 173、revise 157——后面两个话轮才是增益，scout 只是起盘。

### 8.3 Draft-Plan-Select v4（当前 pipeline）

代码：`HintFlow_draft/`。最终答案**只交出重建轨迹**，draft 不当整卷兜底。

```
题目
  ├─ 14B 裸解 ── draft（只给规划 + 末步抽答案）
  │                │
  │                ▼
  │         4B 动态规划（1–8 步，目标写方法不写取值）
  │                │
  ▼                ▼
context = 题目 + 已完成步骤
  每步：4B hint → 14B baseline / hinted → 4B 盲选 A/B
  末步：若三个抽出答案两两可分、draft 只与一边相同 → 覆盖 selector
出门 = 最后一步选中文本
```

规划之后，hint / 解题 / selector **都看不到 draft 正文**。

护栏（都是规则，不再加 4B 整卷裁判）：

1. **规划失败** → 单步双候选盲选，不是 hinted-only HF1。
2. **Planner 指令**：目标写成方法；禁止 `check n=1`、`Final Answer: 371`。
3. **末步 draft vote**：只比抽出的答案。中间步不触发。
4. **Stub + 泄漏不锁 baseline**：计划第一步仍是 few-shot 占位 `result-free executable goal`，且末步泄漏了 draft 的具体答案时，**不做 baseline 方向的 vote**，并把泄漏洗成 `<answer>`。完整多步计划不洗。
5. **不硬拒提前交卷**。中间步写出 `Final Answer` 只记日志。

### 8.4 迭代：什么留下、什么回退

64 题 keep 规则：EM 升且 harm 不增，或 EM 不降且 harm 降。对照同一 63 题成功子集。

| 版 | 改动 | 63 题 EM | rec / harm | 决定 |
|----|------|----------|------------|------|
| v1 | `plan_only` 逐步重建 | 55.6% | 6 / 3 | 基线 |
| v2 | 4B 出口比 draft vs 重建；硬拒提前交卷；规划失败改 hinted-only | 52.4%（−3.2pp） | 3 / 2 | **回退** |
| v3 | 末步答案级 draft vote；planner 禁止抄取值 | 57.1%（+1.6pp vs v1） | 5 / 1 | **keep** |
| v4 | stub+泄漏时跳过 baseline vote，并清洗末步泄漏 | **60.3%**（+3.2pp vs v3） | 7 / 1 | **keep** |

**v1 提升从哪来（6 个 recover）：** 约一半是规划失败后的双候选纠错（17346、17369）；其余是末步/中间步 hinted 纠错。另有格式 recover：draft 已写出正确答案，但写成 `Answer:` 抽不出来，重建写成 `Final Answer:` 才计分。

**v1 伤害从哪来（3 个 harm）：**

- **17301**：planner 把 draft 的不完整枚举抄进目标（「先查 n=1，再查 n=3」），后面每步都在执行残缺搜索。
- **17310**：末步 baseline=736 / hinted=371，draft 也是 371，selector 选了 736。
- **17314**：中间步几何判断错，之后两边一起错。末步没有分歧，vote 用不上。

**v2 为什么挂：**

- 4B 盲选两篇完整解答，答案不同的 10 题里救 5 错 5。重建已经对了也会被错误 draft 覆盖。
- 「非末步禁止 Final Answer」太狠：hint 采纳率 58%→36%，重建本身从 55.6% 掉到 53.1%。两边都提前交卷就留 baseline 是任意的。
- 规划失败改成只跑一次 hinted，丢掉双候选收益。
- 「末步两边一致但和 draft 不同就留 draft」反事实净零：会救 17301/17314，同时丢掉 17373/17435。

**v3 留下的：** 17301 因 planner 不再抄 case list 而救回；17310 被末步 draft vote 救回。副作用是 17369：错误 draft=83592 泄漏进末步 goal，baseline 抄泄漏，vote 再锁死。

**v4 留下的：** 识别 few-shot stub 计划，洗掉 `Final Answer: 83592`，两边都做出 152。17283 / 17310 的真多步计划不受影响。

64 题官方（失败记 0）：v4 EM **59.4%**，draft 50.0%，**+9.4pp**，rec 7 / harm 1。512 上增益收缩到 +4.9pp，64 上的 keep 规则没有被 512 打脸。

### 8.5 有意思的发现

**关于 4B 的能力边界**

- **4B 能做步内盲选，不能做整卷裁判。** 步内选 A/B 有用；把两篇完整解答交给它，答案不同时大约一半一半。出口对比不要做。
- **学到的 hint 不是翻转器。** clean-128 上 router ≈ 固定 CoT ≪ oracle。73 道失败里 69 道是「hint 看起来像、14B 还是算错」。协议必须按「第二份噪声求解」来设计，不能按「小模型会指路」来设计。
- **Selector 解析失败可以做成 0。** 把 `A | B` / tie 固定解析成盲选标签 A（provenance 已随机，不偏向 baseline）。HF5 的 7 次解析失败曾经静默吞掉决策。

**关于 draft / plan**

- **Draft 只适合当「答案级第三票」，不适合当写作成品。** 只在末步、三个答案都抽得出、且只与一边相同时覆盖 selector。v4 上 512 题触发 29 次。
- **Baseline 方向的 vote 会被泄漏污染。** Planner 把 draft 答案写进末步 goal 后，baseline 抄 goal，vote 再确认抄袭（17369）。hinted 与 draft 一致更像独立重合，可以留。
- **Plan 克隆是一种独立 harm。** 目标写成「检查 n=1 / n=3」等于把 draft 的残缺搜索当成计划。要写方法，不要写 draft 查过哪些值。
- **4B planner 仍会泄漏数字。** 只有 stub 计划的末步泄漏会被洗掉。真多步计划里的 `Final Answer: N` 故意不洗，以免误伤「最后写答案」这种合法 goal。
- **4B planner 有 context 墙。** holdout512 有 6 题 planner 输入 ≥7425，再加 768 输出超过 8192（`17438`、`17679`、`17764`、`17889`、`17907`、`18306`）。规划失败 40/506（7.9%），走单步 fallback。

**关于选择与伤害**

- **提前交卷普遍，但不能硬砍。** 512 题选中文本里有 614 次非末步 `Final Answer`。硬拒会把提前写对的一边扔掉（v2 的 17373、17283）。
- **Fallback 必须保留双候选。** v1 的 17346/17369 都靠「规划失败 + baseline vs hinted」。hinted-only 会把对的 baseline 丢掉。
- **「两边一致就听 draft」净零。** 反事实会救 17301/17314，同时丢掉 17373/17435。不要做。
- **增益来自少害，不是多救。** 配对 506：Draft rec/harm 48/23，HF1 50/38。Interact rec/harm 57/33——多救了一些，害也更多。OSS 上 HF1 能做到 rec 267 / harm 5，14B 上同一哲学做不到，因为 challenger 并不明显强于 baseline。

**关于协议税和预算**

- **固定 5 步的协议税是真的。** HF5 @8k 几乎不换 hint（2465 BASELINE / 90 HINTED），EM 掉到裸解以下。Draft 的 hint 采纳率约 **54.7%**（1029/1881），且各步都在 46–59%，没有 HF5 那种末步关门。
- **步数该由题决定。** Draft 均值 3.72；分布 1:40、2:138、3:45、4:111、5:85、6:59、7:17、8:11。众数是 2 和 4，不是 5。
- **同预算下 Draft 略赢 HF1。** 14B 实际写出的字大约是 HF1 的 1.2×（1 次 8k draft + 平均每步 800–900 字的短候选，对 HF1 的两份整卷）。max-token 帽约 1.43×。4B 调用更多（约 8.4 vs 2），但不改档位。隐藏成本是后几步的 context prefill。
- **Interact 更贵也更高。** holdout64 已经 62.5%（+7.8pp）；512 是 58.2%。平均 2.63 次可解析 14B 求解，外加 plan/critique。绝对 EM 目前最高，不是同一预算。
- **「相对自己的裸解」比跨协议绝对 EM 更稳。** Draft seed `20260905`，HF1 seed `20260817`。裸 draft 52.0% 和 HF1 baseline 53.5% 不是同一次 14B 调用。比的是协议增益和 harm。
- **格式 recover 是真分数。** 有的题 draft 已经写出正确答案，但写成 `Answer:` 抽不出来；重建写成 `Final Answer:` 才计分。评测抽取器和训练抽取器必须一致（见 §7.1）。

### 8.6 产物与怎么跑

```bash
# 需 14B @ :8008，GRPO 4B @ :8086
# holdout 64（迭代用）
python HintFlow_draft/eval_draft.py --mode full --limit 64 --workers 8 \
  --solver-urls http://127.0.0.1:8008/v1 \
  --out-dir checkpoints/eval_hintflow_draft_holdout64_v4

# holdout 512（和 HF1 对照）
bash scripts/run_hintflow_draft_holdout512_vs_hf1.sh
# 产物：
#   checkpoints/eval_hintflow_draft_holdout512_v4/full_summary.json
#   checkpoints/eval_hintflow_draft_holdout512_v4/compare_hf1.json
```

主要产物：

| 路径 | 内容 |
|------|------|
| `checkpoints/eval_hintflow_draft_holdout64_v{1-4}/` | 64 题迭代 |
| `checkpoints/eval_hintflow_draft_holdout512_v4/` | 512 官方 + `compare_hf1.json` |
| `checkpoints/compare_holdout512_hf1_qwen14b_8k/` | HF1 对照 |
| `checkpoints/eval_hintflow_interact_holdout512_v1/` | Interact |
| `checkpoints/compare_holdout512_hf5_qwen14b_{2kx5,8kx5}/` | HF5 |
| `checkpoints/blind_hint_17k_qwen3_14b_nothink/` | 14B clean labels |

单测：`python -m unittest discover -s tests -p 'test_hintflow_draft.py'`。

当前不要做的：4B 出口整卷对比、非末步硬拒 `Final Answer`、hinted-only fallback、「末步两边一致就听 draft」、清洗所有 goal 里的数字、再把 HF5 的固定 5 步加长 token。

---

## 常用命令速查

```bash
# OSS-20B
bash scripts/serve_oss_compact.sh          # 省显存
bash scripts/serve_oss_4gpu.sh             # 高吞吐

# Blind label → SFT → eval（OSS 线）
bash scripts/run_blind_hint_17k_tmux.sh
bash scripts/run_blind_ff_sft_17k_tmux.sh
bash scripts/run_blind_ff_sft_17k_eval_tmux.sh
bash scripts/run_ff_idist_eval_tmux.sh

# Dedup 消融
python run.py ff-dedup-blind --repeats 3
bash scripts/run_blind_ff_sft_17k_dedup_tmux.sh
bash scripts/run_blind_ff_sft_17k_dedup_eval_tmux.sh

# Draft v4（需 14B @ :8008，GRPO 4B @ :8086）
python HintFlow_draft/eval_draft.py --mode full --limit 64 --workers 8 \
  --solver-urls http://127.0.0.1:8008/v1 \
  --out-dir checkpoints/eval_hintflow_draft_holdout64_v4
bash scripts/run_hintflow_draft_holdout512_vs_hf1.sh
```
