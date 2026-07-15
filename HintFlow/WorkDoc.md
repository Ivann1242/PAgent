# HintFlow 交接流程

目标：小模型 orch 编排 frozen OSS；硬指标 = **最终 EM**（不是 `val_pair_acc`）。  
当前阶段：**v2 树 → DPO 已训完并归档**；**v2 的 128 EM 评测因本机 GPU 被锁尚未跑**。

---

## 0. 一分钟现状

| 项 | 状态 |
|----|------|
| 代码 | 已 push：`https://github.com/Ivann1242/PAgent`（`main` @ `b885f46`） |
| 权重 | HF 公开备份（见下） |
| DPO 数据 | repo 内有 `checkpoints/hintflow_trees_v2/dpo_pairs.jsonl`（~98MB，13565 pairs） |
| 原始树 | **未进 git**（单文件 >500MB）；本机仍有 `checkpoints/hintflow_trees_v2/trees.jsonl`（810 棵） |
| v1 DPO 128 EM | **50.78%**（对照 bare 39.84% / baseline 47.66%） |
| v2 DPO | 训完；pair-acc≈50.9%；**128 EM 待评** |

---

## 1. 拉代码 + 权重（别处开实验）

```bash
git clone https://github.com/Ivann1242/PAgent.git
cd PAgent

# 合并权重（vLLM serve 用）
huggingface-cli download ivaning0919/pagent-hintflow-dpo-v2-merged \
  --local-dir checkpoints/hintflow_dpo_v2_merged

# 可选：LoRA adapter
huggingface-cli download ivaning0919/pagent-hintflow-dpo-v2-adapter \
  --local-dir checkpoints/hintflow_dpo_v2_adapter
```

- Merged：https://huggingface.co/ivaning0919/pagent-hintflow-dpo-v2-merged  
- Adapter：https://huggingface.co/ivaning0919/pagent-hintflow-dpo-v2-adapter  
- Base：`Qwen/Qwen3-4B`（训练时本机路径曾是 `.../prompt-r1r/Prompt-R1/Qwen/Qwen3-4B`，别处请改 `config.ROUTER_BASE` 或 `--base-model`）

依赖：Python、`torch`、`transformers`、`peft`、`openai`、`vllm`（serve）、`tqdm`。评测数据：`data/DAPO-Math.parquet`（128）。

**Colab 单卡 G4（96GB）一键评测 notebook**：`HintFlow/eval_dpo_v2_h100.ipynb`（clone → 装依赖 → 下 OSS+orch → 同卡 serve → `eval_hintflow.py`）。

---

## 2. 目录与脚本地图

```
HintFlow/
  HintFlowAgent.py    # orch 协议（plan / review / inject / replan / finalize）
  collect_trees.py    # 离线树状分叉采数
  export_dpo_pairs.py # 树 → DPO pairs（按子树 leaf-EM 均值 V 做 chosen/rejected）
  train_dpo.py        # 离线 DPO（支持 torchrun 多卡 + ref-cache + mem-fence）
  eval_hintflow.py    # 128 EM：live_baseline + hintflow

scripts/
  serve_oss_4gpu_batch8.sh      # OSS ×4 → :8006–8009
  serve_qwen_router_batch8.sh   # orch → :8086（改 MERGED=...）
  run_hintflow_eval128_tmux.sh  # 评测封装
  upload_hintflow_dpo_v2_hf.py  # HF 上传
```

本地关键产物：

| 路径 | 内容 |
|------|------|
| `checkpoints/hintflow_trees_v2/trees.jsonl` | 原始树（本地 only） |
| `checkpoints/hintflow_trees_v2/dpo_pairs.jsonl` | 导出 pairs（在 git） |
| `checkpoints/hintflow_trees_v2/dpo_ref_cache.pt` | ref logp 缓存（本地 only，可重算） |
| `checkpoints/hintflow_dpo_v2_{adapter,merged}/` | 本机训练产物 |
| `checkpoints/eval_hintflow_{128,dpo_128}/` | 旧 128 评测结果 |

---

## 3. 标准流水线（复现 / 续跑）

### 3.1 Serve

```bash
# 4× OSS（gpt-oss-20b）
bash scripts/serve_oss_4gpu_batch8.sh

# orch = v2 merged
MERGED=checkpoints/hintflow_dpo_v2_merged \
MODEL_NAME=qwen3-4b \
GPU=0 PORT=8086 \
bash scripts/serve_qwen_router_batch8.sh
```

健康检查：`curl -s http://127.0.0.1:8006/v1/models` 与 `:8086/v1/models`。

### 3.2 采树（可选；已有 pairs 可跳过）

```bash
python HintFlow/collect_trees.py \
  --workers 24 --max-inflight-solver 256 \
  --out checkpoints/hintflow_trees_v2/trees.jsonl
```

要点：orch 凡调用处分叉（plan / review / replan），k=2；叶子记 EM；V 可后处理。

### 3.3 导出 DPO pairs

```bash
python HintFlow/export_dpo_pairs.py \
  --trees checkpoints/hintflow_trees_v2/trees.jsonl \
  --out checkpoints/hintflow_trees_v2/dpo_pairs.jsonl \
  --tau 0.05 --min-leaves 1
```

当前 v2 统计：810 trees → **13565 pairs**（plan 466 / review 13099），`ΔV` 均值 ≈0.29（弱于早期 2k 的 ~0.63）。

### 3.4 训练 DPO

```bash
# 多卡示例（自动 torchrun）；ref-cache 可跳过二次预计算
python HintFlow/train_dpo.py \
  --pairs-file checkpoints/hintflow_trees_v2/dpo_pairs.jsonl \
  --adapter-dir checkpoints/hintflow_dpo_v2_adapter \
  --merged-dir checkpoints/hintflow_dpo_v2_merged \
  --gpu 0,1,2 \
  --epochs 3 --batch-size 1 --grad-accum 3 \
  --lr 5e-6 --beta 0.1 \
  --ref-cache checkpoints/hintflow_trees_v2/dpo_ref_cache.pt \
  --mem-fence-mb 20480 \
  --log-every 50
```

说明：

- `val_pair_acc` = 验证集上 `logp(chosen) > logp(rejected)`，**不是 EM**。
- `--mem-fence-mb`：sticky 占住剩余显存，防被同卡任务挤爆；`0` 关闭。
- v2 训练结果：e1/e2 pair-acc **50.2%**，e3 **50.9%**（偏好几乎没学到）。

### 3.5 128 EM 评测（优先待办）

```bash
python HintFlow/eval_hintflow.py \
  --limit 128 --workers 32 \
  --solver-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1 \
  --orch-url http://127.0.0.1:8086/v1 \
  --orch-model qwen3-4b \
  --out-dir checkpoints/eval_hintflow_dpo_v2_128
# 若已有 baseline：加 --skip-baseline，再手动把 live_baseline 拷进 summary
```

对照（已有）：

| 设定 | EM |
|------|-----|
| live baseline | 47.66% |
| bare HintFlow（未 DPO） | 39.84% |
| HintFlow + DPO v1 | **50.78%** |
| HintFlow + DPO v2 | **待测** |

---

## 4. 已知问题 / 坑

1. **pair-acc≈50% ≠ 实验失败**：最终看 128 EM；但 v2 数据噪声大时 EM 也可能平。  
2. **数据偏斜**：97% review；建议下一版抬高 `--tau`（如 0.2）、限 review 数量或 plan/review 分开。  
3. **ying-compute GPU 锁**：`/etc/udev/rules.d/99-gpu-reserved.rules` 把 nvidia0/1/2 绑给 `gpu-reserved`（仅 broz）；nvidia3 共享但常被占满。无时限，需协调或换机（如 Great Lakes `gpu-rtx6000`）。  
4. **大文件**：`trees.jsonl` / snap / `dpo_ref_cache.pt` 不进 git；权重走 HF。  
5. **HF token**：写权限用 `Prompt-R1/.env` 里的 `HF_WRITE_TOKEN`（不要误用只读 `HF_TOKEN`）。

---

## 5. 建议下一步（优先级）

1. **在有卡的机器上跑 v2 128 EM**，对比 v1 / bare / baseline。  
2. 若 EM 无明显提升：用更高 `τ` / 降采样 review **重导出 pairs → 再 DPO**（小数据快速验证）。  
3. 需要更大树数据时再开 `collect_trees`；否则直接用现有 `dpo_pairs.jsonl`。  
4. Great Lakes：本机 `ying-compute` 未配 `Host gl` SSH；从笔记本 `ssh gl` 后 `salloc -p gpu-rtx6000 --gres=gpu:2 ...`，再按 §3 serve+eval。

---

## 6. 交接检查清单

- [ ] `git pull` 最新 `main`  
- [ ] HF 下载 v2 merged（+ 可选 adapter）  
- [ ] 配置 `ROUTER_BASE` / OSS 模型路径  
- [ ] 起 OSS + orch，curl 通  
- [ ] 跑 `eval_hintflow.py` → `checkpoints/eval_hintflow_dpo_v2_128/summary.json`  
- [ ] 记录 EM 与 paired flips，决定是否清洗 pairs 再训  

联系上下文：本仓库 HintFlow 目录 + 上述 HF repo；训练日志本机 `logs/hintflow-dpo-v2-train.log`。
