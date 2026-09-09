# 补充实验：只允许物理 GPU 1、2

**本地 Mac 不运行模型。训练与推理只在 Linux 服务器执行。物理 GPU 1 固定放训练 policy / 推理 policy，物理 GPU 2 固定放 Qwen3-14B solver。不能切换到 0、3，也不能在显存不足时自动借用其他卡。**

入口通过 `nvidia-smi` 把物理编号解析为 GPU UUID，再设置每个子进程的 `CUDA_VISIBLE_DEVICES`。因此日志里的逻辑 `cuda:0` 是被绑定的物理1或2，不是物理0。两张卡已有计算任务时直接退出；不会杀已有服务、不会复用旧端口服务。服务器需要先获得这两张卡的合法调度分配。同账号加锁避免重复启动本套实验。

## 服务布局与依赖

- 训练时 GPU1：4B LoRA policy + 冻结4B reference；GPU2：14B vLLM reward server。
- 评测时 GPU1：4B vLLM，加载 SFT/current 两个LoRA；GPU2：14B vLLM。
- 六次RL训练串行执行，默认3 seeds × off/vp；训练结束后释放GPU1，再启动评测policy。评测不同checkpoint也串行。
- 使用仓库现有 Python 环境，需要 torch、transformers、peft、openai、vllm；画图另需 matplotlib。不在脚本中自动安装或升级依赖。
- OOM或服务启动失败就停止；检查日志并降低并发/上下文等后用新的实验输出目录，不能改GPU约束。未在本地验证服务器显存容量与依赖兼容性。

## 服务器操作

```bash
git pull --ff-only
cp experiments/supplement/config.example.json experiments/supplement/config.local.json
```

编辑 `config.local.json` 中的模型路径、SFT adapter路径、train/eval文件。输入JSONL每行至少有唯一 `id`、`problem`、`gold`。默认正式长度200步、batch32、K8，固定checkpoint每25步保留。新评测集默认路径是占位路径，脚本不会擅自把旧holdout或训练题当新测试集。

先准备测试题来源；示例中的文件必须替换为实际可用来源，并列全训练/标签/开发题排除清单：

```bash
python -m experiments.supplement freeze-data \
  --candidates data/new_candidate_pool.jsonl \
  --exclude data/train.jsonl \
  --exclude data/val.jsonl \
  --exclude data/grpo_clean_iid.jsonl \
  --exclude data/all_previous_development_problems.jsonl \
  --output data/supplement_test_frozen.jsonl --count 1024
```

这个命令仅做规范化精确去重；近重复、标签题覆盖与历史开发用途仍需审计。不能因此声称排除了预训练污染。保存的数据manifest记录来源hash、题目ID和抽样seed，不覆盖已冻结集合。

服务器上先运行不调用模型的回归测试：

```bash
python -m unittest discover -s tests -p test_supplement.py
```

查看配置和执行计划（无模型加载）：

```bash
python -m experiments.supplement run train --config experiments/supplement/config.local.json
```

建议复制一份独立smoke配置，将 `max_steps=1`、`checkpoint_every=1`、`batch_size=2`、`k=2`、`seeds=[41]`、`eval_seeds=[101]`，使用小型dev文件，输出改为 `checkpoints/supplement_smoke`。运行train/eval确认依赖、模型模板和显存可用后，再使用正式配置。smoke结果不能混入论文。

正式执行（建议在已有 tmux 中）：

```bash
python -m experiments.supplement run train --config experiments/supplement/config.local.json --execute
python -m experiments.supplement run eval --config experiments/supplement/config.local.json --execute
python -m experiments.supplement run hfm --config experiments/supplement/config.local.json --execute
python -m experiments.supplement analyze --output checkpoints/supplement_v1 --plots
```

也可一次串行执行以上全部阶段：

```bash
python -m experiments.supplement run all --config experiments/supplement/config.local.json --execute
```

每个阶段会启动自己需要的服务，结束/正常中断时回收自己的子进程。`kill -9` 无法清理服务；此时先人工检查所属PID，入口会因GPU占用退出，不会自行杀进程。

查看进度与日志：

```bash
python -m experiments.supplement status --output checkpoints/supplement_v1
tail -f checkpoints/supplement_v1/logs/train-off-41.log
```

## 已实现的实验和记录

- A1：off/vp只改变全错组advantage，off仍保留KL。固定SFT reference、seed、题目顺序、反馈次数配置。真实reward使用现有anti-leak-aware评分；不是把API错误记成失败reward。
- A2：SFT/off/vp、fixed-CoT、resample；每题每seed共享bare，显式统一challenger温度。SFT selector固定；当前checkpoint selector另报原生系统结果。缓存bare占用概念call0，challenger固定使用call1种子。
- A3：候选池直接算bare/challenger/random期望/union，并对SFT与VP做selector交换。没有正确候选时oracle也失败，oracle不进入部署选择流程。
- A4：8192/16384/32768每题solver输出cap；HF1各半，HFM draft四分之一、剩余额度平分实际计划的候选调用；HFM请求前ledger检查。客户端隐藏重试全部关闭，错误也保留请求记录。
- B：逐组reward/advantage/K/原文、逐步PG/KL/组别、每次API原始request/response/usage/耗时、失败前hint/draft；主结果图、R/H图与训练全错组曲线从实测记录生成。

HFM的draft提示不同于HF1，不复用为同一bare；只比较同题最终EM，不能把各自recovery/harm差异解释为同baseline上的因果效应。等solver输出cap不等于等总计算成本；完整调用usage包含小模型。缓存组件耗时求和是估计的串行服务耗时，不是重新部署后的真实端到端计时。

## 恢复、成本与结果边界

训练重新执行同一命令会从最后一个有 `COMPLETE` 标记的不可变snapshot恢复adapter、optimizer、RNG和数据cursor；最多重做checkpoint间隔内的更新。每次尝试有独立API日志，丢弃的训练尝试仍计实际资源成本。根目录的临时最新权重不是可信恢复点。

评测每题每组件原子落盘，已有错误记录也保留，不“重试到成功”。请求发出前写journal和 `.inflight`；突然中断后需要检查该marker和调用记录，禁止自动重复未知成本的请求。协议、代码commit、数据、base模型或SFT权重hash变动会拒绝复用旧实验目录。更新代码后用新输出目录；不要删manifest来强行继续。

`analysis/statistics.json` 中CI是以题为单位的配对bootstrap，保留同题全部重复；同时列训练seed均值。不把题数×重复数当独立样本。不完整方法只输出进度性均值、不输出CI；副比较没有进行多重显著性宣称。usage缺失时token为已知下界，不当成免费请求。

新框架不自动执行未冻结的长SFT矩阵、人工盲标或hint语义编辑；这些需要各自的数据/标注协议。训练日志图不是固定probe学习曲线，也不是虚构的收敛证明。所有正式模型实验和运行测试均须在服务器验证，源码提交不意味着实验已完成。

结果目录默认被git忽略，避免上传权重、原始大轨迹和大量调用。`git pull`同步代码；服务器实验进度在服务器 `status.json` 和日志中，不能通过拉代码获得模型权重。需要汇报时可单独提交脱敏的统计摘要。
