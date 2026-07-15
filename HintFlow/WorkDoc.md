# HintFlow 对齐方案（短版）

目标：小模型 orch 编排 frozen OSS；唯一硬指标 = 最终 EM。  
路径：**先修协议 → 再离线采树（本阶段停在数据）→ 对齐后再做 offline RL**。

---

## Step 1 — 禁止无答案早停（先做，再采树）

**规则**

- 无 `Final Answer:` / 可解析短答案（含 `\boxed{}`）→ **禁止 FINALIZE**。
- 计划最后一步（或收尾步）必须显式要求：`Final Answer: <answer>`。
- 抽取与 baseline 对齐：至少支持 `Final Answer:` + `\boxed{}`。

**验收**：同一 128 eval 上，早停假阳性明显下降；再开始 Step 2。

---

## Step 2 — 离线树状分叉采数（只采数据，先不训）

把每道题的 orch 决策当成一棵树：每个 trainable 节点分叉，前缀共享，每条边一次 OSS。

**采样（第一版）**

| 项 | 设定 |
|----|------|
| 模式 | **Offline** 批采，不 online RL |
| 分支 k | **2**（每节点 2 条边） |
| 深度 | 跟现有 Agent：按 plan / 允许的 replan 走，**自由结束**（受 Step 1 约束） |
| 分叉内容 | **凡 orch 调用都分叉**：根上 plan、中间 review/inject、REPLAN 再 plan；各采 k=2 |
| 题集 | 先 hard / baseline 错优先；暂不计单题预算（4×8 worker） |

**节点上的训练信号（先记清楚，RL 下一步再写）**

- 样本键：`(problem + prefix context) → 当前节点orch 输出`（你写的 hint；实现时可含 action）
- **Reward / value**：以该节点为根，**所有后代叶子最终 EM 的均值**  这个我觉得都可以不用现在计算，反正树状结构跑出来可以之后再想怎么处理。
  \(V(u)=\mathrm{mean}\{EM(\ell):\ell\in\mathrm{leaves}(u)\}\)
- 同父两条边 → 自然有相对好坏（正/负都保留，留给后面 RL，不在此步训）

**本步交付物**

- 树/边 jsonl：state、action/hint、子节点、叶子 EM、\(V\)  
- **不**在本步写训练脚本；采通、格式对齐后再开 offline RL。

---

## Planner（原先缺口 → 同一棵树的根）

中间步的树已经能 credit **control/inject**；**plan 是同一套机制的第 0 层**，不必另发明监督。

```text
problem
  ├─ plan_A ─┬─ ctrl… → leaves EM
  │          └─ ctrl… → …
  └─ plan_B ─┬─ …
             └─ …
```

- **根节点也 k=2**：同一题采样 2 个 plan，各自往下按现有 Agent 展开（含 REPLAN 时的再 plan，同样当 orch 节点分叉）。
- 样本键：`(problem [+ replan context]) → plan JSON`，和中间步一样进树；叶子 EM 后处理算 \(V\)。
- **代价**：根分叉会乘整棵子树 OSS 量；你们暂不计预算则可开。若以后要省，只降根的 k 或先 k=1 只训中间步。
- **和中间步的差别（记一下就行）**：plan 是结构化长文本、决定拓扑；credit 仍靠叶子，不靠逐步「plan 是否好看」。

结论：Step 2 采数时 **orch 凡是会调用处都分叉**（plan / review+inject / replan），planner 不单列阶段。

---

## 刻意后置（对齐后再开）

- Offline RL 目标与 loss（plan vs control 是否分开头）
- 单题 OSS 预算硬顶（若需要）
- Online refresh

---

## 已对齐

1. Step 2 分叉 = 该交互节点上**小模型全部输出**（要训的就是逐步表现）。
2. \(V\) 后处理；采数只保证树结构 + 叶子 EM。
3. REPLAN 不特殊开关；**训推一致**。
4. **Planner = 树根（及 REPLAN 处）的同构分叉**，不另开一条训练线。
