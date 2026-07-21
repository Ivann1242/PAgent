# HintFlow Agent V2（画图用 · 简版）

**一句话**：可训练小 Orch（Qwen3-4B）用紧凑可信状态控制冻结强
Solver（GPT-OSS-20B），同时保留 baseline，避免后续步骤破坏已有正确答案。

## 主流程
`Baseline → Plan → [Fresh Solver step → StepResult → Review] × N → Candidate Selector`

- 每次 Solver 都是 fresh context：`Problem + accepted results + current goal + hint`
- Solver 返回结构化 StepResult：短结果、关键公式、候选答案、不确定性
- Reviewer 将结果分为 `ACCEPT / UNCERTAIN / REJECT`
- 三层状态：accepted facts / open hypotheses / rejected results
- 所有可解析答案进入 candidate archive；空 incumbent 可由完整高置信候选接管，
  已有答案则至少需要两个独立候选一致后才允许 selector 替换
- Turn budget 默认 7，包含 baseline；全局最多 1 次 RETRY 和 1 次 REPLAN

## 兼容与消融
- `legacy`：旧多轮 messages
- `fresh`：fresh context
- `structured`：fresh + StepResult + 分层状态
- `retained`：完整 V2，再加 baseline-first + candidate retention