````md
# GRPO Action Router 方案

## 目标

实现一个端到端强化学习框架：训练一个小模型 `Qwen4B`，让它根据题目从固定 `ACTION_SPACE` 中选择最合适的 Hint，然后把该 Hint 提供给 frozen 大模型 `GPT-OSS-20B` 解题。

只训练小模型，大模型完全冻结。

---

## 模型

- Policy / Router：`Qwen4B`，可训练
- Solver：`GPT-OSS-20B`，冻结
- 数据集：DAPO
- 训练方法：GRPO
- Reward：大模型最终答案的 EM + 格式奖励

---

## Action Space

```python
ACTION_SPACE = {
    "baseline": "",
    "careful_reading": (
        "Read the problem carefully. Identify all given quantities, constraints, "
        "and what is being asked before solving."
    ),
    "step_by_step": (
        "Reason step by step internally, but output only the final answer in the required format."
    ),
    "format_first": (
        "Determine the required answer format first, then derive the final result in that exact format."
    ),
    "reject_unsupported": (
        "Do not introduce facts or intermediate steps that are not supported by the problem."
    ),
    "type_aware": (
        "Identify the problem type and use a strategy suited to that type."
    ),
}
````

---

## 小模型 Prompt

```python
prompt_to_small = f"""
You are an action router.

Problem:
{problem}

Action Space:
{ACTION_SPACE}

Choose exactly one action key from the action space.

Output only JSON:
{{"action": "<action_key>"}}
"""
```

小模型输出示例：

```json
{"action": "type_aware"}
```

如果输出无法解析，默认使用 `"baseline"`，并给 invalid action penalty。

---

## 大模型 Prompt

```python
hint = ACTION_SPACE[selected_action]

prompt_to_large = f"""
Solve this problem.

Problem:
{problem}

Hint:
{hint}

Output the final answer in this format:

Final Answer: <answer>
"""
```

---

## Reward

只根据 frozen 大模型的最终输出打分。

```python
reward = em_reward + format_reward + invalid_action_penalty
```

推荐设置：

```python
em_reward = 1.0 if pred_answer == gold_answer else 0.0
format_reward = 0.1 if format_ok else -0.1
invalid_action_penalty = -0.2 if action_parse_failed else 0.0
```

注意：EM 是主 reward，format reward 不要太大。

---

## 训练流程

对每个 problem：

1. 小模型读取 `problem + ACTION_SPACE`
2. 小模型采样一个 action
3. 解析 action，得到 hint
4. 冻结大模型读取 `problem + hint`
5. 大模型输出答案
6. 提取 `Final Answer`
7. 计算 EM 和 reward
8. 用 GRPO 更新小模型

伪代码：

```python
for batch in train_loader:
    rollouts = []

    for example in batch:
        problem = example["problem"]
        gold = example["answer"]

        group = []

        for _ in range(K):
            small_prompt = build_small_prompt(problem, ACTION_SPACE)

            small_output, logprob = small_model.generate_with_logprob(
                small_prompt,
                temperature=1.0,
            )

            action, parse_ok = parse_action(small_output)
            hint = ACTION_SPACE[action]

            large_prompt = build_large_prompt(problem, hint)

            large_output = large_model.generate(
                large_prompt,
                temperature=0.0,
            )

            pred = extract_final_answer(large_output)

            reward = compute_reward(
                pred=pred,
                gold=gold,
                parse_ok=parse_ok,
                large_output=large_output,
            )

            group.append({
                "problem": problem,
                "gold": gold,
                "action": action,
                "hint": hint,
                "small_output": small_output,
                "large_output": large_output,
                "pred": pred,
                "reward": reward,
                "logprob": logprob,
            })

        advantages = normalize_rewards_within_group(group)

        rollouts.extend(add_advantages(group, advantages))

    grpo_update(small_model, rollouts)
```

---

## GRPO 设置

推荐初始配置：

```python
K = 8
batch_size = 32
learning_rate = 1e-6
clip_range = 0.2
kl_beta = 0.02
small_temperature_train = 1.0
small_temperature_eval = 0.0
large_temperature = 0.0
```

Advantage 使用组内归一化：

```python
adv = (reward - mean(group_rewards)) / (std(group_rewards) + 1e-6)
```

---

## 必须记录的日志

每条 rollout 保存：

```python
{
    "problem": problem,
    "gold_answer": gold,
    "selected_action": action,
    "hint": hint,
    "small_output": small_output,
    "large_output": large_output,
    "pred_answer": pred,
    "em": em,
    "format_ok": format_ok,
    "reward": reward,
}
```

---

## 评估

至少跑四组：

1. **Baseline**：大模型直接解题，无 hint
2. **Random Action**：随机选择 hint
3. **Router**：训练后小模型选择 hint
4. **Oracle Action**：每题遍历所有 action，取 reward 最高的 action

报告指标：

```text
EM
Format accuracy
Average reward
Invalid action rate
Action distribution
Per-action EM
Oracle EM
Router vs Baseline improvement
Router vs Random improvement
```

---

## 关键检查

训练前必须先跑：

```text
per-action eval
oracle-action eval
```

如果：

```text
oracle EM - baseline EM < 1%
```

说明当前 action space 没有足够 headroom，GRPO 很难学出提升，需要重新设计更强的 action space。

---

## 成功标准

训练后的 router 应满足：

```text
Router EM > Baseline EM
Router EM > Random Action EM
Invalid action rate low
Oracle EM significantly > Baseline EM
Action distribution non-trivial
```

---

## 一句话总结

训练一个 Qwen4B action router，用 GRPO 学会为每道 DAPO 题选择最合适的 hint，并让 frozen GPT-OSS-20B 在该 hint 帮助下解题，reward 来自大模型最终答案的 EM 和格式正确性。

```
```
