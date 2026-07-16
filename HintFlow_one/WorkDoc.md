# HintFlow_one（Blind FF + Conservative Selector）

**一句话**：one-step Blind Free-Form hint 生成 challenger，再用保守 selector
决定是否替换 bare OSS baseline，目标是保留 Blind FF 的 recover、压掉 harm。

## 流程

```text
Baseline (bare OSS)
   → Blind FF router → free-form hint
   → Challenger (OSS + hint)
   → Selector KEEP | REPLACE
   → Final answer = incumbent
```

## 防 harm 规则

- baseline-first：默认 KEEP
- challenger 不可解析 → KEEP
- baseline 不可解析且 challenger 可解析 → REPLACE
- 答案相同 → KEEP
- orch selector 需 confidence ≥ threshold（默认 0.90）才 REPLACE
- selector 失败 → KEEP

## 与 HintFlow V2 / Residual 关系

- 复用同一套 KEEP/REPLACE 安全哲学
- 不做多步 plan/review；只做 one-step FF
- 小模型既可做 hint router，也可做 selector（或分两个 endpoint）
