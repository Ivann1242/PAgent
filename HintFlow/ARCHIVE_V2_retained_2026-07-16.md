# HintFlow V2 Retained — Result Archive (2026-07-16)

Frozen snapshot of the poster-facing result. HintFlow Agent work paused after this;
mainline continues on one-step Blind FF.

## Artifact paths

| Item | Path |
|---|---|
| Repeat aggregate | `checkpoints/eval_hintflow_retained_fixed_repeat4/repeat_summary.json` |
| Run 1 (seed 41) | `checkpoints/eval_hintflow_retained_fixed_repeat4/run_1_seed_41/` |
| Run 2 (seed 42) | `checkpoints/eval_hintflow_retained_fixed_repeat4/run_2_seed_42/` |
| Run 3 (seed 43) | `checkpoints/eval_hintflow_retained_fixed_repeat4/run_3_seed_43/` |
| Run 4 (seed 44) | `checkpoints/eval_hintflow_retained_fixed_repeat4/run_4_seed_44/` |
| Master log | `logs/eval-hintflow-retained-fixed-repeat4.log` |
| Code (runtime) | `HintFlow/HintFlowAgent.py` (`runtime_mode=retained`) |
| Eval entry | `HintFlow/eval_hintflow.py` |
| Launch script | `scripts/run_hintflow_repeat4_gpu3.sh` |

## Protocol

- Dataset: DAPO-Math 128
- Solver: frozen GPT-OSS-20B (`:8006`)
- Orch: Qwen3-4B HQ checkpoint (`:8086`), `orch_temperature=0`
- Budget: **4k tokens per solver call**
- Mode: `retained` (baseline-first + compact state + conservative selector)
- Seeds: 41–44 (sequential ×4, skip live external baseline; report internal baseline)

## Headline numbers

| Seed | HintFlow EM | Internal baseline | Δ (pp) | Recover | Harm | Retention |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 51.56% | 43.75% | +7.81 | 11 | 1 | 98.21% |
| 42 | 49.22% | 41.41% | +7.81 | 10 | 0 | 100% |
| 43 | 51.56% | 46.09% | +5.47 | 7 | 0 | 100% |
| 44 | 49.22% | 41.41% | +7.81 | 10 | 0 | 100% |
| **Mean** | **50.39%** | **43.16%** | **+7.23** | **9.5** | **0.25** | **99.55%** |

Aggregate over 4 runs: recover **38**, harm **1**.

## How harm is avoided (frozen design)

1. **Baseline-first incumbent**: turn-0 bare solve is the default final answer.
2. **Eligibility gate**: only complete + high-confidence StepResults may challenge.
3. **Consensus guard**: parseable incumbent is replaced only if ≥2 independent STEP candidates agree (`min_replace_support=2`).
4. **Selector threshold**: orch KEEP/REPLACE needs `confidence ≥ 0.90`; uncertain → KEEP.
5. **Fail-closed**: selector errors / malformed JSON → KEEP.
6. **Unparseable exception**: empty baseline may be replaced by one complete high-confidence challenger (recovery, not harm).

## Context vs Blind FF (same 4k regime)

| Method | EM mean | Notes |
|---|---:|---|
| Bare OSS@4k ×4 | 41.8% | no orch |
| Residual v3@4k ×4 | 47.7% | retention≈100% |
| Blind FF-SFT 17K full ×4 | 51.56% | recover high, still has baseline-only harm |
| **HintFlow V2 retained ×4** | **50.39%** | near-zero harm |

## Status

- Poster result: **report this table**.
- Next research focus: **one-step Blind FF** (+ optional KEEP/REPLACE selector migration).
- HintFlow multi-step Agent: parked; code kept for later V2 controller training.
