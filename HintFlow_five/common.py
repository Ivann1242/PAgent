from __future__ import annotations

import json
import re
from typing import Any

N_TURNS = 5
SELECTOR_SYSTEM = """You are a conservative pairwise selector for incremental math reasoning. Compare the no-hint BASELINE and HINTED candidate for the same current step. Select HINTED only when it is more mathematically correct, useful, and consistent with accepted progress. If uncertain, select BASELINE. Output only valid JSON with decision, confidence, and reason."""
RANKER_SYSTEM = """You are a math-process ranker. Rank every supplied successful hinted candidate exactly once, best first. Output only valid JSON with ranking and reason."""


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    raw = strip_fences(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            # Math models often emit LaTeX with JSON-invalid single backslashes.
            candidate = re.sub(chr(92) * 2 + '(?!["' + chr(92) + '/bfnrtu])', lambda _: chr(92) * 2, candidate)
            obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


def state_text(steps: list[str]) -> str:
    if not steps:
        return "(none; start from the problem)"
    return "\n".join(f"Step {i}: {step.strip()}" for i, step in enumerate(steps, 1))


def build_router_prompt(problem: str, accepted_steps: list[str], turn: int, n_turns: int = N_TURNS) -> str:
    remaining = n_turns - turn + 1
    return f"""You are a concise process-hint router for a math solver.

Problem:
{problem.strip()}

Accepted progress:
{state_text(accepted_steps)}

Current turn: {turn} of {n_turns}
Turns remaining including this one: {remaining}

Give one short, actionable hint for what the solver should do in THIS turn.
The hint may identify a method, missing check, or correction. Do not solve the
whole problem and do not state the final answer. On turn {n_turns}, focus on
checking the accumulated work and producing the requested final-answer format.

Output only the hint text."""


def build_step_prompt(
    problem: str, accepted_steps: list[str], turn: int, hint: str = "", n_turns: int = N_TURNS
) -> str:
    final_rule = (
        "This is the final turn. Verify the accumulated work, finish the solution, "
        "and end with exactly: Final Answer: <answer>."
        if turn == n_turns
        else "Do not give a final answer yet. Produce one compact, self-contained next step."
    )
    hint_block = f"\nProcess hint:\n{hint.strip()}\n" if hint.strip() else ""
    return f"""Solve the math problem through exactly {n_turns} incremental turns.

Problem:
{problem.strip()}

Accepted progress:
{state_text(accepted_steps)}

Current turn: {turn} of {n_turns}.
{hint_block}
{final_rule}
Do not repeat earlier accepted steps. Return only the content of the current step."""


def build_selector_prompt(
    problem: str,
    accepted_steps: list[str],
    turn: int,
    baseline: str,
    hinted: str,
    n_turns: int = N_TURNS,
) -> str:
    return f"""You are a conservative selector for incremental math reasoning.
Choose which candidate is more correct, useful, and consistent with the accepted
progress. Do not prefer the hinted candidate merely because it received a hint.
If evidence is inconclusive, choose BASELINE. On the final turn, strongly check the final
answer and requested format. Do not use any hidden gold answer.

Problem:
{problem.strip()}

Accepted progress:
{state_text(accepted_steps)}

Turn: {turn} of {n_turns}

[BASELINE]
{baseline.strip()}

[HINTED]
{hinted.strip()}

Output only JSON:
{{"decision":"BASELINE | HINTED","confidence":0.0,"reason":"short concrete reason"}}"""


def build_ranker_prompt(problem: str, accepted_steps: list[str], turn: int, candidates: list[tuple[int, str, str]], n_turns: int = N_TURNS) -> str:
    blocks = []
    for candidate_id, hint, output in candidates:
        blocks.append(f"[CANDIDATE {candidate_id}]\nHint:\n{hint.strip()}\n\nResulting step:\n{output.strip()}")
    joined = "\n\n".join(blocks)
    return f"""You are ranking successful process hints for incremental math reasoning.
Every candidate already beat the shared no-hint baseline. Rank them by mathematical
correctness, usefulness, consistency with accepted progress, and support for completion.
Judge each hint through its resulting step. Do not use a hidden gold answer.

Problem:
{problem.strip()}

Accepted progress:
{state_text(accepted_steps)}

Turn: {turn} of {n_turns}

{joined}

Output only JSON containing every candidate id exactly once, best first:
{{"ranking":[1,2],"reason":"short concrete reason"}}"""
