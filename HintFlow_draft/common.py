from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from core import exact_match, extract_final_answer, has_parseable_answer
from HintFlow_five.common import parse_json_object

_CONCRETE_ANSWER_LABEL = re.compile(
    r"(?:final\s+answer|answer)\s*:\s*(?!<answer\s*>)(\S+)",
    re.IGNORECASE,
)
_TEMPLATE_STUB = "result-free executable goal"


PLANNER_SYSTEM = """You decompose a draft math solution into a short executable plan.
Return only valid JSON. Write goals as methods to perform, not values the draft
already computed or cases it already checked."""

SELECTOR_SYSTEM = """You are a blinded pairwise selector for one step of a math solution.
Choose the candidate that is more mathematically correct, consistent with completed
work, and successful at the stated current goal. Candidate labels reveal no provenance.
Output one line of valid JSON with decision, confidence, and a concrete reason of at
most 20 words."""

FALLBACK_FINAL_GOAL = (
    "Solve the problem completely, verify the reasoning and requested format, "
    "and output the final answer as Final Answer: <answer>."
)


@dataclass(frozen=True)
class StepPlan:
    goal: str
    is_final: bool = False


@dataclass(frozen=True)
class AcceptedStep:
    goal: str
    result: str


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = max(limit // 2 - 24, 1)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def fallback_plan() -> list[StepPlan]:
    return [StepPlan(goal=FALLBACK_FINAL_GOAL, is_final=True)]


def parse_step_plan(text: str, *, max_steps: int = 8) -> list[StepPlan]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    obj = parse_json_object(text)
    raw_steps = obj.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("plan must contain a non-empty steps list")
    if len(raw_steps) > max_steps:
        raise ValueError(f"plan has {len(raw_steps)} steps; maximum is {max_steps}")

    steps: list[StepPlan] = []
    for index, raw in enumerate(raw_steps, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"step {index} must be an object")
        goal = str(raw.get("goal") or "").strip()
        if not goal:
            raise ValueError(f"step {index} has an empty goal")
        if len(goal) > 800:
            raise ValueError(f"step {index} goal is too long")
        is_final = raw.get("is_final")
        if not isinstance(is_final, bool):
            raise ValueError(f"step {index} is_final must be boolean")
        steps.append(StepPlan(goal=goal, is_final=is_final))

    final_indices = [i for i, step in enumerate(steps) if step.is_final]
    if final_indices != [len(steps) - 1]:
        raise ValueError("exactly the last step must have is_final=true")
    return steps


def parse_selector_output(text: str) -> tuple[str, float, str]:
    obj = parse_json_object(text)
    decision = str(obj.get("decision") or "").strip().upper()
    if decision in {"A | B", "A/B", "A OR B", "TIE"}:
        # A/B provenance is randomized, so resolving an explicit tie to A is
        # deterministic without introducing a baseline-source preference.
        decision = "A"
    if decision not in {"A", "B"}:
        raise ValueError("selector decision must be A or B")
    try:
        confidence = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("selector confidence must be numeric") from exc
    return decision, confidence, str(obj.get("reason") or "").strip()


def blind_candidates(
    baseline: str,
    hinted: str,
    *,
    seed: int,
    step_index: int,
) -> tuple[str, str, dict[str, str]]:
    digest = hashlib.sha256(f"{seed}:{step_index}".encode()).digest()
    baseline_is_a = bool(digest[0] & 1)
    if baseline_is_a:
        return baseline, hinted, {"A": "BASELINE", "B": "HINTED"}
    return hinted, baseline, {"A": "HINTED", "B": "BASELINE"}


def resolve_blinded_selection(decision: str, mapping: dict[str, str]) -> str:
    label = decision.strip().upper()
    source = mapping.get(label)
    if source not in {"BASELINE", "HINTED"}:
        raise ValueError("invalid blinded candidate mapping")
    return source


def goal_leaks_answer(goal: str, answer: str) -> bool:
    if not goal or not answer:
        return False
    match = _CONCRETE_ANSWER_LABEL.search(goal)
    if not match:
        return False
    leaked = match.group(1).strip().strip("$")
    if exact_match(leaked, answer):
        return True
    return bool(exact_match(extract_final_answer(f"Final Answer: {leaked}"), answer))


def plan_looks_like_draft_copy(goals: list[str] | None) -> bool:
    if not goals:
        return False
    first = (goals[0] or "").strip().lower()
    return _TEMPLATE_STUB in first


def sanitize_stub_leaked_plan(steps: list[StepPlan]) -> tuple[list[StepPlan], bool]:
    if not plan_looks_like_draft_copy([step.goal for step in steps]):
        return steps, False
    last = steps[-1]
    if not last.is_final or not _CONCRETE_ANSWER_LABEL.search(last.goal):
        return steps, False
    cleaned = _CONCRETE_ANSWER_LABEL.sub("Final Answer: <answer>", last.goal)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return [*steps[:-1], StepPlan(goal=cleaned, is_final=True)], True


def prefer_draft_consistent(
    baseline: str,
    hinted: str,
    draft: str,
    *,
    is_final: bool,
    goal: str = "",
    plan_goals: list[str] | None = None,
) -> tuple[str, str] | None:
    if not is_final:
        return None
    if not (
        has_parseable_answer(baseline)
        and has_parseable_answer(hinted)
        and has_parseable_answer(draft)
    ):
        return None
    base_answer = extract_final_answer(baseline)
    hint_answer = extract_final_answer(hinted)
    draft_answer = extract_final_answer(draft)
    if exact_match(base_answer, hint_answer):
        return None
    match_baseline = bool(exact_match(draft_answer, base_answer))
    match_hinted = bool(exact_match(draft_answer, hint_answer))
    if match_baseline == match_hinted:
        return None
    if match_baseline:
        if goal_leaks_answer(goal, draft_answer) and plan_looks_like_draft_copy(
            plan_goals
        ):
            return None
        return "BASELINE", "last-step draft answer matches baseline only"
    return "HINTED", "last-step draft answer matches hinted only"


def context_text(accepted_steps: list[AcceptedStep]) -> str:
    if not accepted_steps:
        return "(none; begin directly from the problem)"
    blocks = []
    for index, step in enumerate(accepted_steps, 1):
        blocks.append(
            f"[Completed step {index}]\n"
            f"Goal: {step.goal.strip()}\n"
            f"Result:\n{step.result.strip()}"
        )
    return "\n\n".join(blocks)


def build_draft_prompt(problem: str) -> str:
    return f"""Solve this problem.

Problem:
{clip(problem, 12000)}

Give a complete mathematical solution and output the final answer in this format:

Final Answer: <answer>"""


def build_segment_prompt(problem: str, draft: str, *, max_steps: int = 8) -> str:
    return f"""Read the problem and the stronger model's bare draft. Convert the draft's
reasoning trajectory into the smallest useful sequence of executable step goals.

The goals will guide a fresh solution. That fresh solution will NOT see the draft.
Therefore:
- Write goals as methods or operations, not claims of what a quantity equals.
- Do not copy the draft's final answer or any concrete numeric result into a goal.
- Do not name specific candidate values to check (for example "check n=1" or
  "verify x=4"). Describe the enumeration or case-analysis method instead.
- Keep each goal specific enough to execute from the problem and completed steps.
- Use a dynamic number of steps appropriate for this problem, from 1 to {max_steps}.
- Mark exactly the last goal with is_final=true.
- The last goal must include verification and producing the requested final-answer
  format. The only allowed answer token is the placeholder Final Answer: <answer>.

Problem:
{clip(problem, 4000)}

Bare draft:
{clip(draft, 10000)}

Output only JSON:
{{"steps":[{{"goal":"result-free executable goal","is_final":false}},
           {{"goal":"verify the work and produce Final Answer: <answer>","is_final":true}}]}}"""


def build_step_hint_prompt(
    problem: str,
    accepted_steps: list[AcceptedStep],
    goal: str,
) -> str:
    return f"""You are helping a stronger math model complete one current goal.
Give one short, actionable hint about the method, check, or correction needed now.
Do not solve the goal, state its result, or give the problem's final answer.
Do not mention any draft solution.

Problem:
{clip(problem, 4000)}

Completed work:
{clip(context_text(accepted_steps), 5000)}

Current goal:
{clip(goal, 1200)}

Output only the hint text."""


def build_goal_step_prompt(
    problem: str,
    accepted_steps: list[AcceptedStep],
    goal: str,
    *,
    hint: str = "",
    is_final: bool = False,
) -> str:
    hint_block = f"\nOptional process hint:\n{hint.strip()}\n" if hint.strip() else ""
    output_rule = (
        "Complete this final goal, verify the accumulated reasoning, and end with "
        "exactly: Final Answer: <answer>."
        if is_final
        else (
            "Stop immediately after a concise result for this goal. Do not solve any "
            "later goal. Even if the original problem requests an answer line, do not "
            "emit Answer:, Final Answer:, <answer>, or a boxed final answer yet."
        )
    )
    return f"""Continue solving the problem from the completed work.

Problem:
{clip(problem, 6000)}

Completed work:
{clip(context_text(accepted_steps), 10000)}

Current goal:
{clip(goal, 1200)}
{hint_block}
If the completed work conflicts with the current goal, explicitly repair the
inconsistency instead of blindly following it. Do not repeat completed steps.
{output_rule}"""


def build_selector_prompt(
    problem: str,
    accepted_steps: list[AcceptedStep],
    goal: str,
    candidate_a: str,
    candidate_b: str,
    *,
    is_final: bool = False,
) -> str:
    final_rule = (
        "This is the final goal: strongly check the requested answer and its format."
        if is_final
        else "This is an intermediate goal: reject premature unsupported final answers."
    )
    return f"""Select the better result for the current goal. Judge only mathematical
correctness, consistency with completed work, goal completion, and usefulness for the
remaining solution. The labels A and B are randomly assigned; do not infer provenance.
{final_rule}

Problem:
{clip(problem, 4000)}

Completed work:
{clip(context_text(accepted_steps), 5000)}

Current goal:
{clip(goal, 1000)}

[Candidate A]
{clip(candidate_a, 4500)}

[Candidate B]
{clip(candidate_b, 4500)}

Output only JSON:
{{"decision":"A","confidence":0.0,"reason":"concrete reason, at most 20 words"}}
The decision value must be exactly "A" or exactly "B", even when they appear tied."""


def step_plan_to_dict(step: StepPlan) -> dict[str, Any]:
    return {"goal": step.goal, "is_final": step.is_final}
