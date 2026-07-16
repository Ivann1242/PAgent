"""HintFlow agent: plan → execute → review/control → optional inject → next step."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import (  # noqa: E402
    build_large_prompt,
    exact_match,
    extract_final_answer,
    has_parseable_answer,
)

ORCH_URL = "http://127.0.0.1:8086/v1"
ORCH_MODEL = "qwen3-4b"
SOLVER_URL = "http://127.0.0.1:8006/v1"
SOLVER_MODEL = "gpt-oss-20b"

ACTIONS = ("NO_HINT", "INJECT", "RETRY", "REPLAN", "FINALIZE")
RUNTIME_MODES = ("legacy", "fresh", "structured", "retained")
VERDICTS = ("ACCEPT", "UNCERTAIN", "REJECT")

STEP_RESULT_SCHEMA = """Return ONLY valid JSON (no markdown fences):
{
  "result": "short, self-contained result of this step",
  "key_equations": ["only equations needed by later steps"],
  "candidate_answer": "answer if this step has one, otherwise empty",
  "is_complete": true/false,
  "confidence": "low | medium | high",
  "uncertainty": "specific unresolved concern, or empty"
}

Do the mathematical reasoning needed for the requested step, but keep `result`
compact and faithful. Never treat an unverified hypothesis as an established fact.
If is_complete=true, candidate_answer must be non-empty."""

SELECTOR_SYSTEM = """You are HintFlow's conservative candidate selector.
Compare two candidate math solutions for the same problem. Preserve the incumbent
unless the challenger is clearly more likely to have the correct requested answer.
Output ONLY valid JSON:
{"decision":"KEEP | REPLACE","confidence":0.0,"reason":"short concrete reason"}"""

PLAN_SYSTEM = """You are HintFlow's orchestrator (planner).
Given a math problem, design a customized multi-step plan for a stronger solver.

Output ONLY valid JSON (no markdown fences):
{
  "nodes": [
    {
      "instruction": "<what the solver should do in this step>",
      "inject_after": true/false,
      "is_final": false
    }
  ]
}

Rules:
- Customize the plan to THIS problem (step count and content may vary).
- inject_after is ONLY a soft prior (suggested review intensity), NOT a hard switch.
  The controller will decide NO_HINT/INJECT/RETRY/REPLAN/FINALIZE after seeing
  the solver response.
- Exactly one node must have is_final=true (usually the last); that step MUST
  explicitly ask the solver to end with: Final Answer: <answer>
- Prefer 2-4 nodes. Do not solve the problem yourself.
- HARD TURN BUDGET: the episode allows only a fixed number of solver turns
  (you will be told the remaining budget). Every step, RETRY and REPLAN each
  consume one turn. Your plan MUST fit within the remaining budget; when the
  budget runs out the system force-stops and demands a final answer."""

REVIEW_SYSTEM = """You are HintFlow's orchestrator (reviewer / controller).
After each solver step you must maintain long-horizon reasoning state and choose
the next control action.

Output ONLY valid JSON (no markdown fences):
{
  "summary": "Compressed state of progress so far (facts established)",
  "status": "correct | incomplete | incorrect",
  "issue": "Main mistake or unresolved bottleneck (or empty)",
  "hint": "1-2 sentence steering prompt for the next attempt/step, or empty",
  "action": "NO_HINT | INJECT | RETRY | REPLAN | FINALIZE"
}

Action meanings:
- NO_HINT: continue to the next planned step with no extra prompt
- INJECT: continue to the next step, and inject `hint` into that step's prompt
- RETRY: re-run the SAME step instruction, optionally with `hint`
- REPLAN: discard remaining planned steps; a new plan will be made; may use `hint`
- FINALIZE: stop ONLY if the solver response already contains an explicit final
  answer (`Final Answer: ...` or \\boxed{...})

Rules:
- Keep summary short but cumulative (state tracking).
- hint must NOT reveal the final answer or dump a full solution.
- Prefer INJECT/NO_HINT; use RETRY/REPLAN sparingly.
- NEVER FINALIZE on incomplete drafts / intermediate steps without an explicit answer.
- If is_last_planned_node and there is still no explicit answer, choose RETRY and
  hint the solver to output: Final Answer: <answer>
- Respect inject_prior only as a weak suggestion (true => lean toward INJECT if useful).
- HARD TURN BUDGET: you are told the current turn and how many solver turns remain.
  Every next step, RETRY and REPLAN each consume one turn. If remaining turns are
  low, steer toward finishing (get an explicit answer, then FINALIZE). When the
  budget hits zero the system force-stops and demands a final answer."""


@dataclass
class PlanNode:
    instruction: str
    inject_after: bool = False  # soft prior only
    is_final: bool = False


@dataclass
class Plan:
    nodes: list[PlanNode]
    raw: str = ""

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("empty plan")
        first_final = next(
            (i for i, node in enumerate(self.nodes) if node.is_final),
            len(self.nodes) - 1,
        )
        self.nodes = self.nodes[: first_final + 1]
        for node in self.nodes:
            node.is_final = False
        self.nodes[-1].is_final = True
        self.nodes[-1].inject_after = False
        last = self.nodes[-1]
        if "final answer" not in last.instruction.lower():
            last.instruction = (
                last.instruction.rstrip()
                + "\nEnd with the final answer as: Final Answer: <answer>"
            )


@dataclass
class StepReview:
    summary: str = ""
    status: str = "incomplete"
    issue: str = ""
    hint: str = ""
    action: str = "NO_HINT"
    raw: str = ""


@dataclass
class StepResult:
    result: str = ""
    key_equations: list[str] = field(default_factory=list)
    candidate_answer: str = ""
    is_complete: bool = False
    confidence: str = "medium"
    uncertainty: str = ""
    parse_ok: bool = True
    raw: str = ""

    def compact_text(self) -> str:
        parts = [self.result.strip()]
        if self.key_equations:
            parts.append("Key equations: " + "; ".join(self.key_equations))
        if self.candidate_answer:
            parts.append(f"Candidate answer: {self.candidate_answer}")
        if self.uncertainty:
            parts.append(f"Uncertainty: {self.uncertainty}")
        return "\n".join(part for part in parts if part).strip()


@dataclass
class StateEntry:
    step_index: int
    tier: str
    result: str
    key_equations: list[str] = field(default_factory=list)
    uncertainty: str = ""


@dataclass
class AnswerCandidate:
    index: int
    source: str
    solution: str
    answer: str
    parseable: bool
    confidence: str = "medium"
    step_index: int | None = None
    em: int | None = None
    selection_eligible: bool = False


@dataclass
class CandidateSelection:
    decision: str = "KEEP"
    confidence: float = 0.0
    reason: str = ""
    raw: str = ""


@dataclass
class StepRecord:
    index: int
    instruction: str
    observation: str
    review: StepReview | None = None
    injected_prompt: str = ""
    is_final: bool = False
    retried: bool = False
    step_result: StepResult | None = None
    verdict: str = "UNCERTAIN"
    state_tier: str = "candidate"


@dataclass
class Trajectory:
    problem: str
    plan: Plan | None = None
    steps: list[StepRecord] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    running_summary: str = ""
    final_answer: str = ""
    gold: str = ""
    em: int | None = None
    runtime_mode: str = "legacy"
    accepted_state: list[StateEntry] = field(default_factory=list)
    candidate_state: list[StateEntry] = field(default_factory=list)
    rejected_state: list[StateEntry] = field(default_factory=list)
    candidates: list[AnswerCandidate] = field(default_factory=list)
    selections: list[CandidateSelection] = field(default_factory=list)
    incumbent_index: int | None = None
    baseline_em: int | None = None
    oracle_em: int | None = None
    retry_count: int = 0
    replan_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _message_text(resp: Any) -> str:
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    if content:
        return content.strip()
    for name in ("reasoning", "reasoning_content"):
        reasoning = getattr(msg, name, None) or ""
        if str(reasoning).strip():
            return str(reasoning).strip()
    return ""


def _short_hint(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:2]).strip()


def parse_plan(text: str) -> Plan:
    raw = _strip_fences(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))

    nodes_raw = obj.get("nodes") or obj.get("steps") or []
    nodes: list[PlanNode] = []
    for item in nodes_raw:
        if isinstance(item, str):
            nodes.append(PlanNode(instruction=item))
            continue
        nodes.append(
            PlanNode(
                instruction=str(item.get("instruction") or item.get("content") or "").strip(),
                inject_after=bool(item.get("inject_after", False)),
                is_final=bool(item.get("is_final", False)),
            )
        )
    nodes = [n for n in nodes if n.instruction]
    plan = Plan(nodes=nodes, raw=raw)
    plan.validate()
    return plan


def parse_review(text: str) -> StepReview:
    raw = _strip_fences(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return StepReview(summary="", status="incomplete", action="NO_HINT", raw=raw)
        obj = json.loads(m.group(0))

    action = str(obj.get("action") or "NO_HINT").strip().upper().replace(" ", "_")
    # aliases
    if action in {"CONTINUE", "NONE", "SKIP"}:
        action = "NO_HINT"
    if action not in ACTIONS:
        action = "NO_HINT"

    hint = _short_hint(str(obj.get("hint") or ""))
    status = str(obj.get("status") or "incomplete").strip().lower()
    if status not in {"correct", "incomplete", "incorrect"}:
        status = "incomplete"

    return StepReview(
        summary=str(obj.get("summary") or "").strip(),
        status=status,
        issue=str(obj.get("issue") or "").strip(),
        hint=hint,
        action=action,
        raw=raw,
    )


def parse_step_result(text: str, *, structured: bool = True) -> StepResult:
    raw = (text or "").strip()
    if not structured:
        answer = extract_final_answer(raw) if has_parseable_answer(raw) else ""
        return StepResult(
            result=raw,
            candidate_answer=answer,
            is_complete=bool(answer),
            parse_ok=True,
            raw=raw,
        )

    clean = _strip_fences(raw)
    fallback_answer = extract_final_answer(raw) if has_parseable_answer(raw) else ""

    def malformed_fallback() -> StepResult:
        return StepResult(
            result=raw,
            candidate_answer=fallback_answer,
            is_complete=bool(fallback_answer),
            confidence="low",
            parse_ok=False,
            raw=raw,
        )

    try:
        obj = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return malformed_fallback()
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return malformed_fallback()
    if not isinstance(obj, dict):
        return malformed_fallback()

    equations_raw = obj.get("key_equations") or []
    if isinstance(equations_raw, str):
        equations_raw = [equations_raw]
    equations = [
        str(item).strip()[:500]
        for item in equations_raw
        if str(item).strip()
    ][:8]
    confidence = str(obj.get("confidence") or "medium").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    answer = str(obj.get("candidate_answer") or "").strip()
    complete_raw = obj.get("is_complete", False)
    complete = complete_raw if isinstance(complete_raw, bool) else False
    if complete and not answer and has_parseable_answer(raw):
        answer = extract_final_answer(raw)
    return StepResult(
        result=str(obj.get("result") or "").strip()[:4000],
        key_equations=equations,
        candidate_answer=answer[:1000],
        is_complete=complete and bool(answer),
        confidence=confidence,
        uncertainty=str(obj.get("uncertainty") or "").strip()[:1000],
        parse_ok=True,
        raw=raw,
    )


def verdict_from_review(review: StepReview) -> str:
    if review.status == "incorrect":
        return "REJECT"
    if review.status == "correct":
        return "ACCEPT"
    return "UNCERTAIN"


def parse_candidate_selection(text: str) -> CandidateSelection:
    raw = _strip_fences(text or "")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return CandidateSelection(raw=raw)
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return CandidateSelection(raw=raw)
    if not isinstance(obj, dict):
        return CandidateSelection(raw=raw)
    decision = str(obj.get("decision") or "KEEP").strip().upper()
    if decision not in {"KEEP", "REPLACE"}:
        decision = "KEEP"
    try:
        confidence = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return CandidateSelection(
        decision=decision,
        confidence=confidence,
        reason=str(obj.get("reason") or "").strip(),
        raw=raw,
    )


def fallback_plan(problem: str) -> Plan:
    return Plan(
        nodes=[
            PlanNode(
                instruction=(
                    "Solve the problem carefully. "
                    "Output the final answer as: Final Answer: <answer>"
                ),
                inject_after=False,
                is_final=True,
            )
        ],
        raw="fallback",
    )


def fallback_review(*, inject_prior: bool) -> StepReview:
    if inject_prior:
        return StepReview(
            summary="",
            status="incomplete",
            action="NO_HINT",
            hint="",
            raw="fallback",
        )
    return StepReview(action="NO_HINT", raw="fallback")


def build_solver_user_message(
    problem: str,
    instruction: str,
    *,
    injected_prompt: str = "",
    step_index: int,
    running_summary: str = "",
    force_final: bool = False,
) -> str:
    parts = [f"Problem:\n{problem}", ""]
    if running_summary.strip():
        parts.extend([
            "[Orchestrator state summary]",
            running_summary.strip(),
            "",
        ])
    parts.extend([f"Step {step_index + 1} instruction:\n{instruction}"])
    if injected_prompt.strip():
        parts.extend([
            "",
            "[Injected prompt from orchestrator]",
            injected_prompt.strip(),
        ])
    if force_final:
        parts.extend([
            "",
            "[FINAL TURN] This is the last allowed turn. Based on ALL the work "
            "above, you MUST commit to an answer NOW and end with exactly: "
            "Final Answer: <answer>",
        ])
    parts.extend([
        "",
        "Follow the instruction for this step only.",
        "If this step asks for the final answer, end with: Final Answer: <answer>",
    ])
    return "\n".join(parts)


def build_compact_solver_message(
    problem: str,
    instruction: str,
    *,
    accepted_state: list[StateEntry],
    candidate_state: list[StateEntry],
    injected_prompt: str = "",
    step_index: int,
    force_final: bool = False,
    structured: bool = True,
) -> str:
    parts = [f"Problem:\n{problem}", ""]
    if accepted_state:
        parts.append("[Accepted results — use as established state]")
        for i, entry in enumerate(accepted_state[-8:], 1):
            parts.append(f"{i}. {entry.result[:800]}")
            if entry.key_equations:
                parts.append("   Equations: " + "; ".join(entry.key_equations)[:800])
        parts.append("")
    if candidate_state:
        parts.append("[Open hypotheses — verify before using]")
        for entry in candidate_state[-3:]:
            text = entry.result[:500]
            if entry.uncertainty:
                text += f" (uncertainty: {entry.uncertainty[:250]})"
            parts.append(f"- {text}")
        parts.append("")
    parts.append(f"[Current goal — step {step_index + 1}]\n{instruction}")
    if injected_prompt.strip():
        parts.extend(["", f"[Orchestrator hint]\n{injected_prompt.strip()}"])
    if force_final and structured:
        parts.extend([
            "",
            "[FINAL TURN] Produce the best final answer now. Set is_complete=true "
            "and fill candidate_answer.",
        ])
    elif force_final:
        parts.extend([
            "",
            "[FINAL TURN] Produce the best final answer now and end with exactly: "
            "Final Answer: <answer>",
        ])
    if structured:
        parts.extend(["", STEP_RESULT_SCHEMA])
    else:
        parts.extend([
            "",
            "Return the result of this step. If complete, end with: "
            "Final Answer: <answer>",
        ])
    return "\n".join(parts)


def build_candidate_selection_prompt(
    problem: str,
    incumbent: AnswerCandidate,
    challenger: AnswerCandidate,
) -> str:
    return (
        f"Problem:\n{problem[:6000]}\n\n"
        f"[INCUMBENT]\n{incumbent.solution[:7000]}\n\n"
        f"[CHALLENGER]\n{challenger.solution[:7000]}\n\n"
        "Choose KEEP or REPLACE. Prefer KEEP when evidence is inconclusive."
    )


def format_turn_info(used_turns: int, max_turns: int) -> str:
    """Turn-budget broadcast shown to the orchestrator (part of train prompt)."""
    rem = max(max_turns - used_turns, 0)
    return (
        f"solver turns used: {used_turns}/{max_turns}; remaining: {rem}. "
        "Each next step, RETRY or REPLAN consumes one solver turn. "
        "When remaining hits 0 the system force-stops and demands the final answer."
    )


def build_plan_user(problem: str, *, context: str = "", turn_info: str = "") -> str:
    user = f"Problem:\n{problem}"
    if context.strip():
        user += f"\n\nContext from prior execution (replan):\n{context.strip()}"
    if turn_info.strip():
        user += f"\n\n[Turn budget]\n{turn_info.strip()}"
    return user


def format_history(history: list["StepRecord"]) -> str:
    hist_lines: list[str] = []
    for s in history[-4:]:
        hist_lines.append(f"- step{s.index + 1}: {s.instruction[:160]}")
        if s.review and s.review.summary:
            hist_lines.append(f"  summary: {s.review.summary[:200]}")
        if s.review and s.review.issue:
            hist_lines.append(f"  issue: {s.review.issue[:160]}")
        if s.injected_prompt:
            hist_lines.append(f"  injected: {s.injected_prompt}")
    return "\n".join(hist_lines) if hist_lines else "(none)"


def build_review_user(
    problem: str,
    *,
    instruction: str,
    observation: str,
    hist: str,
    running_summary: str,
    inject_prior: bool,
    is_last_planned: bool,
    turn_info: str = "",
) -> str:
    user = (
        f"Problem:\n{problem}\n\n"
        f"Running summary:\n{running_summary or '(empty)'}\n\n"
        f"Recent reviews:\n{hist or '(none)'}\n\n"
        f"Current instruction:\n{instruction}\n\n"
        f"Solver response:\n{observation}\n\n"
        f"inject_prior (soft): {str(inject_prior).lower()}\n"
        f"is_last_planned_node: {str(is_last_planned).lower()}\n"
    )
    if turn_info.strip():
        user += f"turn_budget: {turn_info.strip()}\n"
    user += "\nReturn the JSON review now."
    return user


class HintFlowAgent:
    """Trainable orchestrator loop + frozen solver (inference)."""

    def __init__(
        self,
        *,
        orch_url: str = ORCH_URL,
        orch_model: str = ORCH_MODEL,
        solver_url: str = SOLVER_URL,
        solver_model: str = SOLVER_MODEL,
        max_nodes: int = 6,
        max_steps: int = 7,
        max_retries_per_node: int = 1,
        orch_temperature: float = 0.7,
        solver_temperature: float = 0.0,
        orch_max_tokens: int = 1024,
        solver_max_tokens: int = 8192,
        runtime_mode: str = "retained",
        max_retries_total: int = 1,
        max_replans_total: int = 1,
        replace_threshold: float = 0.90,
        min_replace_support: int = 2,
        solver_seed: int | None = None,
    ) -> None:
        if runtime_mode not in RUNTIME_MODES:
            raise ValueError(f"runtime_mode must be one of {RUNTIME_MODES}")
        self.orch = OpenAI(base_url=orch_url, api_key="EMPTY")
        self.solver = OpenAI(base_url=solver_url, api_key="EMPTY")
        self.orch_model = orch_model
        self.solver_model = solver_model
        self.max_nodes = max_nodes
        self.max_steps = max_steps
        self.max_retries_per_node = max_retries_per_node
        self.orch_temperature = orch_temperature
        self.solver_temperature = solver_temperature
        self.orch_max_tokens = orch_max_tokens
        self.solver_max_tokens = solver_max_tokens
        self.runtime_mode = runtime_mode
        self.max_retries_total = max(0, max_retries_total)
        self.max_replans_total = max(0, max_replans_total)
        self.replace_threshold = min(max(replace_threshold, 0.0), 1.0)
        self.min_replace_support = max(1, min_replace_support)
        self.solver_seed = solver_seed
        self._solver_call_index = 0

    def _orch_chat(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        resp = self.orch.chat.completions.create(
            model=self.orch_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.orch_temperature,
            max_tokens=max_tokens or self.orch_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _message_text(resp)

    # ------------------------------------------------------------------ plan
    def plan(self, problem: str, *, context: str = "", turn_info: str = "") -> Plan:
        """Customized instruction plan. inject_after is a soft prior only."""
        plan, _raw = self.sample_plan(problem, context=context, turn_info=turn_info)
        return plan

    # ------------------------------------------------- review_and_control
    def review_and_control(
        self,
        problem: str,
        *,
        instruction: str,
        observation: str,
        history: list[StepRecord],
        running_summary: str,
        inject_prior: bool,
        is_last_planned: bool,
        turn_info: str = "",
    ) -> StepReview:
        """Summarize state, diagnose issues, choose action (+ optional hint)."""
        review, _raw = self.sample_review(
            problem,
            instruction=instruction,
            observation=observation,
            history=history,
            running_summary=running_summary,
            inject_prior=inject_prior,
            is_last_planned=is_last_planned,
            turn_info=turn_info,
        )
        return review

    def _enforce_finalize_gate(
        self,
        review: StepReview,
        *,
        observation: str,
        is_last_planned: bool,
    ) -> StepReview:
        """Step1: never FINALIZE without a parseable short answer."""
        parseable = has_parseable_answer(observation)

        if review.action == "FINALIZE" and not parseable:
            if is_last_planned:
                review.action = "RETRY"
                if not review.hint:
                    review.hint = (
                        "Provide the final numerical/symbolic answer now as: "
                        "Final Answer: <answer>"
                    )
            else:
                review.action = "NO_HINT"

        if is_last_planned and review.action in {"NO_HINT", "INJECT", "REPLAN"}:
            if parseable:
                review.action = "FINALIZE"
            else:
                review.action = "RETRY"
                if not review.hint:
                    review.hint = (
                        "Provide the final numerical/symbolic answer now as: "
                        "Final Answer: <answer>"
                    )
        return review

    def sample_plan(
        self,
        problem: str,
        *,
        context: str = "",
        turn_info: str = "",
        avoid_texts: list[str] | None = None,
    ) -> tuple[Plan, str]:
        """One planner sample; returns (plan, raw_orch_text).

        avoid_texts: data-collection-only exploration constraint (sibling outputs
        to diverge from). Appended AFTER the clean prompt; training must use the
        clean prompt (build_plan_user) without this block.
        """
        user = build_plan_user(problem, context=context, turn_info=turn_info)
        if avoid_texts:
            prev = "\n\n".join(
                f"[Previous sample {i + 1}]\n{t[:800]}" for i, t in enumerate(avoid_texts)
            )
            user += (
                "\n\n[Exploration constraint — data collection only]\n"
                "Another sample already produced the plan(s) below. Output a DISTINCTLY "
                "different valid plan JSON: use a different decomposition, step count, "
                "or solution strategy. Do NOT repeat or lightly rephrase them.\n" + prev
            )
        last_text = ""
        for _ in range(2):
            last_text = self._orch_chat(PLAN_SYSTEM, user)
            try:
                plan = parse_plan(last_text)
                if len(plan.nodes) > self.max_nodes:
                    plan.nodes = plan.nodes[: self.max_nodes]
                    plan.validate()
                return plan, last_text
            except Exception:
                continue
        plan = fallback_plan(problem)
        return plan, last_text or plan.raw

    def sample_review(
        self,
        problem: str,
        *,
        instruction: str,
        observation: str,
        history: list[StepRecord],
        running_summary: str,
        inject_prior: bool,
        is_last_planned: bool,
        turn_info: str = "",
        avoid_texts: list[str] | None = None,
    ) -> tuple[StepReview, str]:
        """One controller sample; returns (review, raw_orch_text).

        avoid_texts: data-collection-only exploration constraint (sibling outputs
        to diverge from). Appended AFTER the clean prompt; training must use the
        clean prompt (build_review_user) without this block.
        """
        user = build_review_user(
            problem,
            instruction=instruction,
            observation=observation,
            hist=format_history(history),
            running_summary=running_summary,
            inject_prior=inject_prior,
            is_last_planned=is_last_planned,
            turn_info=turn_info,
        )
        if avoid_texts:
            prev = "\n\n".join(
                f"[Previous sample {i + 1}]\n{t[:800]}" for i, t in enumerate(avoid_texts)
            )
            user += (
                "\n\n[Exploration constraint — data collection only]\n"
                "Another sample already produced the review(s) below for this exact state. "
                "Output a DISTINCTLY different valid review JSON: choose a different action "
                "if reasonable, or a materially different hint/diagnosis. Do NOT repeat or "
                "lightly rephrase them.\n" + prev
            )
        try:
            text = self._orch_chat(REVIEW_SYSTEM, user, max_tokens=512)
            review = parse_review(text)
        except Exception:
            text = ""
            review = fallback_review(inject_prior=inject_prior)
        review = self._enforce_finalize_gate(
            review, observation=observation, is_last_planned=is_last_planned
        )
        return review, text or review.raw

    # -------------------------------------------------------------- reason
    def _solver_step(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.solver_model,
            "messages": messages,
            "temperature": self.solver_temperature,
            "max_tokens": self.solver_max_tokens,
        }
        if self.solver_seed is not None:
            kwargs["extra_body"] = {
                "seed": int(self.solver_seed + self._solver_call_index)
            }
        self._solver_call_index += 1
        resp = self.solver.chat.completions.create(
            **kwargs,
        )
        return _message_text(resp)

    def _answer_candidate(
        self,
        *,
        index: int,
        source: str,
        solution: str,
        gold: str,
        confidence: str = "medium",
        step_index: int | None = None,
        explicit_answer: str = "",
        selection_eligible: bool = False,
    ) -> AnswerCandidate:
        parseable = bool(explicit_answer) or has_parseable_answer(solution)
        answer = explicit_answer or (
            extract_final_answer(solution) if parseable else ""
        )
        return AnswerCandidate(
            index=index,
            source=source,
            solution=solution,
            answer=answer,
            parseable=parseable,
            confidence=confidence,
            step_index=step_index,
            em=exact_match(answer, gold) if gold and answer else (0 if gold else None),
            selection_eligible=selection_eligible,
        )

    def _select_candidate(
        self,
        problem: str,
        incumbent: AnswerCandidate,
        challenger: AnswerCandidate,
    ) -> CandidateSelection:
        if not challenger.parseable:
            return CandidateSelection(reason="challenger has no parseable answer")
        if not incumbent.parseable:
            return CandidateSelection(
                decision="REPLACE",
                confidence=1.0,
                reason="incumbent has no parseable answer",
            )
        if exact_match(challenger.answer, incumbent.answer):
            return CandidateSelection(reason="same normalized candidate answer")
        try:
            text = self._orch_chat(
                SELECTOR_SYSTEM,
                build_candidate_selection_prompt(problem, incumbent, challenger),
                max_tokens=192,
            )
            return parse_candidate_selection(text)
        except Exception as exc:
            return CandidateSelection(
                reason=(
                    "selector failure; conservatively kept incumbent: "
                    f"{type(exc).__name__}"
                )
            )

    def _generate_baseline(self, problem: str, *, gold: str) -> AnswerCandidate:
        # Match eval_hintflow's live native baseline exactly.
        prompt = build_large_prompt(problem, "")
        solution = self._solver_step([{"role": "user", "content": prompt}])
        return self._answer_candidate(
            index=0,
            source="BASELINE",
            solution=solution,
            gold=gold,
            selection_eligible=True,
        )

    def reason(self, problem: str, plan: Plan | None = None, *, gold: str = "") -> Trajectory:
        if self.runtime_mode != "legacy":
            return self._reason_stateful(problem, plan, gold=gold)
        return self._reason_legacy(problem, plan, gold=gold)

    def _reason_legacy(
        self,
        problem: str,
        plan: Plan | None = None,
        *,
        gold: str = "",
    ) -> Trajectory:
        """
        Execute plan with dynamic control after each observation:
        NO_HINT | INJECT | RETRY | REPLAN | FINALIZE.
        Hint is injected at most once via pending_inject into the next user message.
        """
        traj = Trajectory(problem=problem, gold=gold, runtime_mode="legacy")
        traj.plan = plan or self.plan(
            problem, turn_info=format_turn_info(0, self.max_steps)
        )
        nodes = list(traj.plan.nodes)
        pending_inject = ""
        i = 0
        retries_on_node = 0
        step_counter = 0

        while i < len(nodes) and step_counter < self.max_steps:
            node = nodes[i]
            is_last_planned = i >= len(nodes) - 1 or node.is_final
            is_last_turn = step_counter >= self.max_steps - 1

            user_msg = build_solver_user_message(
                problem,
                node.instruction,
                injected_prompt=pending_inject,
                step_index=step_counter,
                running_summary=traj.running_summary,
                force_final=is_last_turn,
            )
            used_inject = pending_inject
            pending_inject = ""

            traj.messages.append({"role": "user", "content": user_msg})
            observation = self._solver_step(traj.messages)
            traj.messages.append({"role": "assistant", "content": observation})

            review = self.review_and_control(
                problem,
                instruction=node.instruction,
                observation=observation,
                history=traj.steps,
                running_summary=traj.running_summary,
                inject_prior=node.inject_after,
                is_last_planned=is_last_planned,
                turn_info=format_turn_info(step_counter + 1, self.max_steps),
            )
            if review.summary:
                traj.running_summary = review.summary

            step = StepRecord(
                index=step_counter,
                instruction=node.instruction,
                observation=observation,
                review=review,
                injected_prompt=used_inject,
                is_final=False,
                retried=retries_on_node > 0,
            )
            traj.steps.append(step)
            step_counter += 1

            action = review.action

            if action == "FINALIZE":
                step.is_final = True
                break

            if action == "RETRY" and retries_on_node < self.max_retries_per_node:
                retries_on_node += 1
                pending_inject = review.hint  # single injection path only
                continue

            # Exhausted RETRY on last node without parseable answer: stop anyway.
            if action == "RETRY" and retries_on_node >= self.max_retries_per_node:
                step.is_final = True
                break

            if action == "REPLAN":
                ctx = (
                    f"Running summary:\n{traj.running_summary}\n"
                    f"Last issue:\n{review.issue}\n"
                    f"Last observation (truncated):\n{observation[:800]}"
                )
                new_plan = self.plan(
                    problem,
                    context=ctx,
                    turn_info=format_turn_info(step_counter, self.max_steps),
                )
                # Replace remaining queue (including current) with new plan.
                nodes = nodes[:i] + new_plan.nodes
                traj.plan.nodes = list(nodes)
                pending_inject = review.hint
                retries_on_node = 0
                # Stay on index i to execute first node of the new tail.
                continue

            if action == "INJECT":
                pending_inject = review.hint
            else:
                # NO_HINT or exhausted RETRY
                pending_inject = ""

            retries_on_node = 0
            i += 1
            # Do not hard-stop on is_final without FINALIZE gate (handled above).

        if traj.steps:
            traj.steps[-1].is_final = True
            traj.final_answer = extract_final_answer(traj.steps[-1].observation)
        if gold:
            traj.em = exact_match(traj.final_answer, gold)
        return traj

    def _reason_stateful(
        self,
        problem: str,
        plan: Plan | None = None,
        *,
        gold: str = "",
    ) -> Trajectory:
        """Fresh-context V2 with structured state and optional candidate retention."""
        structured = self.runtime_mode in {"structured", "retained"}
        retained = self.runtime_mode == "retained"
        traj = Trajectory(problem=problem, gold=gold, runtime_mode=self.runtime_mode)

        step_counter = 0
        if retained and self.max_steps > 0:
            baseline = self._generate_baseline(problem, gold=gold)
            traj.candidates.append(baseline)
            traj.incumbent_index = 0
            step_counter = 1

        traj.plan = plan or self.plan(
            problem, turn_info=format_turn_info(step_counter, self.max_steps)
        )
        nodes = list(traj.plan.nodes)
        pending_inject = ""
        node_i = 0
        retries_on_node = 0

        while node_i < len(nodes) and step_counter < self.max_steps:
            node = nodes[node_i]
            is_last_planned = node_i >= len(nodes) - 1 or node.is_final
            is_last_turn = step_counter >= self.max_steps - 1
            if structured:
                prompt_accepted = traj.accepted_state
                prompt_candidates = traj.candidate_state
            else:
                prompt_accepted = (
                    [
                        StateEntry(
                            step_index=max(step_counter - 1, 0),
                            tier="accepted",
                            result=traj.running_summary,
                        )
                    ]
                    if traj.running_summary
                    else []
                )
                prompt_candidates = []
            user_msg = build_compact_solver_message(
                problem,
                node.instruction,
                accepted_state=prompt_accepted,
                candidate_state=prompt_candidates,
                injected_prompt=pending_inject,
                step_index=step_counter,
                force_final=is_last_turn,
                structured=structured,
            )
            used_inject = pending_inject
            pending_inject = ""
            # Every solver call receives a fresh, single-turn context.
            observation = self._solver_step([{"role": "user", "content": user_msg}])
            traj.messages.extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": observation},
            ])
            result = parse_step_result(observation, structured=structured)
            review_observation = result.compact_text() if structured else observation
            if structured and result.is_complete and result.candidate_answer:
                review_observation += (
                    f"\nFinal Answer: {result.candidate_answer}"
                )
            review = self.review_and_control(
                problem,
                instruction=node.instruction,
                observation=review_observation,
                history=traj.steps,
                running_summary=traj.running_summary,
                inject_prior=node.inject_after,
                is_last_planned=is_last_planned,
                turn_info=format_turn_info(step_counter + 1, self.max_steps),
            )
            verdict = verdict_from_review(review)
            tier = {
                "ACCEPT": "accepted",
                "UNCERTAIN": "candidate",
                "REJECT": "rejected",
            }[verdict]
            entry = StateEntry(
                step_index=step_counter,
                tier=tier,
                result=result.result or review.summary or review_observation[:1200],
                key_equations=result.key_equations,
                uncertainty=result.uncertainty or review.issue,
            )
            getattr(traj, f"{tier}_state").append(entry)
            if review.summary:
                traj.running_summary = review.summary

            step = StepRecord(
                index=step_counter,
                instruction=node.instruction,
                observation=observation,
                review=review,
                injected_prompt=used_inject,
                retried=retries_on_node > 0,
                step_result=result,
                verdict=verdict,
                state_tier=tier,
            )
            traj.steps.append(step)
            step_counter += 1

            candidate_answer = result.candidate_answer
            if not structured and has_parseable_answer(observation):
                candidate_answer = extract_final_answer(observation)
            if candidate_answer:
                eligible = not structured or result.is_complete
                selection_eligible = eligible and (
                    not structured or result.confidence == "high"
                )
                candidate = self._answer_candidate(
                    index=len(traj.candidates),
                    source="STEP",
                    solution=observation,
                    explicit_answer=candidate_answer,
                    confidence=result.confidence,
                    step_index=step.index,
                    gold=gold,
                    selection_eligible=selection_eligible,
                )
                traj.candidates.append(candidate)
                if eligible:
                    if traj.incumbent_index is None:
                        traj.incumbent_index = candidate.index
                    elif not retained:
                        traj.incumbent_index = candidate.index
                    else:
                        incumbent = traj.candidates[traj.incumbent_index]
                        if not incumbent.parseable and candidate.selection_eligible:
                            selection = CandidateSelection(
                                decision="REPLACE",
                                confidence=1.0,
                                reason=(
                                    "complete high-confidence challenger replaces "
                                    "an unparseable incumbent"
                                ),
                            )
                        elif verdict == "ACCEPT" and candidate.selection_eligible:
                            support = sum(
                                1
                                for other in traj.candidates
                                if (
                                    other.source == "STEP"
                                    and other.selection_eligible
                                    and exact_match(other.answer, candidate.answer)
                                )
                            )
                            if support < self.min_replace_support:
                                selection = CandidateSelection(
                                    reason=(
                                        "consensus guard kept parseable incumbent: "
                                        f"support={support}/{self.min_replace_support}"
                                    )
                                )
                            else:
                                selection = self._select_candidate(
                                    problem, incumbent, candidate
                                )
                        else:
                            selection = CandidateSelection(
                                reason=(
                                    "challenger did not pass completeness, confidence, "
                                    "or reviewer acceptance gate"
                                )
                            )
                        traj.selections.append(selection)
                        if (
                            selection.decision == "REPLACE"
                            and selection.confidence >= self.replace_threshold
                        ):
                            traj.incumbent_index = candidate.index

            action = review.action
            if action == "FINALIZE":
                step.is_final = True
                break
            if (
                action == "RETRY"
                and retries_on_node < self.max_retries_per_node
                and traj.retry_count < self.max_retries_total
                and step_counter < self.max_steps
            ):
                retries_on_node += 1
                traj.retry_count += 1
                pending_inject = review.hint
                continue
            if (
                action == "REPLAN"
                and traj.replan_count < self.max_replans_total
                and step_counter < self.max_steps
            ):
                traj.replan_count += 1
                context = (
                    f"Accepted state:\n"
                    + "\n".join(f"- {e.result}" for e in traj.accepted_state[-8:])
                    + f"\nLast issue:\n{review.issue or result.uncertainty}"
                )
                new_plan = self.plan(
                    problem,
                    context=context,
                    turn_info=format_turn_info(step_counter, self.max_steps),
                )
                nodes = nodes[:node_i] + new_plan.nodes
                traj.plan.nodes = list(nodes)
                pending_inject = review.hint
                retries_on_node = 0
                continue

            if action == "INJECT":
                pending_inject = review.hint
            retries_on_node = 0
            node_i += 1

        if traj.steps:
            traj.steps[-1].is_final = True
        if traj.incumbent_index is not None:
            incumbent = traj.candidates[traj.incumbent_index]
            traj.final_answer = incumbent.answer
        elif traj.steps:
            last_result = traj.steps[-1].step_result
            if (
                last_result
                and last_result.candidate_answer
                and (not structured or last_result.is_complete)
            ):
                traj.final_answer = last_result.candidate_answer
            elif has_parseable_answer(traj.steps[-1].observation):
                traj.final_answer = extract_final_answer(traj.steps[-1].observation)
        if gold:
            traj.em = exact_match(traj.final_answer, gold) if traj.final_answer else 0
            if traj.candidates:
                traj.baseline_em = (
                    int(traj.candidates[0].em or 0)
                    if traj.candidates[0].source == "BASELINE"
                    else None
                )
                traj.oracle_em = max(int(c.em or 0) for c in traj.candidates)
        return traj

    def run(self, problem: str, *, gold: str = "") -> Trajectory:
        if self.runtime_mode == "legacy":
            return self.reason(problem, self.plan(problem), gold=gold)
        return self.reason(problem, gold=gold)

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Placeholder: trajectory SFT / RL later (not DSPy)."""
        raise NotImplementedError(
            "HintFlow train() not implemented — use trajectory SFT/RL, not DSPy"
        )


def _demo() -> None:
    data_path = _ROOT / "data" / "train.jsonl"
    with data_path.open(encoding="utf-8") as f:
        row = json.loads(next(line for line in f if line.strip()))

    agent = HintFlowAgent()
    print("=== PLAN ===")
    plan = agent.plan(row["problem"])
    for i, n in enumerate(plan.nodes):
        print(f"[{i}] prior_inject={n.inject_after} final={n.is_final}")
        print(f"    {n.instruction[:200]}")

    print("\n=== REASON ===")
    traj = agent.reason(row["problem"], plan, gold=row.get("gold", ""))
    print("running_summary:", traj.running_summary)
    for s in traj.steps:
        r = s.review
        print(f"\n-- step {s.index} action={r.action if r else None} status={r.status if r else None} --")
        if r and r.summary:
            print("SUMMARY:", r.summary[:240])
        if r and r.issue:
            print("ISSUE:", r.issue[:200])
        if s.injected_prompt:
            print("INJECTED_IN_THIS_STEP:", s.injected_prompt)
        if r and r.hint and r.action == "INJECT":
            print("HINT->NEXT:", r.hint)
        print("OBS:", s.observation[:320].replace("\n", " "))
    print("\nFINAL:", traj.final_answer)
    print("EM:", traj.em)


if __name__ == "__main__":
    _demo()
