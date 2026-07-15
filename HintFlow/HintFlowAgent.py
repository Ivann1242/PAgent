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

from core import exact_match, extract_final_answer, has_parseable_answer  # noqa: E402

ORCH_URL = "http://127.0.0.1:8086/v1"
ORCH_MODEL = "qwen3-4b"
SOLVER_URL = "http://127.0.0.1:8006/v1"
SOLVER_MODEL = "gpt-oss-20b"

ACTIONS = ("NO_HINT", "INJECT", "RETRY", "REPLAN", "FINALIZE")

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
class StepRecord:
    index: int
    instruction: str
    observation: str
    review: StepReview | None = None
    injected_prompt: str = ""
    is_final: bool = False
    retried: bool = False


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
    reasoning = getattr(msg, "reasoning_content", None) or ""
    return str(reasoning).strip()


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
    ) -> None:
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
        resp = self.solver.chat.completions.create(
            model=self.solver_model,
            messages=messages,
            temperature=self.solver_temperature,
            max_tokens=self.solver_max_tokens,
        )
        return _message_text(resp)

    def reason(self, problem: str, plan: Plan | None = None, *, gold: str = "") -> Trajectory:
        """
        Execute plan with dynamic control after each observation:
        NO_HINT | INJECT | RETRY | REPLAN | FINALIZE.
        Hint is injected at most once via pending_inject into the next user message.
        """
        traj = Trajectory(problem=problem, gold=gold)
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

    def run(self, problem: str, *, gold: str = "") -> Trajectory:
        return self.reason(problem, self.plan(problem), gold=gold)

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
