from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from core import (
    call_llm,
    exact_match,
    extract_final_answer,
    has_parseable_answer,
    parse_optimizer_output,
)
from HintFlow_draft.common import (
    PLANNER_SYSTEM,
    SELECTOR_SYSTEM,
    AcceptedStep,
    StepPlan,
    blind_candidates,
    build_draft_prompt,
    build_goal_step_prompt,
    build_segment_prompt,
    build_selector_prompt,
    build_step_hint_prompt,
    fallback_plan,
    parse_selector_output,
    parse_step_plan,
    prefer_draft_consistent,
    resolve_blinded_selection,
    sanitize_stub_leaked_plan,
    step_plan_to_dict,
)


@dataclass
class DraftStepRecord:
    step_index: int
    goal: str
    is_final: bool
    hint_prompt: str = ""
    hint_raw: str = ""
    hint: str = ""
    hint_parse_ok: bool = False
    baseline_prompt: str = ""
    baseline: str = ""
    hinted_prompt: str = ""
    hinted: str = ""
    candidate_mapping: dict[str, str] = field(default_factory=dict)
    selector_prompt: str = ""
    selector_raw: str = ""
    selector_label: str = ""
    decision: str = "BASELINE"
    confidence: float = 0.0
    reason: str = ""
    selector_parse_ok: bool = True
    selected: str = ""
    selected_has_answer: bool = False
    premature_answer: bool = False
    draft_vote: bool = False


@dataclass
class DraftTrajectory:
    problem: str
    gold: str = ""
    mode: str = "full"
    draft_prompt: str = ""
    draft: str = ""
    draft_answer: str = ""
    draft_parseable: bool = False
    draft_em: int | None = None
    plan_raw: str = ""
    plan: list[dict[str, Any]] = field(default_factory=list)
    plan_parse_ok: bool = False
    plan_fallback: bool = False
    plan_sanitized: bool = False
    plan_error: str = ""
    steps: list[DraftStepRecord] = field(default_factory=list)
    final_answer: str = ""
    final_parseable: bool = False
    em: int | None = None
    recovered: int = 0
    harmed: int = 0
    n_large_calls: int = 0
    n_small_calls: int = 0
    calls_by_role: dict[str, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DraftStepAgent:
    def __init__(
        self,
        *,
        small_url: str,
        small_model: str,
        solver_url: str,
        solver_model: str,
        planner_url: str | None = None,
        planner_model: str | None = None,
        hinter_url: str | None = None,
        hinter_model: str | None = None,
        selector_url: str | None = None,
        selector_model: str | None = None,
        max_steps: int = 8,
        draft_max_tokens: int = 8192,
        step_max_tokens: int = 2048,
        planner_max_tokens: int = 768,
        hint_max_tokens: int = 256,
        selector_max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 0,
        request_timeout: float = 3600.0,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.planner = OpenAI(
            base_url=planner_url or small_url,
            api_key="EMPTY",
            timeout=request_timeout,
            max_retries=0,
        )
        self.hinter = OpenAI(
            base_url=hinter_url or small_url,
            api_key="EMPTY",
            timeout=request_timeout,
            max_retries=0,
        )
        self.selector = OpenAI(
            base_url=selector_url or small_url,
            api_key="EMPTY",
            timeout=request_timeout,
            max_retries=0,
        )
        self.solver = OpenAI(
            base_url=solver_url,
            api_key="EMPTY",
            timeout=request_timeout,
            max_retries=0,
        )
        self.planner_model = planner_model or small_model
        self.hinter_model = hinter_model or small_model
        self.selector_model = selector_model or small_model
        self.solver_model = solver_model
        self.max_steps = max_steps
        self.draft_max_tokens = draft_max_tokens
        self.step_max_tokens = step_max_tokens
        self.planner_max_tokens = planner_max_tokens
        self.hint_max_tokens = hint_max_tokens
        self.selector_max_tokens = selector_max_tokens
        self.temperature = temperature
        self.seed = seed
        self._call_index = 0
        self._large_calls = 0
        self._small_calls = 0
        self._calls_by_role: dict[str, int] = {}

    def _reset_counters(self) -> None:
        self._call_index = 0
        self._large_calls = 0
        self._small_calls = 0
        self._calls_by_role = {}

    def _call(
        self,
        client: OpenAI,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        role: str,
        is_large: bool,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        call_seed = self.seed + self._call_index
        self._call_index += 1
        if is_large:
            self._large_calls += 1
        else:
            self._small_calls += 1
        self._calls_by_role[role] = self._calls_by_role.get(role, 0) + 1
        return call_llm(
            client,
            model,
            prompt,
            system=system,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            seed=call_seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).strip()

    def generate_draft(self, problem: str) -> tuple[str, str]:
        prompt = build_draft_prompt(problem)
        output = self._call(
            self.solver,
            self.solver_model,
            prompt,
            max_tokens=self.draft_max_tokens,
            role="draft",
            is_large=True,
        )
        return prompt, output

    def segment_draft(
        self, problem: str, draft: str
    ) -> tuple[list[StepPlan], str, bool, str]:
        prompt = build_segment_prompt(problem, draft, max_steps=self.max_steps)
        raw = self._call(
            self.planner,
            self.planner_model,
            prompt,
            max_tokens=self.planner_max_tokens,
            role="planner",
            is_large=False,
            system=PLANNER_SYSTEM,
            temperature=0.0,
        )
        try:
            plan = parse_step_plan(raw, max_steps=self.max_steps)
            plan, sanitized = sanitize_stub_leaked_plan(plan)
            error = "sanitized stub leaked last goal" if sanitized else ""
            return plan, raw, True, error
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            return fallback_plan(), raw, False, error

    def generate_hint(
        self,
        problem: str,
        accepted: list[AcceptedStep],
        goal: str,
    ) -> tuple[str, str, str, bool]:
        prompt = build_step_hint_prompt(problem, accepted, goal)
        raw = self._call(
            self.hinter,
            self.hinter_model,
            prompt,
            max_tokens=self.hint_max_tokens,
            role="hint",
            is_large=False,
        )
        hint, parse_ok = parse_optimizer_output(raw)
        return prompt, raw, hint, parse_ok

    def generate_step_candidate(
        self,
        problem: str,
        accepted: list[AcceptedStep],
        step: StepPlan,
        *,
        hint: str = "",
        source: str,
    ) -> tuple[str, str]:
        prompt = build_goal_step_prompt(
            problem,
            accepted,
            step.goal,
            hint=hint,
            is_final=step.is_final,
        )
        output = self._call(
            self.solver,
            self.solver_model,
            prompt,
            max_tokens=self.step_max_tokens,
            role=source.lower(),
            is_large=True,
        )
        return prompt, output

    def select_step(
        self,
        problem: str,
        accepted: list[AcceptedStep],
        step: StepPlan,
        baseline: str,
        hinted: str,
        *,
        step_index: int,
        draft: str = "",
        plan_goals: list[str] | None = None,
    ) -> tuple[
        str,
        float,
        str,
        bool,
        str,
        str,
        dict[str, str],
        str,
    ]:
        candidate_a, candidate_b, mapping = blind_candidates(
            baseline,
            hinted,
            seed=self.seed,
            step_index=step_index,
        )
        prompt = build_selector_prompt(
            problem,
            accepted,
            step.goal,
            candidate_a,
            candidate_b,
            is_final=step.is_final,
        )
        if not baseline.strip() and hinted.strip():
            label = next(k for k, source in mapping.items() if source == "HINTED")
            return (
                "HINTED",
                1.0,
                "baseline candidate is empty",
                True,
                label,
                "",
                mapping,
                prompt,
            )
        if not hinted.strip():
            label = next(k for k, source in mapping.items() if source == "BASELINE")
            return (
                "BASELINE",
                1.0,
                "hinted candidate is empty",
                True,
                label,
                "",
                mapping,
                prompt,
            )

        forced = prefer_draft_consistent(
            baseline,
            hinted,
            draft,
            is_final=step.is_final,
            goal=step.goal,
            plan_goals=plan_goals,
        )
        if forced is not None:
            source, reason = forced
            label = next(k for k, name in mapping.items() if name == source)
            return source, 1.0, reason, True, label, "", mapping, prompt

        raw = self._call(
            self.selector,
            self.selector_model,
            prompt,
            max_tokens=self.selector_max_tokens,
            role="selector",
            is_large=False,
            system=SELECTOR_SYSTEM,
            temperature=0.0,
        )
        try:
            label, confidence, reason = parse_selector_output(raw)
            source = resolve_blinded_selection(label, mapping)
            return source, confidence, reason, True, label, raw, mapping, prompt
        except Exception as exc:  # noqa: BLE001
            label = next(k for k, source in mapping.items() if source == "BASELINE")
            reason = f"selector parse failure; chose baseline: {type(exc).__name__}"
            return "BASELINE", 0.0, reason, False, label, raw, mapping, prompt

    def run(
        self,
        problem: str,
        *,
        gold: str = "",
        mode: str = "full",
    ) -> DraftTrajectory:
        if mode not in {"full", "baseline"}:
            raise ValueError("mode must be full or baseline")
        self._reset_counters()
        started = time.monotonic()
        traj = DraftTrajectory(problem=problem, gold=gold, mode=mode)

        draft_prompt, draft = self.generate_draft(problem)
        traj.draft_prompt = draft_prompt
        traj.draft = draft
        traj.draft_parseable = has_parseable_answer(draft)
        traj.draft_answer = (
            extract_final_answer(draft) if traj.draft_parseable else ""
        )
        traj.draft_em = (
            exact_match(traj.draft_answer, gold)
            if gold and traj.draft_parseable
            else (0 if gold else None)
        )

        plan, plan_raw, plan_ok, plan_error = self.segment_draft(problem, draft)
        traj.plan_raw = plan_raw
        traj.plan = [step_plan_to_dict(step) for step in plan]
        traj.plan_parse_ok = plan_ok
        traj.plan_fallback = not plan_ok
        traj.plan_sanitized = plan_ok and plan_error.startswith("sanitized")
        traj.plan_error = plan_error

        accepted: list[AcceptedStep] = []
        for step_index, step in enumerate(plan, 1):
            record = DraftStepRecord(
                step_index=step_index,
                goal=step.goal,
                is_final=step.is_final,
            )
            if mode == "full":
                (
                    record.hint_prompt,
                    record.hint_raw,
                    record.hint,
                    record.hint_parse_ok,
                ) = self.generate_hint(problem, accepted, step.goal)
                record.hinted_prompt, record.hinted = self.generate_step_candidate(
                    problem,
                    accepted,
                    step,
                    hint=record.hint,
                    source="HINTED",
                )

            record.baseline_prompt, record.baseline = self.generate_step_candidate(
                problem,
                accepted,
                step,
                source="BASELINE",
            )

            if mode == "full":
                (
                    record.decision,
                    record.confidence,
                    record.reason,
                    record.selector_parse_ok,
                    record.selector_label,
                    record.selector_raw,
                    record.candidate_mapping,
                    record.selector_prompt,
                ) = self.select_step(
                    problem,
                    accepted,
                    step,
                    record.baseline,
                    record.hinted,
                    step_index=step_index,
                    draft=draft,
                    plan_goals=[item.goal for item in plan],
                )
            else:
                record.decision = "BASELINE"
                record.confidence = 1.0
                record.reason = "baseline mode"
                record.selector_parse_ok = True

            record.selected = (
                record.hinted if record.decision == "HINTED" else record.baseline
            )
            record.selected_has_answer = has_parseable_answer(record.selected)
            record.premature_answer = (
                not step.is_final and record.selected_has_answer
            )
            record.draft_vote = record.reason.startswith(
                "last-step draft answer matches"
            )
            accepted.append(AcceptedStep(goal=step.goal, result=record.selected))
            traj.steps.append(record)

        final_output = traj.steps[-1].selected if traj.steps else ""
        traj.final_parseable = has_parseable_answer(final_output)
        traj.final_answer = (
            extract_final_answer(final_output) if traj.final_parseable else ""
        )
        traj.em = (
            exact_match(traj.final_answer, gold)
            if gold and traj.final_parseable
            else (0 if gold else None)
        )
        if gold:
            traj.recovered = int(not bool(traj.draft_em) and bool(traj.em))
            traj.harmed = int(bool(traj.draft_em) and not bool(traj.em))
        traj.n_large_calls = self._large_calls
        traj.n_small_calls = self._small_calls
        traj.calls_by_role = dict(self._calls_by_role)
        traj.elapsed_sec = round(time.monotonic() - started, 3)
        return traj
