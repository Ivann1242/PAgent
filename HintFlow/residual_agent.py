#!/usr/bin/env python3
"""Baseline-first residual agent with turn-local feedback and candidate retention."""

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
    hint_leaks_gold,
)

ORCH_URL = "http://127.0.0.1:8086/v1"
ORCH_MODEL = "qwen3-4b"
SOLVER_URL = "http://127.0.0.1:8006/v1"
SOLVER_MODEL = "gpt-oss-20b"

RESIDUAL_ACTIONS = (
    "STOP",
    "VERIFY_REPAIR",
    "ALTERNATE_SOLVE",
    "TARGETED_CHECK",
)
NON_STOP_ACTIONS = RESIDUAL_ACTIONS[1:]
ERROR_TYPES = (
    "NONE",
    "INTERPRETATION",
    "FORMULA_CONSTRAINT",
    "ALGEBRA",
    "ARITHMETIC",
    "INCOMPLETE",
    "FORMAT",
    "UNKNOWN",
)
SELECTOR_DECISIONS = ("KEEP", "REPLACE")

DEFAULT_ACTION_SCHEDULE = (
    "VERIFY_REPAIR",
    "ALTERNATE_SOLVE",
    "TARGETED_CHECK",
    "ALTERNATE_SOLVE",
    "VERIFY_REPAIR",
    "TARGETED_CHECK",
)

FEEDBACK_SYSTEM = """You are a conservative math-solution feedback controller.
Inspect the problem and the current candidate solution without access to the gold
answer. Your job is to decide whether to stop or request one more full solution.

Output ONLY valid JSON:
{
  "p_correct": 0.0,
  "error_type": "NONE | INTERPRETATION | FORMULA_CONSTRAINT | ALGEBRA | ARITHMETIC | INCOMPLETE | FORMAT | UNKNOWN",
  "evidence": "short concrete reason",
  "action": "STOP | VERIFY_REPAIR | ALTERNATE_SOLVE | TARGETED_CHECK",
  "repair_hint": "short non-answer-revealing instruction, or empty",
  "confidence": 0.0
}

Be conservative about declaring a solution correct. STOP only when the reasoning,
constraints, arithmetic, requested quantity, and final-answer format all check out.
Do not solve the problem from scratch and do not state a replacement final answer."""

SELECTOR_JSON_SYSTEM = """You are a conservative pairwise math-solution selector.
Compare an incumbent and a challenger for the same problem. Select REPLACE only
when the challenger is clearly more likely to contain the correct requested final
answer. If uncertain, KEEP the incumbent.

Output ONLY valid JSON:
{"decision":"KEEP | REPLACE","confidence":0.0,"reason":"short reason"}"""

SELECTOR_SYSTEM = """Conservatively compare two candidate math solutions for the
same problem. Output exactly one token: KEEP or REPLACE. Output REPLACE only when
the challenger is clearly more likely correct; if uncertain, output KEEP."""

CORRECTNESS_SYSTEM = """Classify whether the candidate's requested final answer is
correct for the math problem. Output exactly one token: CORRECT or INCORRECT."""

ACTION_SYSTEM = """Choose the single best next residual action for improving the
current incumbent under the remaining call budget. Output exactly one token from:
STOP, VERIFY_REPAIR, ALTERNATE_SOLVE, TARGETED_CHECK."""

DIAGNOSIS_SYSTEM = """Diagnose the candidate solution without revealing or directly
stating the final answer. Output ONLY JSON:
{"error_type":"NONE | INTERPRETATION | FORMULA_CONSTRAINT | ALGEBRA | ARITHMETIC | INCOMPLETE | FORMAT | UNKNOWN",
 "evidence":"short concrete reason",
 "repair_hint":"short instruction that does not contain the final answer"}"""

TEACHER_SYSTEM = """You label math-solution errors for training. You may use the
provided gold answer to locate the first consequential error, but your repair hint
MUST NOT reveal, quote, or algebraically encode the gold answer.

Output ONLY JSON:
{"error_type":"NONE | INTERPRETATION | FORMULA_CONSTRAINT | ALGEBRA | ARITHMETIC | INCOMPLETE | FORMAT | UNKNOWN",
 "evidence":"short concrete reason",
 "repair_hint":"short instruction that does not contain the final answer"}"""


def _clip(text: str, limit: int = 7000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    # Preserve both the setup and final derivation/answer.
    half = max(limit // 2 - 40, 1)
    return text[:half] + "\n...[middle truncated]...\n" + text[-half:]


def _estimate_tokens(text: str) -> int:
    # Conservative mixed English/CJK/LaTeX estimate without loading a tokenizer.
    cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
    return cjk + max(len(text) - cjk, 0) // 3 + 32


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    for match in reversed(list(re.finditer(r"\{.*?\}", raw, re.DOTALL))):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


def _message_text(resp: Any) -> str:
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    if content:
        return str(content).strip()
    for name in ("reasoning", "reasoning_content"):
        value = getattr(msg, name, None)
        if value:
            return str(value).strip()
    return ""


def _prob(value: Any, default: float = 0.5) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _leaks_gold(text: str, gold: str) -> bool:
    if hint_leaks_gold(text, gold):
        return True
    gold = (gold or "").strip()
    return bool(
        gold
        and re.search(
            rf"(?<![A-Za-z0-9]){re.escape(gold)}(?![A-Za-z0-9])",
            text or "",
        )
    )


def _redact_numeric_literals(text: str) -> str:
    return re.sub(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?",
        "<number>",
        text or "",
    )


def build_correctness_prompt(problem: str, candidate: str) -> str:
    return (
        f"Problem:\n{_clip(problem, 6000)}\n\n"
        f"Candidate solution:\n{_clip(candidate)}\n\n"
        "Is the candidate's requested final answer correct?"
    )


def build_selection_prompt(problem: str, incumbent: str, challenger: str) -> str:
    return (
        f"Problem:\n{_clip(problem, 5000)}\n\n"
        f"[INCUMBENT]\n{_clip(incumbent, 6500)}\n\n"
        f"[CHALLENGER]\n{_clip(challenger, 6500)}\n\n"
        "Choose KEEP or REPLACE."
    )


def build_action_prompt(
    problem: str,
    incumbent: str,
    *,
    p_correct: float,
    error_type: str,
    evidence: str,
    tried_actions: list[str],
    remaining_calls: int,
) -> str:
    return (
        f"Problem:\n{_clip(problem, 4500)}\n\n"
        f"Current incumbent:\n{_clip(incumbent, 6000)}\n\n"
        f"Estimated correctness: {p_correct:.3f}\n"
        f"Suspected error type: {error_type}\n"
        f"Evidence: {evidence or '(none)'}\n"
        f"Actions already tried: {', '.join(tried_actions) or '(none)'}\n"
        f"Remaining solver calls: {remaining_calls}\n"
    )


def build_diagnosis_prompt(problem: str, candidate: str) -> str:
    return (
        f"Problem:\n{_clip(problem, 5500)}\n\n"
        f"Candidate solution:\n{_clip(candidate, 7000)}\n\n"
        "Diagnose the candidate."
    )


def build_teacher_prompt(problem: str, candidate: str, gold: str) -> str:
    return (
        f"Problem:\n{_clip(problem, 5500)}\n\n"
        f"Candidate solution:\n{_clip(candidate, 7000)}\n\n"
        f"Gold final answer (training only): {gold}\n\n"
        "Locate the first consequential error. Do not reveal the gold answer."
    )


def build_residual_solver_prompt(
    problem: str,
    action: str,
    *,
    incumbent: str,
    feedback: "TurnFeedback",
    variant: int = 0,
) -> str:
    final_rule = (
        "\n\nReturn a complete self-contained solution, not just a critique. "
        "End exactly with: Final Answer: <answer>"
    )
    if action == "VERIFY_REPAIR":
        modes = (
            "Audit the candidate line by line, especially the requested quantity and arithmetic. "
            "Keep valid work, repair the first consequential error, and recompute the answer.",
            "Independently reverse-check the candidate from its claimed final answer back to every "
            "constraint. If any check fails, repair the solution before committing.",
        )
        instruction = modes[variant % len(modes)]
        return (
            f"Problem:\n{_clip(problem, 5000)}\n\n"
            f"Candidate to verify:\n{_clip(incumbent, 7000)}\n\n"
            f"Task:\n{instruction}{final_rule}"
        )
    if action == "ALTERNATE_SOLVE":
        modes = (
            "Solve independently using a materially different method. Do not trust or copy any "
            "previous derivation; check the exact quantity requested.",
            "Re-solve from first principles. Prefer explicit enumeration, substitution, or an "
            "independent invariant/check whenever applicable.",
            "Find a second solution route and use a small-case or boundary sanity check before "
            "committing to the requested answer.",
        )
        instruction = modes[variant % len(modes)]
        # Deliberately omit incumbent to reduce anchoring.
        return f"Problem:\n{_clip(problem, 7000)}\n\nTask:\n{instruction}{final_rule}"
    if action == "TARGETED_CHECK":
        hint = feedback.repair_hint or (
            "Check the interpretation, constraints, algebra, arithmetic, and requested output."
        )
        return (
            f"Problem:\n{_clip(problem, 5000)}\n\n"
            f"Current candidate:\n{_clip(incumbent, 7000)}\n\n"
            f"Suspected error type: {feedback.error_type}\n"
            f"Evidence: {feedback.evidence or '(none)'}\n"
            f"Targeted check: {hint}\n\n"
            "Verify the suspected point, repair any error, and re-check the final requested "
            f"quantity.{final_rule}"
        )
    raise ValueError(f"unsupported residual action: {action}")


@dataclass
class TurnFeedback:
    p_correct: float = 0.5
    error_type: str = "UNKNOWN"
    evidence: str = ""
    action: str = "VERIFY_REPAIR"
    repair_hint: str = ""
    confidence: float = 0.0
    raw: str = ""


@dataclass
class Selection:
    decision: str = "KEEP"
    confidence: float = 0.0
    reason: str = ""
    raw: str = ""


@dataclass
class Candidate:
    index: int
    action: str
    solution: str
    answer: str
    parseable: bool
    prompt: str = ""
    seed: int | None = None
    em: int | None = None


@dataclass
class ResidualTurn:
    index: int
    incumbent_before: int
    feedback: TurnFeedback
    action: str
    candidate_index: int | None
    selection: Selection | None
    incumbent_after: int


@dataclass
class ResidualTrajectory:
    problem: str
    gold: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    turns: list[ResidualTurn] = field(default_factory=list)
    incumbent_index: int = 0
    final_answer: str = ""
    baseline_em: int | None = None
    em: int | None = None
    oracle_em: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualHintFlowAgent:
    """Generate diverse full candidates while conservatively retaining an incumbent."""

    def __init__(
        self,
        *,
        orch_url: str = ORCH_URL,
        orch_model: str = ORCH_MODEL,
        solver_url: str = SOLVER_URL,
        solver_model: str = SOLVER_MODEL,
        max_solver_calls: int = 7,
        solver_max_tokens: int = 8192,
        branch_max_tokens: int = 4096,
        solver_context_limit: int = 16384,
        request_timeout: float = 600.0,
        branch_temperature: float = 0.2,
        replace_threshold: float = 0.70,
        policy_mode: str = "fixed",
        selector_mode: str = "orch",
        feedback_mode: str = "json",
        action_schedule: tuple[str, ...] = DEFAULT_ACTION_SCHEDULE,
    ) -> None:
        if not 1 <= max_solver_calls <= 7:
            raise ValueError("max_solver_calls must be in [1, 7], including baseline")
        if policy_mode not in {"fixed", "adaptive"}:
            raise ValueError("policy_mode must be fixed or adaptive")
        if selector_mode not in {"orch", "keep", "replace"}:
            raise ValueError("selector_mode must be orch, keep, or replace")
        if feedback_mode not in {"json", "trained"}:
            raise ValueError("feedback_mode must be json or trained")
        self.orch = OpenAI(
            base_url=orch_url,
            api_key="EMPTY",
            max_retries=0,
            timeout=request_timeout,
        )
        self.solver = OpenAI(
            base_url=solver_url,
            api_key="EMPTY",
            max_retries=0,
            timeout=request_timeout,
        )
        self.orch_model = orch_model
        self.solver_model = solver_model
        self.max_solver_calls = max_solver_calls
        self.solver_max_tokens = solver_max_tokens
        self.branch_max_tokens = branch_max_tokens
        self.solver_context_limit = solver_context_limit
        self.branch_temperature = branch_temperature
        self.replace_threshold = replace_threshold
        self.policy_mode = policy_mode
        self.selector_mode = selector_mode
        self.feedback_mode = feedback_mode
        self.action_schedule = tuple(a for a in action_schedule if a in NON_STOP_ACTIONS)

    def _orch_chat(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        resp = self.orch.chat.completions.create(
            model=self.orch_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _message_text(resp)

    def _solver_chat(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        extra_body: dict[str, Any] = {}
        if seed is not None:
            extra_body["seed"] = int(seed)
        requested = max_tokens or self.solver_max_tokens
        available = self.solver_context_limit - _estimate_tokens(prompt) - 128
        output_tokens = max(256, min(requested, available))
        kwargs: dict[str, Any] = {
            "model": self.solver_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": output_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self.solver.chat.completions.create(**kwargs)
        return _message_text(resp)

    @staticmethod
    def _candidate(
        index: int,
        action: str,
        prompt: str,
        solution: str,
        *,
        gold: str,
        seed: int | None,
    ) -> Candidate:
        answer = extract_final_answer(solution)
        return Candidate(
            index=index,
            action=action,
            solution=solution,
            answer=answer,
            parseable=has_parseable_answer(solution),
            prompt=prompt,
            seed=seed,
            em=exact_match(answer, gold) if gold else None,
        )

    def generate_baseline(
        self,
        problem: str,
        *,
        gold: str = "",
        seed: int | None = None,
        index: int = 0,
    ) -> Candidate:
        prompt = build_large_prompt(problem, "")
        solution = self._solver_chat(
            prompt,
            seed=seed,
            temperature=0.0,
            max_tokens=self.solver_max_tokens,
        )
        return self._candidate(
            index, "BASELINE", prompt, solution, gold=gold, seed=seed
        )

    def generate_action_candidate(
        self,
        problem: str,
        incumbent: Candidate,
        feedback: TurnFeedback,
        action: str,
        *,
        variant: int,
        gold: str = "",
        seed: int | None = None,
        index: int,
    ) -> Candidate:
        prompt = build_residual_solver_prompt(
            problem,
            action,
            incumbent=incumbent.solution,
            feedback=feedback,
            variant=variant,
        )
        solution = self._solver_chat(
            prompt,
            seed=seed,
            temperature=self.branch_temperature,
            max_tokens=self.branch_max_tokens,
        )
        return self._candidate(
            index, action, prompt, solution, gold=gold, seed=seed
        )

    def _diagnose(self, problem: str, candidate: Candidate) -> TurnFeedback:
        text = self._orch_chat(
            FEEDBACK_SYSTEM,
            build_diagnosis_prompt(problem, candidate.solution),
            max_tokens=384,
        )
        obj = _extract_json(text)
        error_type = str(obj.get("error_type") or "UNKNOWN").strip().upper()
        if error_type not in ERROR_TYPES:
            error_type = "UNKNOWN"
        action = str(obj.get("action") or "VERIFY_REPAIR").strip().upper()
        if action not in RESIDUAL_ACTIONS:
            action = "VERIFY_REPAIR"
        return TurnFeedback(
            p_correct=_prob(obj.get("p_correct"), 0.5),
            error_type=error_type,
            evidence=str(obj.get("evidence") or "").strip(),
            action=action,
            repair_hint=str(obj.get("repair_hint") or "").strip(),
            confidence=_prob(obj.get("confidence"), 0.0),
            raw=text,
        )

    def _trained_feedback(
        self,
        problem: str,
        candidate: Candidate,
        *,
        tried_actions: list[str],
        remaining_calls: int,
    ) -> TurnFeedback:
        correctness, p_correct = self._trained_correctness(problem, candidate)
        diagnosis_raw = self._orch_chat(
            DIAGNOSIS_SYSTEM,
            build_diagnosis_prompt(problem, candidate.solution),
            max_tokens=256,
        )
        diagnosis = _extract_json(diagnosis_raw)
        error_type = str(diagnosis.get("error_type") or "UNKNOWN").strip().upper()
        if error_type not in ERROR_TYPES:
            error_type = "UNKNOWN"
        action_text = self._orch_chat(
            ACTION_SYSTEM,
            build_action_prompt(
                problem,
                candidate.solution,
                p_correct=p_correct,
                error_type=error_type,
                evidence=str(diagnosis.get("evidence") or ""),
                tried_actions=tried_actions,
                remaining_calls=remaining_calls,
            ),
            max_tokens=12,
        ).upper()
        action = next((a for a in RESIDUAL_ACTIONS if a in action_text), "VERIFY_REPAIR")
        return TurnFeedback(
            p_correct=p_correct,
            error_type=error_type,
            evidence=str(diagnosis.get("evidence") or "").strip(),
            action=action,
            repair_hint=str(diagnosis.get("repair_hint") or "").strip(),
            confidence=abs(p_correct - 0.5) * 2,
            raw=json.dumps(
                {
                    "correctness": correctness,
                    "diagnosis": diagnosis_raw,
                    "action": action_text,
                },
                ensure_ascii=False,
            ),
        )

    def _trained_correctness(
        self,
        problem: str,
        candidate: Candidate,
    ) -> tuple[str, float]:
        raw = self._orch_chat(
            CORRECTNESS_SYSTEM,
            build_correctness_prompt(problem, candidate.solution),
            max_tokens=8,
        )
        label = raw.strip().upper().split()[0] if raw.strip() else "INCORRECT"
        if label not in {"CORRECT", "INCORRECT"}:
            label = "INCORRECT"
        return label, (0.9 if label == "CORRECT" else 0.1)

    def feedback(
        self,
        problem: str,
        candidate: Candidate,
        *,
        tried_actions: list[str] | None = None,
        remaining_calls: int = 0,
    ) -> TurnFeedback:
        if self.feedback_mode == "trained":
            return self._trained_feedback(
                problem,
                candidate,
                tried_actions=tried_actions or [],
                remaining_calls=remaining_calls,
            )
        return self._diagnose(problem, candidate)

    def select(
        self,
        problem: str,
        incumbent: Candidate,
        challenger: Candidate,
    ) -> Selection:
        if self.selector_mode == "keep":
            return Selection(decision="KEEP", confidence=1.0, reason="configured keep")
        if self.selector_mode == "replace":
            return Selection(
                decision="REPLACE", confidence=1.0, reason="configured replace"
            )
        if self.feedback_mode == "trained":
            text = self._orch_chat(
                SELECTOR_SYSTEM,
                build_selection_prompt(problem, incumbent.solution, challenger.solution),
                max_tokens=8,
            )
            label = text.strip().upper().split()[0] if text.strip() else "KEEP"
            if label not in SELECTOR_DECISIONS:
                label = "KEEP"
            incumbent_label, _ = self._trained_correctness(problem, incumbent)
            challenger_label, _ = self._trained_correctness(problem, challenger)
            # Pointwise correctness is the calibrated safety gate; pairwise
            # selection is auxiliary because strict pair labels are much sparser.
            supported_replace = (
                incumbent_label == "INCORRECT"
                and challenger_label == "CORRECT"
            )
            decision = "REPLACE" if supported_replace else "KEEP"
            return Selection(
                decision=decision,
                confidence=(
                    0.95
                    if supported_replace and label == "REPLACE"
                    else 0.80 if supported_replace else 0.90
                ),
                reason=(
                    f"selector={label}; incumbent={incumbent_label}; "
                    f"challenger={challenger_label}"
                ),
                raw=text,
            )
        text = self._orch_chat(
            SELECTOR_JSON_SYSTEM,
            build_selection_prompt(problem, incumbent.solution, challenger.solution),
            max_tokens=192,
        )
        obj = _extract_json(text)
        decision = str(obj.get("decision") or "KEEP").strip().upper()
        if decision not in SELECTOR_DECISIONS:
            decision = "KEEP"
        return Selection(
            decision=decision,
            confidence=_prob(obj.get("confidence"), 0.0),
            reason=str(obj.get("reason") or "").strip(),
            raw=text,
        )

    def teacher_feedback(
        self,
        problem: str,
        candidate: Candidate,
        gold: str,
    ) -> TurnFeedback:
        if candidate.em:
            return TurnFeedback(
                p_correct=1.0,
                error_type="NONE",
                evidence="",
                action="STOP",
                repair_hint="",
                confidence=1.0,
                raw="gold-verified-correct",
            )
        text = self._orch_chat(
            TEACHER_SYSTEM,
            build_teacher_prompt(problem, candidate.solution, gold),
            max_tokens=256,
        )
        obj = _extract_json(text)
        error_type = str(obj.get("error_type") or "UNKNOWN").strip().upper()
        if error_type not in ERROR_TYPES:
            error_type = "UNKNOWN"
        if error_type == "NONE":
            error_type = "UNKNOWN"
        hint = str(obj.get("repair_hint") or "").strip()
        evidence = str(obj.get("evidence") or "").strip()
        if _leaks_gold(hint, gold):
            hint = ""
        if _leaks_gold(evidence, gold):
            evidence = ""
        hint = _redact_numeric_literals(hint)
        evidence = _redact_numeric_literals(evidence)
        return TurnFeedback(
            p_correct=float(candidate.em or 0),
            error_type=error_type,
            evidence=evidence,
            action="STOP" if candidate.em else "TARGETED_CHECK",
            repair_hint=hint,
            confidence=1.0,
            raw=text,
        )

    def run(
        self,
        problem: str,
        *,
        gold: str = "",
        seed: int = 0,
    ) -> ResidualTrajectory:
        traj = ResidualTrajectory(problem=problem, gold=gold)
        baseline = self.generate_baseline(problem, gold=gold, seed=seed, index=0)
        traj.candidates.append(baseline)
        traj.incumbent_index = 0
        tried_actions: list[str] = []
        feedback_cache: dict[int, TurnFeedback] = {}

        schedule = list(self.action_schedule)[: max(self.max_solver_calls - 1, 0)]
        for turn_index in range(self.max_solver_calls - 1):
            incumbent = traj.candidates[traj.incumbent_index]
            if (
                self.policy_mode == "fixed"
                and self.feedback_mode == "json"
                and traj.incumbent_index in feedback_cache
            ):
                feedback = feedback_cache[traj.incumbent_index]
            else:
                feedback = self.feedback(
                    problem,
                    incumbent,
                    tried_actions=tried_actions,
                    remaining_calls=self.max_solver_calls - len(traj.candidates),
                )
                feedback_cache[traj.incumbent_index] = feedback
            if self.policy_mode == "fixed":
                if turn_index >= len(schedule):
                    break
                action = schedule[turn_index]
            else:
                action = feedback.action
                if action == "STOP":
                    traj.turns.append(
                        ResidualTurn(
                            index=turn_index,
                            incumbent_before=traj.incumbent_index,
                            feedback=feedback,
                            action="STOP",
                            candidate_index=None,
                            selection=None,
                            incumbent_after=traj.incumbent_index,
                        )
                    )
                    break

            before = traj.incumbent_index
            variant = sum(1 for a in tried_actions if a == action)
            challenger = self.generate_action_candidate(
                problem,
                incumbent,
                feedback,
                action,
                variant=variant,
                gold=gold,
                seed=seed + turn_index + 1,
                index=len(traj.candidates),
            )
            traj.candidates.append(challenger)
            selection = self.select(problem, incumbent, challenger)
            if (
                selection.decision == "REPLACE"
                and selection.confidence >= self.replace_threshold
            ):
                traj.incumbent_index = challenger.index
            traj.turns.append(
                ResidualTurn(
                    index=turn_index,
                    incumbent_before=before,
                    feedback=feedback,
                    action=action,
                    candidate_index=challenger.index,
                    selection=selection,
                    incumbent_after=traj.incumbent_index,
                )
            )
            tried_actions.append(action)

        incumbent = traj.candidates[traj.incumbent_index]
        traj.final_answer = incumbent.answer
        if gold:
            traj.baseline_em = int(traj.candidates[0].em or 0)
            traj.em = int(incumbent.em or 0)
            traj.oracle_em = max(int(c.em or 0) for c in traj.candidates)
        return traj


__all__ = [
    "ACTION_SYSTEM",
    "CORRECTNESS_SYSTEM",
    "DEFAULT_ACTION_SCHEDULE",
    "DIAGNOSIS_SYSTEM",
    "ERROR_TYPES",
    "FEEDBACK_SYSTEM",
    "NON_STOP_ACTIONS",
    "ORCH_MODEL",
    "ORCH_URL",
    "RESIDUAL_ACTIONS",
    "SELECTOR_SYSTEM",
    "SELECTOR_JSON_SYSTEM",
    "SOLVER_MODEL",
    "SOLVER_URL",
    "Candidate",
    "ResidualHintFlowAgent",
    "ResidualTrajectory",
    "Selection",
    "TurnFeedback",
    "build_action_prompt",
    "build_correctness_prompt",
    "build_diagnosis_prompt",
    "build_residual_solver_prompt",
    "build_selection_prompt",
    "build_teacher_prompt",
]
