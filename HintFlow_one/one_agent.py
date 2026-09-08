#!/usr/bin/env python3
"""One-step Blind FF + conservative KEEP/REPLACE selector."""

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
    build_optimizer_prompt,
    exact_match,
    extract_final_answer,
    has_parseable_answer,
    parse_optimizer_output,
)

ORCH_URL = "http://127.0.0.1:8086/v1"
ORCH_MODEL = "qwen3-4b-blind-ff-17k"
SOLVER_URL = "http://127.0.0.1:8006/v1"
SOLVER_MODEL = "qwen3-14b"

SELECTOR_SYSTEM = """You are a conservative pairwise math-solution selector.
Compare an incumbent (bare baseline) and a challenger for the same problem.
Select REPLACE only when the challenger is clearly more likely to contain
the correct requested final answer. If uncertain, KEEP the incumbent.

Output ONLY valid JSON:
{"decision":"KEEP | REPLACE","confidence":0.0,"reason":"short concrete reason"}"""

CHALLENGER_MODES = ("blind_ff", "resample_baseline", "fixed_cot")

DEFAULT_FIXED_COT = (
    "Think step by step. Write a clear chain of reasoning before the final answer. "
    "Double-check arithmetic and algebraic manipulations. Do not skip intermediate steps."
)

CHALLENGER_PROMPT_LABEL = {
    "blind_ff": "CHALLENGER — Blind FF hinted solve",
    "resample_baseline": "CHALLENGER — resampled bare baseline (higher temperature)",
    "fixed_cot": "CHALLENGER — fixed chain-of-thought prompt",
}


def _message_text(resp: Any) -> str:
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    if content:
        return str(content).strip()
    for name in ("reasoning", "reasoning_content"):
        value = getattr(msg, name, None)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = max(limit // 2 - 40, 1)
    return text[:half] + "\n...[middle truncated]...\n" + text[-half:]


def parse_selection(text: str) -> "Selection":
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return Selection(raw=raw)
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return Selection(raw=raw)
    if not isinstance(obj, dict):
        return Selection(raw=raw)
    decision = str(obj.get("decision") or "KEEP").strip().upper()
    if decision not in {"KEEP", "REPLACE"}:
        decision = "KEEP"
    try:
        confidence = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return Selection(
        decision=decision,
        confidence=confidence,
        reason=str(obj.get("reason") or "").strip(),
        raw=raw,
    )


def build_selection_prompt(
    problem: str,
    incumbent: "Candidate",
    challenger: "Candidate",
    *,
    challenger_mode: str = "blind_ff",
) -> str:
    chal_label = CHALLENGER_PROMPT_LABEL.get(
        challenger_mode, "CHALLENGER — alternative solve"
    )
    return (
        f"Problem:\n{_clip(problem, 5000)}\n\n"
        f"[INCUMBENT — bare baseline]\n{_clip(incumbent.solution, 6500)}\n\n"
        f"[{chal_label}]\n{_clip(challenger.solution, 6500)}\n\n"
        "Choose KEEP or REPLACE. Prefer KEEP when evidence is inconclusive."
    )


@dataclass
class Candidate:
    source: str
    solution: str
    answer: str
    parseable: bool
    prompt: str = ""
    hint: str = ""
    em: int | None = None


@dataclass
class Selection:
    decision: str = "KEEP"
    confidence: float = 0.0
    reason: str = ""
    raw: str = ""


@dataclass
class OneTrajectory:
    problem: str
    gold: str = ""
    baseline: Candidate | None = None
    hint: str = ""
    hint_raw: str = ""
    hint_parse_ok: bool = False
    challenger: Candidate | None = None
    selection: Selection = field(default_factory=Selection)
    final_answer: str = ""
    baseline_em: int | None = None
    challenger_em: int | None = None
    em: int | None = None
    recovered: int = 0
    harmed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HintFlowOneAgent:
    """Baseline-first KEEP/REPLACE; challenger can be Blind FF / resample / fixed CoT."""

    def __init__(
        self,
        *,
        orch_url: str = ORCH_URL,
        orch_model: str = ORCH_MODEL,
        solver_url: str = SOLVER_URL,
        solver_model: str = SOLVER_MODEL,
        solver_max_tokens: int = 20000,
        orch_temperature: float = 0.0,
        solver_temperature: float = 0.0,
        challenger_mode: str = "blind_ff",
        challenger_temperature: float | None = None,
        fixed_cot: str = DEFAULT_FIXED_COT,
        replace_threshold: float = 0.90,
        selector_mode: str = "orch",
        request_timeout: float = 3600.0,
        solver_seed: int | None = None,
    ) -> None:
        if selector_mode not in {"orch", "keep", "replace"}:
            raise ValueError("selector_mode must be orch, keep, or replace")
        if challenger_mode not in CHALLENGER_MODES:
            raise ValueError(f"challenger_mode must be one of {CHALLENGER_MODES}")
        self.orch = OpenAI(
            base_url=orch_url, api_key="EMPTY", max_retries=0, timeout=request_timeout
        )
        self.solver = OpenAI(
            base_url=solver_url, api_key="EMPTY", max_retries=0, timeout=request_timeout
        )
        self.orch_model = orch_model
        self.solver_model = solver_model
        self.solver_max_tokens = solver_max_tokens
        self.orch_temperature = orch_temperature
        self.solver_temperature = solver_temperature
        self.challenger_mode = challenger_mode
        if challenger_temperature is None:
            # Resample needs T>0; fixed_cot / blind_ff default to greedy like baseline.
            self.challenger_temperature = (
                0.7 if challenger_mode == "resample_baseline" else solver_temperature
            )
        else:
            self.challenger_temperature = float(challenger_temperature)
        self.fixed_cot = (fixed_cot or DEFAULT_FIXED_COT).strip()
        self.replace_threshold = min(max(replace_threshold, 0.0), 1.0)
        self.selector_mode = selector_mode
        self.solver_seed = solver_seed
        self._solver_call_index = 0

    def _orch_chat(
        self,
        user: str,
        *,
        system: str | None = None,
        max_tokens: int = 192,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self.orch.chat.completions.create(
            model=self.orch_model,
            messages=messages,
            temperature=self.orch_temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _message_text(resp)

    def _solver_chat(self, prompt: str, *, temperature: float | None = None) -> str:
        temp = self.solver_temperature if temperature is None else temperature
        kwargs: dict[str, Any] = {
            "model": self.solver_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": self.solver_max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if self.solver_seed is not None:
            kwargs["extra_body"]["seed"] = int(
                self.solver_seed + self._solver_call_index
            )
        self._solver_call_index += 1
        resp = self.solver.chat.completions.create(**kwargs)
        return _message_text(resp)

    def _candidate(
        self,
        *,
        source: str,
        prompt: str,
        solution: str,
        gold: str,
        hint: str = "",
    ) -> Candidate:
        parseable = has_parseable_answer(solution)
        answer = extract_final_answer(solution) if parseable else ""
        return Candidate(
            source=source,
            solution=solution,
            answer=answer,
            parseable=parseable,
            prompt=prompt,
            hint=hint,
            em=exact_match(answer, gold) if gold and answer else (0 if gold else None),
        )

    def generate_baseline(self, problem: str, *, gold: str = "") -> Candidate:
        prompt = build_large_prompt(problem, "")
        solution = self._solver_chat(prompt, temperature=self.solver_temperature)
        return self._candidate(
            source="BASELINE", prompt=prompt, solution=solution, gold=gold
        )

    def generate_hint(self, problem: str) -> tuple[str, str, bool]:
        # Match eval.py Blind FF router call: user-only optimizer prompt.
        raw = self._orch_chat(build_optimizer_prompt(problem), max_tokens=256)
        hint, ok = parse_optimizer_output(raw)
        return hint, raw, ok

    def generate_challenger(
        self, problem: str, hint: str, *, gold: str = ""
    ) -> Candidate:
        prompt = build_large_prompt(problem, hint)
        solution = self._solver_chat(prompt, temperature=self.challenger_temperature)
        source = {
            "blind_ff": "FF_CHALLENGER",
            "resample_baseline": "RESAMPLE_CHALLENGER",
            "fixed_cot": "FIXED_COT_CHALLENGER",
        }[self.challenger_mode]
        return self._candidate(
            source=source,
            prompt=prompt,
            solution=solution,
            gold=gold,
            hint=hint,
        )

    def select(self, problem: str, incumbent: Candidate, challenger: Candidate) -> Selection:
        if self.selector_mode == "keep":
            return Selection(decision="KEEP", confidence=1.0, reason="configured keep")
        if self.selector_mode == "replace":
            return Selection(
                decision="REPLACE", confidence=1.0, reason="configured replace"
            )
        if not challenger.parseable:
            return Selection(reason="challenger has no parseable answer")
        if not incumbent.parseable:
            return Selection(
                decision="REPLACE",
                confidence=1.0,
                reason="incumbent has no parseable answer",
            )
        if exact_match(challenger.answer, incumbent.answer):
            return Selection(reason="same normalized candidate answer")
        try:
            text = self._orch_chat(
                build_selection_prompt(
                    problem,
                    incumbent,
                    challenger,
                    challenger_mode=self.challenger_mode,
                ),
                system=SELECTOR_SYSTEM,
                max_tokens=192,
            )
            return parse_selection(text)
        except Exception as exc:  # noqa: BLE001
            return Selection(
                reason=f"selector failure; kept incumbent: {type(exc).__name__}"
            )

    def run(
        self, problem: str, *, gold: str = "", baseline_solution: str | None = None,
    ) -> OneTrajectory:
        traj = OneTrajectory(problem=problem, gold=gold)
        if baseline_solution is None:
            baseline = self.generate_baseline(problem, gold=gold)
        else:
            baseline = self._candidate(
                source="CACHED_BASELINE",
                prompt=build_large_prompt(problem, ""),
                solution=baseline_solution,
                gold=gold,
            )
        traj.baseline = baseline
        traj.baseline_em = baseline.em

        if self.challenger_mode == "blind_ff":
            hint, hint_raw, hint_ok = self.generate_hint(problem)
        elif self.challenger_mode == "fixed_cot":
            hint, hint_raw, hint_ok = self.fixed_cot, self.fixed_cot, True
        else:  # resample_baseline: same bare prompt, higher T
            hint, hint_raw, hint_ok = "", "", True
        traj.hint = hint
        traj.hint_raw = hint_raw
        traj.hint_parse_ok = hint_ok

        challenger = self.generate_challenger(problem, hint, gold=gold)
        traj.challenger = challenger
        traj.challenger_em = challenger.em

        selection = self.select(problem, baseline, challenger)
        traj.selection = selection
        use_challenger = (
            selection.decision == "REPLACE"
            and selection.confidence >= self.replace_threshold
        )
        chosen = challenger if use_challenger else baseline
        traj.final_answer = chosen.answer
        if gold:
            traj.em = exact_match(traj.final_answer, gold) if traj.final_answer else 0
            base = int(traj.baseline_em or 0)
            final = int(traj.em or 0)
            traj.recovered = int((not base) and final)
            traj.harmed = int(base and not final)
        return traj


__all__ = [
    "CHALLENGER_MODES",
    "DEFAULT_FIXED_COT",
    "Candidate",
    "HintFlowOneAgent",
    "OneTrajectory",
    "Selection",
    "build_selection_prompt",
    "parse_selection",
]
