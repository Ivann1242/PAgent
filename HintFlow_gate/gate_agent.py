#!/usr/bin/env python3
"""Hint-gated single-OSS solve: hint → gate → one solver rollout."""

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

MODES = ("bare", "gated")

GATE_SYSTEM = """You are a conservative hint gate for a math solver.
You are given a problem and a candidate hint. Decide whether to USE_HINT
(pass the hint to the solver) or DROP it (solve with the bare problem only).

Use USE_HINT only when the hint clearly provides useful strategy/guidance
without spoiling a numeric final answer, and is unlikely to mislead.
If uncertain, DROP.

Output ONLY valid JSON:
{"decision":"USE_HINT | DROP","confidence":0.0,"reason":"short concrete reason"}"""


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


@dataclass
class GateDecision:
    decision: str = "DROP"
    confidence: float = 0.0
    reason: str = ""
    raw: str = ""


def parse_gate(text: str) -> GateDecision:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return GateDecision(raw=raw)
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return GateDecision(raw=raw)
    if not isinstance(obj, dict):
        return GateDecision(raw=raw)
    decision = str(obj.get("decision") or "DROP").strip().upper().replace("-", "_")
    if decision in {"USE", "KEEP", "USEHINT"}:
        decision = "USE_HINT"
    if decision not in {"USE_HINT", "DROP"}:
        decision = "DROP"
    try:
        confidence = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return GateDecision(
        decision=decision,
        confidence=confidence,
        reason=str(obj.get("reason") or "").strip(),
        raw=raw,
    )


def build_gate_prompt(problem: str, hint: str) -> str:
    return (
        f"Problem:\n{_clip(problem, 5000)}\n\n"
        f"Candidate hint:\n{_clip(hint, 2000)}\n\n"
        "Decide USE_HINT or DROP. Prefer DROP when evidence is inconclusive."
    )


@dataclass
class GateTrajectory:
    problem: str
    gold: str = ""
    mode: str = "gated"
    hint: str = ""
    hint_raw: str = ""
    hint_parse_ok: bool = False
    gate: GateDecision = field(default_factory=GateDecision)
    used_hint: bool = False
    solver_prompt: str = ""
    solution: str = ""
    final_answer: str = ""
    parseable: bool = False
    em: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HintGateAgent:
    """Blind FF hint + conservative gate + single OSS solve."""

    def __init__(
        self,
        *,
        orch_url: str = ORCH_URL,
        orch_model: str = ORCH_MODEL,
        solver_url: str = SOLVER_URL,
        solver_model: str = SOLVER_MODEL,
        solver_max_tokens: int = 8192,
        orch_temperature: float = 0.0,
        solver_temperature: float = 0.0,
        use_threshold: float = 0.90,
        mode: str = "gated",
        request_timeout: float = 3600.0,
        solver_seed: int | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
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
        self.use_threshold = min(max(use_threshold, 0.0), 1.0)
        self.mode = mode
        self.solver_seed = solver_seed

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

    def _solver_chat(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.solver_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.solver_temperature,
            "max_tokens": self.solver_max_tokens,
        }
        if self.solver_seed is not None:
            kwargs["extra_body"] = {"seed": int(self.solver_seed)}
        resp = self.solver.chat.completions.create(**kwargs)
        return _message_text(resp)

    def generate_hint(self, problem: str) -> tuple[str, str, bool]:
        raw = self._orch_chat(build_optimizer_prompt(problem), max_tokens=256)
        hint, ok = parse_optimizer_output(raw)
        return hint, raw, ok

    def gate_hint(self, problem: str, hint: str) -> GateDecision:
        if not (hint or "").strip():
            return GateDecision(decision="DROP", confidence=1.0, reason="empty hint")
        try:
            text = self._orch_chat(
                build_gate_prompt(problem, hint),
                system=GATE_SYSTEM,
                max_tokens=192,
            )
            return parse_gate(text)
        except Exception as exc:  # noqa: BLE001
            return GateDecision(
                decision="DROP",
                confidence=0.0,
                reason=f"gate failure; dropped: {type(exc).__name__}",
            )

    def run(self, problem: str, *, gold: str = "") -> GateTrajectory:
        traj = GateTrajectory(problem=problem, gold=gold, mode=self.mode)

        if self.mode == "bare":
            traj.used_hint = False
            hint_for_solver = ""
        else:
            hint, hint_raw, hint_ok = self.generate_hint(problem)
            traj.hint = hint
            traj.hint_raw = hint_raw
            traj.hint_parse_ok = hint_ok
            gate = self.gate_hint(problem, hint)
            traj.gate = gate
            traj.used_hint = (
                gate.decision == "USE_HINT" and gate.confidence >= self.use_threshold
            )
            hint_for_solver = hint if traj.used_hint else ""

        prompt = build_large_prompt(problem, hint_for_solver)
        traj.solver_prompt = prompt
        solution = self._solver_chat(prompt)
        traj.solution = solution
        traj.parseable = has_parseable_answer(solution)
        traj.final_answer = extract_final_answer(solution) if traj.parseable else ""
        if gold:
            traj.em = (
                exact_match(traj.final_answer, gold) if traj.final_answer else 0
            )
        return traj


__all__ = [
    "MODES",
    "GateDecision",
    "GateTrajectory",
    "HintGateAgent",
    "build_gate_prompt",
    "parse_gate",
]
