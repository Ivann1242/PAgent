from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from core import call_llm, exact_match, extract_final_answer
from HintFlow_five.common import (
    N_TURNS,
    build_router_prompt,
    build_selector_prompt,
    build_step_prompt,
    parse_json_object,
    SELECTOR_SYSTEM,
)


@dataclass
class TurnRecord:
    turn: int
    hint: str
    baseline: str
    hinted: str
    decision: str
    confidence: float
    reason: str
    accepted: str
    selector_parse_ok: bool = True


@dataclass
class FiveTrajectory:
    problem: str
    gold: str = ""
    mode: str = "router"
    turns: list[TurnRecord] = field(default_factory=list)
    final_answer: str = ""
    em: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FiveTurnAgent:
    def __init__(
        self,
        *,
        router_url: str,
        router_model: str,
        solver_url: str,
        solver_model: str,
        selector_url: str | None = None,
        selector_model: str | None = None,
        n_turns: int = N_TURNS,
        solver_max_tokens: int = 2048,
        router_max_tokens: int = 192,
        selector_max_tokens: int = 192,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.router = OpenAI(base_url=router_url, api_key="EMPTY", timeout=3600, max_retries=0)
        self.solver = OpenAI(base_url=solver_url, api_key="EMPTY", timeout=3600, max_retries=0)
        self.selector = OpenAI(base_url=selector_url or router_url, api_key="EMPTY", timeout=3600, max_retries=0)
        self.router_model = router_model
        self.solver_model = solver_model
        self.selector_model = selector_model or router_model
        self.n_turns = n_turns
        self.solver_max_tokens = solver_max_tokens
        self.router_max_tokens = router_max_tokens
        self.selector_max_tokens = selector_max_tokens
        self.temperature = temperature
        self.seed = seed
        self._call_index = 0

    def _call(self, client, model: str, prompt: str, max_tokens: int) -> str:
        call_seed = self.seed + self._call_index
        self._call_index += 1
        return call_llm(
            client,
            model,
            prompt,
            temperature=self.temperature,
            max_tokens=max_tokens,
            seed=call_seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).strip()

    def _select(
        self, problem: str, state: list[str], turn: int, baseline: str, hinted: str
    ) -> tuple[str, float, str, bool]:
        if not baseline.strip() and hinted.strip():
            return "HINTED", 1.0, "baseline is empty", True
        if not hinted.strip():
            return "BASELINE", 1.0, "hinted candidate is empty", True
        response = self.selector.chat.completions.create(
            model=self.selector_model,
            messages=[
                {"role": "system", "content": SELECTOR_SYSTEM},
                {"role": "user", "content": build_selector_prompt(problem, state, turn, baseline, hinted, self.n_turns)},
            ],
            temperature=0.0,
            max_tokens=self.selector_max_tokens,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            obj = parse_json_object(raw)
            decision = str(obj.get("decision", "BASELINE")).upper().strip()
            if decision not in {"BASELINE", "HINTED"}:
                decision = "BASELINE"
            confidence = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
            return decision, confidence, str(obj.get("reason", "")).strip(), True
        except Exception:
            return "BASELINE", 0.0, "selector parse failure", False

    def run(self, problem: str, *, gold: str = "", mode: str = "router") -> FiveTrajectory:
        if mode not in {"router", "baseline", "oracle"}:
            raise ValueError("mode must be router, baseline, or oracle")
        traj = FiveTrajectory(problem=problem, gold=gold, mode=mode)
        state: list[str] = []
        for turn in range(1, self.n_turns + 1):
            baseline = self._call(
                self.solver,
                self.solver_model,
                build_step_prompt(problem, state, turn, n_turns=self.n_turns),
                self.solver_max_tokens,
            )
            hint = ""
            hinted = ""
            if mode != "baseline":
                hint = self._call(
                    self.router,
                    self.router_model,
                    build_router_prompt(problem, state, turn, self.n_turns),
                    self.router_max_tokens,
                )
                hinted = self._call(
                    self.solver,
                    self.solver_model,
                    build_step_prompt(problem, state, turn, hint, self.n_turns),
                    self.solver_max_tokens,
                )
                decision, confidence, reason, parse_ok = self._select(
                    problem, state, turn, baseline, hinted
                )
            else:
                decision, confidence, reason, parse_ok = "BASELINE", 1.0, "baseline-only", True
            accepted = hinted if decision == "HINTED" else baseline
            state.append(accepted)
            traj.turns.append(
                TurnRecord(turn, hint, baseline, hinted, decision, confidence, reason, accepted, parse_ok)
            )
        traj.final_answer = extract_final_answer(state[-1]) if state else ""
        traj.em = exact_match(traj.final_answer, gold) if gold else None
        return traj

