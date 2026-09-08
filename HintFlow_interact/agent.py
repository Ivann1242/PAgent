"""Small model writes prompts; large model plans / solves / critiques.

Five interactions. Turn 1 large call is always a bare solve so the HF1 floor
is not discarded. Later turns only replace the incumbent with HF1 KEEP/REPLACE
rules (parseable + confidence threshold).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from core import call_llm, exact_match, extract_final_answer, has_parseable_answer
from HintFlow_interact.prompts import (
    ACTS,
    SOLVE_ACTS,
    build_small_prompt,
    clip,
    wrap_large_prompt,
)
from HintFlow_one.one_agent import (
    SELECTOR_SYSTEM,
    Candidate,
    Selection,
    build_selection_prompt,
    parse_selection,
)

SMALL_MAX_TOKENS = 256
THINK_MAX_TOKENS = 1024


@dataclass
class InteractTurn:
    act: str
    small_prompt: str
    small_output: str
    large_prompt: str
    large_output: str
    candidate: dict[str, Any] | None
    decision: str
    confidence: float
    reason: str
    became_incumbent: bool


@dataclass
class InteractTrajectory:
    problem: str
    gold: str = ""
    turns: list[InteractTurn] = field(default_factory=list)
    final_answer: str = ""
    baseline_em: int | None = None
    em: int | None = None
    recovered: int = 0
    harmed: int = 0
    n_parseable: int = 0
    replace_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InteractAgent:
    def __init__(
        self,
        *,
        small_url: str,
        small_model: str,
        solver_url: str,
        solver_model: str,
        solver_max_tokens: int = 8192,
        replace_threshold: float = 0.90,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.small = OpenAI(base_url=small_url, api_key="EMPTY", timeout=3600, max_retries=0)
        self.solver = OpenAI(base_url=solver_url, api_key="EMPTY", timeout=3600, max_retries=0)
        self.small_model = small_model
        self.solver_model = solver_model
        self.solver_max_tokens = solver_max_tokens
        self.replace_threshold = replace_threshold
        self.temperature = temperature
        self.seed = seed
        self._i = 0

    def _call(self, client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
        seed = self.seed + self._i
        self._i += 1
        return call_llm(
            client,
            model,
            prompt,
            temperature=self.temperature,
            max_tokens=max_tokens,
            seed=seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).strip()

    def _as_candidate(self, source: str, solution: str, gold: str, hint: str = "") -> Candidate:
        parseable = has_parseable_answer(solution)
        answer = extract_final_answer(solution) if parseable else ""
        return Candidate(
            source=source,
            solution=solution,
            answer=answer,
            parseable=parseable,
            hint=hint,
            em=exact_match(answer, gold) if gold and answer else (0 if gold else None),
        )

    def _select_with_system(
        self, problem: str, incumbent: Candidate, challenger: Candidate
    ) -> Selection:
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
            seed = self.seed + self._i
            self._i += 1
            raw = call_llm(
                self.small,
                self.small_model,
                build_selection_prompt(
                    problem, incumbent, challenger, challenger_mode="blind_ff"
                ),
                system=SELECTOR_SYSTEM,
                temperature=0.0,
                max_tokens=192,
                seed=seed,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return parse_selection(raw)
        except Exception as exc:  # noqa: BLE001
            return Selection(reason=f"selector failure; kept incumbent: {type(exc).__name__}")

    def run(self, problem: str, *, gold: str = "") -> InteractTrajectory:
        traj = InteractTrajectory(problem=problem, gold=gold)
        scout_note = ""
        plan = ""
        critique = ""
        prior_hints: list[str] = []
        incumbent: Candidate | None = None

        for act in ACTS:
            small_prompt = build_small_prompt(
                act,
                problem,
                scout_note=scout_note,
                plan=plan,
                critique=critique,
                incumbent_answer=incumbent.answer if incumbent else "",
                incumbent_solution=incumbent.solution if incumbent else "",
                prior_hints=prior_hints,
            )
            small_out = self._call(self.small, self.small_model, small_prompt, SMALL_MAX_TOKENS)
            if act == "scout":
                scout_note = small_out
            elif act in {"alt_solve", "revise"}:
                prior_hints.append(small_out)

            large_prompt = wrap_large_prompt(
                act,
                problem,
                small_out,
                incumbent_solution=incumbent.solution if incumbent else "",
                incumbent_answer=incumbent.answer if incumbent else "",
            )
            max_tok = self.solver_max_tokens if act in SOLVE_ACTS else THINK_MAX_TOKENS
            large_out = self._call(self.solver, self.solver_model, large_prompt, max_tok)

            if act == "plan":
                plan = large_out
            elif act == "critique":
                critique = large_out

            cand: Candidate | None = None
            decision, confidence, reason = "SKIP", 1.0, "not a solve turn"
            became = False
            if act in SOLVE_ACTS:
                cand = self._as_candidate(act.upper(), large_out, gold, hint=small_out)
                traj.n_parseable += int(cand.parseable)
                if incumbent is None:
                    incumbent = cand
                    decision, reason = "INIT", "first solve is the incumbent (bare scout)"
                    became = True
                else:
                    sel = self._select_with_system(problem, incumbent, cand)
                    decision, confidence, reason = sel.decision, sel.confidence, sel.reason
                    if (
                        sel.decision == "REPLACE"
                        and sel.confidence >= self.replace_threshold
                    ):
                        incumbent = cand
                        became = True
                        traj.replace_count += 1

            traj.turns.append(
                InteractTurn(
                    act=act,
                    small_prompt=clip(small_prompt, 2000),
                    small_output=small_out,
                    large_prompt=clip(large_prompt, 2500),
                    large_output=large_out,
                    candidate=asdict(cand) if cand else None,
                    decision=decision,
                    confidence=confidence,
                    reason=reason,
                    became_incumbent=became,
                )
            )

        if incumbent is None:
            traj.final_answer = ""
            traj.em = 0 if gold else None
            return traj

        traj.final_answer = incumbent.answer
        baseline = next((t.candidate for t in traj.turns if t.act == "scout"), None)
        if gold:
            traj.em = exact_match(traj.final_answer, gold) if traj.final_answer else 0
            traj.baseline_em = int(baseline["em"]) if baseline else 0
            traj.recovered = int((not traj.baseline_em) and traj.em)
            traj.harmed = int(traj.baseline_em and not traj.em)
        else:
            traj.baseline_em = baseline.get("em") if baseline else None
        return traj
