"""Speech-act templates: small model writes the next large-model prompt."""

from __future__ import annotations

ACTS = ("scout", "plan", "alt_solve", "critique", "revise")

# Large-model token budgets: solves get the long budget; plan/critique stay short.
SOLVE_ACTS = {"scout", "alt_solve", "revise"}
THINK_ACTS = {"plan", "critique"}


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = max(limit // 2 - 24, 1)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def build_small_prompt(
    act: str,
    problem: str,
    *,
    scout_note: str = "",
    plan: str = "",
    critique: str = "",
    incumbent_answer: str = "",
    incumbent_solution: str = "",
    prior_hints: list[str] | None = None,
) -> str:
    problem = clip(problem, 3500)
    prior = " | ".join(h for h in (prior_hints or []) if h.strip()) or "(none)"
    if act == "scout":
        return f"""You are briefing a stronger math model. Do NOT solve the problem.
Write 2-4 sentences on what kind of problem this is and what a solver should
watch out for (quantities, constraints, answer format). No method recipe,
no numbers that look like a final answer.

Problem:
{problem}

Output only the briefing."""

    if act == "plan":
        return f"""Write the USER PROMPT that we will send to a strong math model.
The prompt must ask it to list TWO distinct solution methods for the problem,
with one short reason each. It must NOT compute the final answer.

Problem:
{problem}

Your earlier briefing:
{clip(scout_note, 600)}

Output only the prompt text for the strong model."""

    if act == "alt_solve":
        return f"""Write a short hint (2-4 sentences) so a strong model solves this
problem with a method DIFFERENT from the current incumbent. Do not state the
final answer. Do not repeat these prior hints: {clip(prior, 400)}

Problem:
{problem}

Incumbent answer: {incumbent_answer or "(none)"}
Incumbent solution (excerpt):
{clip(incumbent_solution, 1200)}

Methods the strong model already listed:
{clip(plan, 1200)}

Output only the hint."""

    if act == "critique":
        return f"""Write the USER PROMPT that we will send to a strong math model.
It should check the incumbent solution for concrete algebra, case, or format
errors. Ask for a bullet list of bugs, or "NONE" if it looks consistent.
Do not ask it to give a new final answer yet.

Problem:
{problem}

Incumbent answer: {incumbent_answer or "(none)"}
Incumbent solution:
{clip(incumbent_solution, 2500)}

Output only the prompt text for the strong model."""

    if act == "revise":
        return f"""Write a short hint (2-4 sentences) so a strong model re-solves
the problem while fixing the listed issues. If the critique says NONE, demand
an independent re-solve with a careful format check. Do not state the answer.

Problem:
{problem}

Incumbent answer: {incumbent_answer or "(none)"}
Critique:
{clip(critique, 1500)}

Output only the hint."""

    raise ValueError(f"unknown act {act}")


def fallback_large_prompt(
    act: str,
    problem: str,
    *,
    incumbent_solution: str = "",
    incumbent_answer: str = "",
) -> str:
    if act == "plan":
        return (
            f"Problem:\n{problem}\n\n"
            "List TWO distinct solution methods (name + one sentence each). "
            "Do not compute the final answer."
        )
    if act == "critique":
        return (
            f"Problem:\n{problem}\n\n"
            f"Incumbent answer: {incumbent_answer or '(none)'}\n"
            f"Incumbent solution:\n{clip(incumbent_solution, 3000)}\n\n"
            "List concrete errors (algebra, missing cases, format). "
            "If none, reply NONE. Do not give a new final answer."
        )
    if act == "alt_solve":
        return _solve_prompt(
            problem,
            "Solve with a different method than a first-pass algebraic grind. "
            "Check the requested answer format.",
        )
    if act == "revise":
        return _solve_prompt(
            problem,
            "Re-solve independently. Recheck algebra and the requested format.",
        )
    return _solve_prompt(problem, "")


def wrap_large_prompt(act: str, problem: str, small_text: str, **kwargs) -> str:
    text = (small_text or "").strip()
    if act == "scout":
        # Floor: large model always sees a bare solve. Scout note is for later rounds.
        return _solve_prompt(problem, "")
    if act in {"plan", "critique"}:
        if len(text) < 20 or _looks_like_answer_dump(text):
            return fallback_large_prompt(act, problem, **kwargs)
        if "problem" not in text.lower():
            return f"Problem:\n{problem}\n\n{text}"
        return text
    # alt_solve / revise: small text is a hint
    if len(text) < 12:
        return fallback_large_prompt(act, problem, **kwargs)
    return _solve_prompt(problem, text)


def _solve_prompt(problem: str, hint: str) -> str:
    parts = [
        "Solve this problem.",
        "",
        f"Problem:\n{problem}",
    ]
    if hint.strip():
        parts.extend(["", f"Hint:\n{hint.strip()}"])
    parts.extend(["", "Output the final answer in this format:", "", "Final Answer: <answer>"])
    return "\n".join(parts)


def _looks_like_answer_dump(text: str) -> bool:
    low = text.lower()
    return "final answer:" in low or "<answer>" in low
