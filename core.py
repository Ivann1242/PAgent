"""Prompts, parsing, reward, data loading, and LLM helpers."""

from __future__ import annotations

import json
import random
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import pandas as pd
from openai import OpenAI

ACTION_SPACE = {
    "baseline": "",
    "careful_reading": (
        "Read the problem carefully. Identify all given quantities, constraints, "
        "and what is being asked before solving."
    ),
    "step_by_step": (
        "Reason step by step internally, but output only the final answer in the required format."
    ),
    "format_first": (
        "Determine the required answer format first, then derive the final result in that exact format."
    ),
    "reject_unsupported": (
        "Do not introduce facts or intermediate steps that are not supported by the problem."
    ),
    "type_aware": (
        "Identify the problem type and use a strategy suited to that type."
    ),
}

ACTION_KEYS = list(ACTION_SPACE.keys())

# Prompt-R1 Table 2 / baseline_direct_inference.py protocol
OSS_SYSTEM_PROMPT = (
    "You are a helpful assistant. Please read the provided content "
    "(including previous conversations and the current task) and help the "
    "user complete the task or answer the question."
)
PAPER_ANSWER_SUFFIX = (
    "Please provide your final answer in the following format:\n"
    "<answer>(final answer for the question)</answer>"
)


def format_action_space() -> str:
    lines = []
    for key, hint in ACTION_SPACE.items():
        desc = hint or "(no hint)"
        lines.append(f"- {key}: {desc}")
    return "\n".join(lines)


def build_optimizer_prompt(problem: str) -> str:
    return f"""You are a prompt optimizer for a math solver.

Problem:
{problem}

Write a short hint or strategy (2-4 sentences) to help a large model solve this problem.
Do NOT include the final answer or any specific numeric result from the solution.

Output only the hint text, no JSON, no preamble."""


def build_small_prompt(problem: str) -> str:
    return f"""You are an action router.

Problem:
{problem}

Action Space:
{format_action_space()}

Choose exactly one action key from the action space.

Output only JSON:
{{"action": "<action_key>"}}"""


def build_large_prompt(problem: str, hint: str) -> str:
    parts = [
        "Solve this problem.",
        "",
        f"Problem:\n{problem}",
    ]
    if hint.strip():
        parts.extend(["", f"Hint:\n{hint}"])
    parts.extend([
        "",
        "Output the final answer in this format:",
        "",
        "Final Answer: <answer>",
    ])
    return "\n".join(parts)


def build_paper_answer_prompt(problem: str, hint: str = "") -> str:
    """Align with Prompt-R1 baseline_direct_inference.py (Table 2)."""
    parts = [problem]
    if hint.strip():
        parts.extend(["", f"Hint: {hint}"])
    parts.extend(["", PAPER_ANSWER_SUFFIX])
    return "\n".join(parts)


def extract_paper_answer(text: str) -> str:
    match = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match[-1].strip()
    return text.strip()


def check_paper_format(text: str) -> bool:
    return bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))


def format_router_input(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tokenizer.apply_chat_template(
                messages, **kwargs, enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    return prompt


def _strip_thinking(text: str) -> str:
    close = "</" + "redacted_thinking" + ">"
    if close in text:
        return text.split(close)[-1].strip()
    return text.strip()


def parse_action(text: str) -> tuple[str, bool]:
    """Return (action_key, parse_ok). Fallback to baseline if parse fails."""
    if not text:
        return "baseline", False
    text = _strip_thinking(text)
    candidates: list[str] = []
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        candidates.append(match.group(0))
    if text.strip().startswith("{"):
        candidates.append(text.strip())
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        action = str(obj.get("action", "")).strip()
        if action in ACTION_SPACE:
            return action, True
    m = re.search(r'"action"\s*:\s*"([a-z_]+)"', text)
    if m and m.group(1) in ACTION_SPACE:
        return m.group(1), True
    return "baseline", False


def check_format(text: str) -> bool:
    return bool(re.search(r"Final Answer:\s*\S", text, re.IGNORECASE))


def _extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} with light brace matching."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    i = start + len(marker)
    depth = 1
    j = i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return text[i : j - 1].strip()


def _clean_answer_span(text: str) -> str:
    span = (text or "").strip()
    for _ in range(3):
        changed = False
        if span.startswith("**") and span.endswith("**") and len(span) > 4:
            span = span[2:-2].strip()
            changed = True
        for left, right in (
            ("\\(", "\\)"),
            ("\\[", "\\]"),
            ("$", "$"),
        ):
            if span.startswith(left) and span.endswith(right):
                span = span[len(left) : -len(right)].strip()
                changed = True
        boxed = _extract_boxed(span)
        if boxed is not None and span.lstrip().startswith("\\boxed{"):
            span = boxed.strip()
            changed = True
        if not changed:
            break
    span = span.strip().strip("*").strip()
    return span.rstrip(" \t\r\n")


def extract_final_answer(text: str) -> str:
    # Supports plain, Markdown-bold, boxed, and next-line final answers.
    label = re.compile(
        r"(?:^|\n)\s*(?:\*\*)?\s*Final\s+Answer\s*:\s*(?:\*\*)?",
        re.IGNORECASE,
    )
    matches = list(label.finditer(text))
    if matches:
        tail = text[matches[-1].end() :].lstrip()
        if tail.startswith("\\[") and "\\]" in tail:
            candidate = tail[: tail.find("\\]") + 2]
        elif tail.startswith("\\(") and "\\)" in tail:
            candidate = tail[: tail.find("\\)") + 2]
        elif tail.startswith("\\boxed{"):
            boxed = _extract_boxed(tail)
            candidate = boxed if boxed is not None else tail.splitlines()[0]
        else:
            candidate = next(
                (line.strip() for line in tail.splitlines() if line.strip()),
                "",
            )
        candidate = _clean_answer_span(candidate)
        if candidate:
            return candidate
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return _clean_answer_span(m[-1])
    boxed = _extract_boxed(text)
    if boxed is not None:
        # strip trivial wrappers like xyz=1004
        boxed = re.sub(r"^[A-Za-z]+\s*=\s*", "", boxed).strip()
        return _clean_answer_span(boxed)
    m = re.findall(r"^Answer:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        return _clean_answer_span(m[-1])
    return text.strip()


def has_parseable_answer(text: str) -> bool:
    """True iff observation contains an explicit short final answer span."""
    if not text or not text.strip():
        return False
    if re.search(r"Final Answer:\s*\S", text, re.IGNORECASE):
        return True
    if re.search(r"<answer>\s*\S", text, re.IGNORECASE):
        return True
    if _extract_boxed(text) is not None:
        return True
    if re.search(r"^Answer:\s*\S", text, re.MULTILINE | re.IGNORECASE):
        pred = extract_final_answer(text)
        return bool(pred) and pred.strip() != text.strip() and len(pred) <= 200
    return False


def normalize_answer(s: str) -> str:
    """Normalize presentation while preserving mathematical structure and signs."""
    text = (s or "").strip().lower()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"^\s*(?:final\s+)?answer\s*[:=]\s*", "", text)
    for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("$", "$"), ("**", "**")):
        if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
            text = text[len(left) : len(text) - len(right)].strip()
    text = text.rstrip(" \t\r\n.,;")
    text = re.sub(r"\s+", "", text)
    return text


def _numeric_answer(s: str) -> Fraction | None:
    text = normalize_answer(s)
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    # Simple scalar LaTeX fractions cover the dominant DAPO answer format.
    match = re.fullmatch(
        r"([+-]?)\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}",
        text,
    )
    if match:
        sign = -1 if match.group(1) == "-" else 1
        try:
            return sign * Fraction(Decimal(match.group(2))) / Fraction(
                Decimal(match.group(3))
            )
        except (InvalidOperation, ZeroDivisionError):
            return None
    if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", text):
        numerator, denominator = text.split("/", 1)
        try:
            return Fraction(int(numerator), int(denominator))
        except (ValueError, ZeroDivisionError):
            return None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", text):
        try:
            return Fraction(Decimal(text))
        except InvalidOperation:
            return None
    return None


def exact_match(pred: str, gold: str) -> int:
    pred_number = _numeric_answer(pred)
    gold_number = _numeric_answer(gold)
    if pred_number is not None and gold_number is not None:
        return int(pred_number == gold_number)
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_reward(
    pred: str,
    gold: str,
    parse_ok: bool,
    large_output: str,
) -> tuple[float, int, int]:
    em = exact_match(pred, gold)
    format_ok = check_format(large_output)
    # EM-primary: binary outcome reward; invalid router JSON gets 0 (same as wrong).
    reward = float(em) if parse_ok else 0.0
    return reward, em, int(format_ok)


def hint_leaks_gold(hint: str, gold: str) -> bool:
    h = normalize_answer(hint)
    g = normalize_answer(gold)
    if not g or len(g) < 2:
        return False
    return g in h


def parse_optimizer_output(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    hint = _strip_thinking(text).strip()
    if hint.lower() in {"(no hint)", "no hint"}:
        return "", True
    return hint, bool(hint)


def compute_ff_reward(
    pred: str,
    gold: str,
    hint: str,
    large_output: str,
) -> tuple[float, int, int]:
    em = exact_match(pred, gold)
    format_ok = check_format(large_output)
    reward = float(em)
    if hint_leaks_gold(hint, gold):
        reward = 0.0
    return reward, em, int(format_ok)


def extract_raw_question(content: str) -> str:
    marker = "\nFirst, provide"
    if marker in content:
        return content.split(marker, 1)[0].strip()
    return content.strip()


def load_dapo_rows(parquet_path: Path) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    rows = []
    for data_id, row in df.iterrows():
        content = row["prompt"][0]["content"]
        gold = row["reward_model"]["ground_truth"]
        if hasattr(gold, "__len__") and not isinstance(gold, str):
            gold = gold[0]
        rows.append({
            "id": int(data_id),
            "problem": extract_raw_question(content),
            "gold": str(gold),
        })
    return rows


def load_baseline_predictions(res_path: Path) -> dict[int, str]:
    data = json.loads(res_path.read_text())
    return {int(r["data_id"]): r["predicted_answer"] for r in data}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    extra_body: dict | None = None,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if content:
        return content
    for key in ("reasoning", "reasoning_content"):
        val = getattr(msg, key, None)
        if val and str(val).strip():
            return str(val).strip()
    return ""


def rollout(
    answer_client: OpenAI,
    answer_model: str,
    problem: str,
    gold: str,
    action: str,
    *,
    small_output: str = "",
    parse_ok: bool = True,
    max_tokens: int = 4096,
    protocol: str = "native",
) -> dict:
    hint = ACTION_SPACE[action]
    if protocol == "paper":
        large_prompt = build_paper_answer_prompt(problem, hint)
        large_output = call_llm(
            answer_client, answer_model, large_prompt,
            system=OSS_SYSTEM_PROMPT,
            temperature=0.0, max_tokens=max_tokens,
        )
        pred = extract_paper_answer(large_output)
        em = exact_match(pred, gold)
        format_ok = int(check_paper_format(large_output))
        reward = float(em)
    else:
        large_prompt = build_large_prompt(problem, hint)
        large_output = call_llm(
            answer_client, answer_model, large_prompt,
            temperature=0.0, max_tokens=max_tokens,
        )
        pred = extract_final_answer(large_output)
        reward, em, format_ok = compute_reward(pred, gold, parse_ok, large_output)
    return {
        "problem": problem,
        "gold_answer": gold,
        "selected_action": action,
        "hint": hint,
        "small_output": small_output,
        "large_output": large_output,
        "pred_answer": pred,
        "em": em,
        "format_ok": format_ok,
        "reward": reward,
        "protocol": protocol,
    }


def random_action() -> str:
    return random.choice(ACTION_KEYS)


def rollout_ff(
    answer_client: OpenAI,
    answer_model: str,
    problem: str,
    gold: str,
    hint: str,
    *,
    small_output: str = "",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    protocol: str = "native",
) -> dict:
    hint = hint.strip()
    if protocol == "paper":
        large_prompt = build_paper_answer_prompt(problem, hint)
        large_output = call_llm(
            answer_client, answer_model, large_prompt,
            system=OSS_SYSTEM_PROMPT,
            temperature=temperature, max_tokens=max_tokens,
        )
        pred = extract_paper_answer(large_output)
        em = exact_match(pred, gold)
        format_ok = int(check_paper_format(large_output))
        reward = float(em)
        if hint_leaks_gold(hint, gold):
            reward = 0.0
    else:
        large_prompt = build_large_prompt(problem, hint)
        large_output = call_llm(
            answer_client, answer_model, large_prompt,
            temperature=temperature, max_tokens=max_tokens,
        )
        pred = extract_final_answer(large_output)
        reward, em, format_ok = compute_ff_reward(pred, gold, hint, large_output)
    return {
        "problem": problem,
        "gold_answer": gold,
        "selected_action": "freeform",
        "hint": hint,
        "small_output": small_output,
        "large_output": large_output,
        "pred_answer": pred,
        "em": em,
        "format_ok": format_ok,
        "reward": reward,
        "protocol": protocol,
        "mode": "freeform",
    }
