#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from core import append_jsonl, call_llm, load_jsonl, write_jsonl
from HintFlow_five.common import N_TURNS, parse_json_object

TEACHER_PROMPT = """Create supervision for a fixed five-turn math-solving process.

Problem:
{problem}

Reference final answer (teacher-only):
{gold}

Return ONLY valid JSON with exactly five objects:
{{"turns":[
  {{"step":"compact mathematical progress produced in turn 1", "hint":"short actionable instruction that would elicit this step"}},
  ...,
  {{"step":"verified finish ending with Final Answer: <answer>", "hint":"instruction to verify and finish in the required format"}}
]}}

Requirements:
- The five steps form one coherent solution and collectively reach the reference answer.
- Each hint is usable from the problem plus all preceding steps.
- A hint must guide the process without stating the final answer or copying the whole step.
- Steps 1-4 must not state Final Answer. Step 5 must end with Final Answer: <answer>.
- Keep each hint under 80 words and each step under 300 words.
"""


def parse_teacher(raw: str) -> list[dict]:
    obj = parse_json_object(raw)
    turns = obj.get("turns")
    if not isinstance(turns, list) or len(turns) != N_TURNS:
        raise ValueError("teacher output must contain exactly five turns")
    cleaned = []
    for i, item in enumerate(turns, 1):
        if not isinstance(item, dict):
            raise ValueError(f"turn {i} is not an object")
        step = str(item.get("step", "")).strip()
        hint = str(item.get("hint", "")).strip()
        if not step or not hint:
            raise ValueError(f"turn {i} has empty step/hint")
        if i < N_TURNS and "final answer:" in step.lower():
            raise ValueError(f"turn {i} prematurely contains Final Answer")
        cleaned.append({"turn": i, "step": step, "hint": hint})
    return cleaned


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="checkpoints/hintflow_five_data/teacher_questions.jsonl")
    p.add_argument("--out", default="checkpoints/hintflow_five_data/teacher_trajectories.jsonl")
    p.add_argument("--urls", default="http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1")
    p.add_argument("--model", default="qwen3-14b")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--seed", type=int, default=20260817)
    args = p.parse_args()
    rows = load_jsonl(Path(args.questions))
    out = Path(args.out)
    done = {r["id"]: r for r in load_jsonl(out)} if out.exists() else {}
    urls = [x.strip() for x in args.urls.split(",") if x.strip()]
    local = threading.local()
    write_lock = threading.Lock()

    def work(row: dict) -> dict:
        idx = int(row["id"])
        url = urls[idx % len(urls)]
        clients = getattr(local, "clients", None)
        if clients is None:
            clients = local.clients = {}
        client = clients.setdefault(url, OpenAI(base_url=url, api_key="EMPTY", timeout=3600, max_retries=0))
        raw = call_llm(
            client,
            args.model,
            TEACHER_PROMPT.format(problem=row["problem"], gold=row["gold"]),
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_object"}},
        )
        turns = parse_teacher(raw)
        return {"id": idx, "problem": row["problem"], "gold": row["gold"], "turns": turns, "raw": raw}

    todo = [r for r in rows if r["id"] not in done]
    with ThreadPoolExecutor(max_workers=min(args.workers, max(len(todo), 1))) as pool:
        futures = {pool.submit(work, row): row for row in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="teacher"):
            row = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                print(f"id={row['id']} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            done[rec["id"]] = rec
            with write_lock:
                append_jsonl(out, rec)
    ordered = [done[r["id"]] for r in rows if r["id"] in done]
    write_jsonl(out, ordered)
    print(f"teacher complete: {len(ordered)}/{len(rows)} -> {out}")
    if len(ordered) != len(rows):
        raise SystemExit("teacher generation incomplete; rerun to resume failed rows")


if __name__ == "__main__":
    main()

