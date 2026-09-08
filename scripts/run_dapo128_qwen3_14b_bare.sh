#!/usr/bin/env bash
# DAPO-128 bare baseline with Qwen3-14B (4× vLLM, continuous batching).
set -euo pipefail
cd /home/ivaning/PAgent

TOKENS=${TOKENS:-20000}
WORKERS=${WORKERS:-96}
SEED=${SEED:-42}
MODEL=${SOLVER_SERVED_NAME:-qwen3-14b}
OUT=${OUT:-checkpoints/eval_qwen3_14b_bare_dapo128_${TOKENS}}
LOG=${LOG:-logs/eval-qwen3-14b-bare-dapo128.log}
URLS=${SOLVER_URLS:-http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1}

mkdir -p logs "$OUT"

for port in 8006 8007 8008 8009; do
  curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q "$MODEL" \
    || { echo "solver :${port} not ready ($MODEL)" >&2; exit 1; }
done

echo "=== $(date -Is) DAPO128 bare Qwen3-14B tokens=${TOKENS} workers=${WORKERS} ===" | tee "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json, time
from pathlib import Path
from openai import OpenAI
from config import EVAL_PARQUET, Config
from core import load_dapo_rows, write_jsonl, call_llm, build_large_prompt, extract_final_answer, exact_match
from eval import _run_parallel, _metrics
from label import _AnswererPool

tokens = $TOKENS
workers = $WORKERS
seed0 = $SEED
model = "$MODEL"
out = Path("$OUT")
urls = [u.strip() for u in "$URLS".split(",") if u.strip()]
rows = load_dapo_rows(EVAL_PARQUET)[:128]
pool = _AnswererPool(urls, model)
print(f"n={len(rows)} urls={len(urls)} workers={workers} max_tokens={tokens} model={model}", flush=True)

def fn(row):
    client = pool.next_client()
    prompt = build_large_prompt(row["problem"], "")
    local_seed = seed0 + int(row["id"])
    try:
        text = call_llm(
            client, model, prompt,
            temperature=0.0, max_tokens=tokens,
            extra_body={"seed": local_seed},
        )
        pred = extract_final_answer(text)
        em = exact_match(pred, row["gold"])
        return {
            "id": row["id"], "problem": row["problem"], "gold": row["gold"],
            "prompt": prompt, "solution": text, "pred": pred, "em": int(em),
            "error": None, "seed": local_seed,
        }
    except Exception as e:
        return {
            "id": row["id"], "problem": row["problem"], "gold": row["gold"],
            "prompt": prompt, "solution": "", "pred": "", "em": 0,
            "error": f"{type(e).__name__}: {e}", "seed": local_seed,
        }

t0 = time.time()
recs = _run_parallel(rows, fn, workers=workers, desc="dapo128_bare_qwen3_14b",
                     resume_path=out / "live_baseline.jsonl")
write_jsonl(out / "live_baseline.jsonl", recs)
m = _metrics(recs)
summary = {
    "meta": {
        "data": str(EVAL_PARQUET), "n": len(rows),
        "max_tokens": tokens, "workers": workers,
        "solver_model": model, "solver_urls": urls,
        "seed": seed0, "protocol": "bare_native_qwen3_14b",
    },
    "live_baseline": m,
    "elapsed_sec": round(time.time() - t0, 1),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
print("DONE", out, flush=True)
PY

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
echo "out: $OUT/summary.json"
