#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

SOLVER_URLS="${SOLVER_URLS:-http://127.0.0.1:8008/v1}"
WORKERS="${WORKERS:-8}"
PYTHON="${PYTHON:-python}"

run_eval() {
  local mode="$1"
  local limit="$2"
  local out_dir="$3"
  echo "===== $(date -Is) mode=${mode} limit=${limit} out=${out_dir} ====="
  "${PYTHON}" HintFlow_draft/eval_draft.py \
    --mode "${mode}" \
    --limit "${limit}" \
    --workers "${WORKERS}" \
    --solver-urls "${SOLVER_URLS}" \
    --out-dir "${out_dir}"
}

run_eval full 64 checkpoints/eval_hintflow_draft_holdout64_v1
run_eval baseline 64 checkpoints/eval_hintflow_draft_holdout64_v1
run_eval full 512 checkpoints/eval_hintflow_draft_holdout512_v1
run_eval baseline 512 checkpoints/eval_hintflow_draft_holdout512_v1

echo "===== $(date -Is) all draft evals finished ====="
