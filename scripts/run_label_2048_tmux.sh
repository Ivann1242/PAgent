#!/usr/bin/env bash
# Free GPUs, deploy 4x OSS-20B, run 2048-question labeling.
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

echo "=== [1/4] Stop non-OSS GPU jobs ==="
# router :8084
pkill -f 'vllm serve /home/ivaning/PAgent/checkpoints/merged' 2>/dev/null || true
# qwen baseline :8083
pkill -f 'vllm serve ./Qwen/Qwen3-4B' 2>/dev/null || true
# old single OSS (will restart on all 4 GPUs)
pkill -f 'vllm.entrypoints.openai.api_server.*gpt-oss-20b' 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*8006' 2>/dev/null || true
# misc trainers / notebooks on GPUs
pkill -f 'cont_trainer.py.*gpu_num 0' 2>/dev/null || true
pkill -f 'trainer.py.*gpu_num' 2>/dev/null || true
pkill -f 'ipykernel_launcher' 2>/dev/null || true
sleep 8
echo "GPU status after cleanup:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo "=== [2/4] Start OSS-20B (skip GPU 3 if occupied) ==="
bash scripts/serve_oss_4gpu.sh 2>&1 | tee "$LOG_DIR/oss-4gpu-start.log" || true

# Use whichever endpoints are up (need at least 1; typically 3 if GPU 3 is busy)
mapfile -t UP_URLS < <(python - <<'PY'
import json, urllib.request
from config import ANSWER_URLS, ANSWER_MODEL
up = []
for url in ANSWER_URLS:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        if ANSWER_MODEL in ids:
            up.append(url)
            print(url)
    except Exception:
        pass
if not up:
    raise SystemExit("No OSS answerer up")
PY
)
echo "Active endpoints (${#UP_URLS[@]}): ${UP_URLS[*]}"
ANSWER_URL_ARG=$(IFS=,; echo "${UP_URLS[*]}")

echo "=== [3/4] Verify train data ==="
wc -l data/train.jsonl data/val.jsonl

echo "=== [4/4] Label 2048 x 6 actions ==="
python run.py label \
  --limit 2048 \
  --workers 32 \
  --answer-urls "$ANSWER_URL_ARG" \
  --out-dir checkpoints/label_2048 \
  2>&1 | tee "$LOG_DIR/label-2048.log"

echo "=== DONE ==="
cat checkpoints/label_2048/stats.json
