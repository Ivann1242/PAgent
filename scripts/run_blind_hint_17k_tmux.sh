#!/usr/bin/env bash
# 4x OSS-20B parallel blind hint labeling on full 17K train set.
# Same settings as blind_hint_2048: k=6, hint_temp=0.8, workers=40, native protocol.
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=pagent-blind-hint-17k
TRAIN_SIZE=17000
VAL_SIZE=256
OUT_DIR=checkpoints/blind_hint_17k
LOG_FILE="$LOG_DIR/blind-hint-17k.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

echo "=== [1/5] Stop non-OSS GPU jobs ==="
pkill -f 'vllm serve checkpoints/blind_ff_grpo_merged' 2>/dev/null || true
pkill -f 'vllm serve checkpoints/blind_ff_sft' 2>/dev/null || true
pkill -f 'vllm serve /home/ivaning/PAgent/checkpoints/merged' 2>/dev/null || true
pkill -f 'vllm serve ./Qwen/Qwen3-4B' 2>/dev/null || true
pkill -f 'run.py ff-train' 2>/dev/null || true
pkill -f 'eval_repeat.py' 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*gpt-oss-20b' 2>/dev/null || true
sleep 8
echo "GPU status after cleanup:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo "=== [2/5] Start 4x OSS-20B ==="
bash scripts/serve_oss_4gpu.sh 2>&1 | tee "$LOG_DIR/oss-4gpu-start-17k.log"

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
if len(up) < 4:
    raise SystemExit(f"Need 4 OSS endpoints, got {len(up)}: {up}")
PY
)
echo "Active endpoints (${#UP_URLS[@]}): ${UP_URLS[*]}"
ANSWER_URL_ARG=$(IFS=,; echo "${UP_URLS[*]}")

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent

  echo '=== [3/5] Prepare train=${TRAIN_SIZE} val=${VAL_SIZE} ===' | tee -a '$LOG_FILE'
  python run.py prepare --train-size ${TRAIN_SIZE} --val-size ${VAL_SIZE} 2>&1 | tee -a '$LOG_FILE'

  echo '=== [4/5] Verify data ===' | tee -a '$LOG_FILE'
  wc -l data/train.jsonl data/val.jsonl | tee -a '$LOG_FILE'

  echo '=== [5/5] Blind hint label (oracle-hint) ===' | tee -a '$LOG_FILE'
  python run.py oracle-hint \
    --limit ${TRAIN_SIZE} \
    --workers 40 \
    --k 6 \
    --hint-temp 0.8 \
    --answer-urls '${ANSWER_URL_ARG}' \
    --out-dir ${OUT_DIR} \
    2>&1 | tee -a '$LOG_FILE'

  echo '=== DONE ===' | tee -a '$LOG_FILE'
  cat ${OUT_DIR}/stats.json | tee -a '$LOG_FILE'
"

echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_FILE"
echo "  out:    $OUT_DIR/"
