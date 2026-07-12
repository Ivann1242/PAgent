#!/usr/bin/env bash
# Re-test multi-candidate flip hints and pick one per question (repeat flip-rate).
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=pagent-blind-dedup-17k
LOG_FILE="$LOG_DIR/blind-dedup-17k.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

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
ANSWER_URL_ARG=$(IFS=,; echo "${UP_URLS[*]}")
echo "OSS endpoints (${#UP_URLS[@]}): ${UP_URLS[*]}"

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  python run.py ff-dedup-blind \
    --labels-file checkpoints/blind_hint_17k/oracle_labels.jsonl \
    --out-file checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl \
    --repeats 3 \
    --workers 32 \
    --answer-urls '${ANSWER_URL_ARG}' \
    2>&1 | tee '$LOG_FILE'
  echo '=== DONE ===' | tee -a '$LOG_FILE'
  cat checkpoints/blind_hint_17k/oracle_labels_dedup.stats.json | tee -a '$LOG_FILE'
"

echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_FILE"
echo "  out:    checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl"
