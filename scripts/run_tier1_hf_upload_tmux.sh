#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=pagent-tier1-hf-upload
LOG=/home/ivaning/PAgent/logs/tier1-hf-upload.log
mkdir -p /home/ivaning/PAgent/logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  python scripts/upload_tier1_to_hf.py 2>&1 | tee '$LOG'
  echo '=== DONE ===' | tee -a '$LOG'
"

echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG"
