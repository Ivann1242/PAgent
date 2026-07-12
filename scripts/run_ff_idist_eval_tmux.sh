#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=pagent-ff-idist-eval
LOG_FILE=/home/ivaning/PAgent/logs/ff-idist-eval.log

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

chmod +x scripts/run_ff_idist_eval_runner.sh
tmux new-session -d -s "$SESSION" bash scripts/run_ff_idist_eval_runner.sh

echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_FILE"
