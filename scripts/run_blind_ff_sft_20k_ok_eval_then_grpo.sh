#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent
SESSION=${SESSION:-pagent-sft20k-eval-then-grpo}
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true
tmux new-session -d -s "$SESSION" bash scripts/_run_sft20k_eval_then_grpo_inner.sh
echo "started tmux: $SESSION"
echo "  eval: tail -f logs/eval-blind-ff-sft-20k-ok-128-repeat.log"
echo "  then GRPO resumes at step 18"
echo "  out:  checkpoints/eval_blind_ff_sft_20k_ok_128_repeat/aggregate.json"
