#!/usr/bin/env bash
# Full 17k blind-hint labeling with Qwen3-14B.
# Old recipe: k=6, hint_temp=0.8, max_tokens=8192, native.
# Repro: VLLM_BATCH_INVARIANT=1 + sticky id%n + fixed seeds.
# Throughput: enable_thinking=false, max_model_len=16384, max_num_seqs=48, workers=128.
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=${SESSION:-pagent-blind-hint-17k-qwen3-14b-nothink}
TRAIN_SIZE=${TRAIN_SIZE:-17000}
VAL_SIZE=${VAL_SIZE:-256}
OUT_DIR=${OUT_DIR:-checkpoints/blind_hint_17k_qwen3_14b_nothink}
LOG_FILE=${LOG_FILE:-logs/blind-hint-17k-qwen3-14b-nothink.log}
MODEL=${SOLVER_MODEL_PATH:-/home/ivaning/models/Qwen3-14B}
SERVED=${SOLVER_SERVED_NAME:-qwen3-14b}
MAX_LEN=${SOLVER_MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${SOLVER_MAX_NUM_SEQS:-48}
WORKERS=${WORKERS:-128}
K=${K:-6}
HINT_TEMP=${HINT_TEMP:-0.8}
TOKENS=${TOKENS:-8192}
SEED=${SEED:-42}
UTIL=${GPU_MEM_UTIL:-0.92}

mkdir -p logs "$OUT_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

echo "=== [1/4] Restart 4x Qwen3-14B (batch_invariant, seqs=${MAX_NUM_SEQS}, len=${MAX_LEN}) ==="
# Kill by PID (avoid pkill -f self-match)
pids=$(ps aux | awk '/vllm.entrypoints.openai.api_server.*Qwen3-14B/ && !/awk/ {print $2}')
if [[ -n "${pids}" ]]; then
  echo "killing: ${pids}"
  kill ${pids} || true
  sleep 8
fi

for pair in 0:8006 1:8007 2:8008 3:8009; do
  gpu=${pair%%:*}
  port=${pair##*:}
  echo "  GPU${gpu} -> :${port}"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_BATCH_INVARIANT=1 \
    nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --port "$port" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$UTIL" \
      --max-model-len "$MAX_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --seed 0 \
      --served-model-name "$SERVED" \
      > "logs/qwen3-14b-${port}.log" 2>&1 &
  echo "    pid=$!"
done

for port in 8006 8007 8008 8009; do
  ok=0
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q "$SERVED"; then
      echo "  :${port} OK"
      ok=1
      break
    fi
    sleep 5
  done
  if [[ "$ok" -eq 0 ]]; then
    echo "  :${port} FAILED" >&2
    tail -40 "logs/qwen3-14b-${port}.log" >&2
    exit 1
  fi
done
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

ANSWER_URLS="http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1"

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  exec > >(tee -a '$LOG_FILE') 2>&1
  echo \"=== \$(date -Is) blind-hint 17k Qwen3-14B ===\"
  echo \"out=$OUT_DIR workers=$WORKERS k=$K hint_temp=$HINT_TEMP max_tokens=$TOKENS seed=$SEED\"
  echo \"servers: batch_invariant max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_LEN\"

  echo '=== [2/4] Prepare train/val ==='
  python run.py prepare --train-size ${TRAIN_SIZE} --val-size ${VAL_SIZE}
  wc -l data/train.jsonl data/val.jsonl

  echo '=== [3/4] Blind hint label (sticky + seeds) ==='
  python run.py oracle-hint \\
    --limit ${TRAIN_SIZE} \\
    --workers ${WORKERS} \\
    --k ${K} \\
    --hint-temp ${HINT_TEMP} \\
    --max-tokens ${TOKENS} \\
    --seed ${SEED} \\
    --answer-urls '${ANSWER_URLS}' \\
    --out-dir ${OUT_DIR}

  echo '=== [4/4] DONE ==='
  date -Is
  cat ${OUT_DIR}/stats.json
"

echo "started tmux: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_FILE"
echo "  out:    $OUT_DIR/"
