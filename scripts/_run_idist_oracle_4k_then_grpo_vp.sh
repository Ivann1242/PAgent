#!/usr/bin/env bash
# 1) IID Oracle hint replay @4k on 1809 (4x OSS)
# 2) Resume virtual-positive GRPO from saved step 40 (GPU0=policy, GPU1/2/3=OSS)
set -euo pipefail
cd /home/ivaning/PAgent

MASTER_LOG=${MASTER_LOG:-logs/idist-oracle-4k-then-grpo-vp.log}
LABELS=checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl
ORACLE_OUT=checkpoints/eval_idist_oraclehint_4k
ORACLE_LOG=logs/eval-idist-oraclehint-4k.log
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}
TOKENS=4096
OSS_MAX_LEN=16384
MAX_NUM_SEQS=16
WORKERS=64

GRPO_SESSION=pagent-blind-ff-grpo-vp320
GRPO_SCRIPT=scripts/run_blind_ff_grpo_virtual_positive_320.sh

mkdir -p logs "$ORACLE_OUT"

wait_model() {
  local url=$1 name=$2
  for _ in $(seq 1 180); do
    if curl -sf "$url/models" | grep -q "$name"; then
      echo "  $url $name OK"
      return 0
    fi
    sleep 3
  done
  echo "FAIL $url $name"
  return 1
}

kill_port() {
  pkill -f "vllm.entrypoints.openai.api_server.*--port $1" 2>/dev/null || true
}

serve_oss() {
  local gpu=$1 port=$2 util=${3:-0.92} maxlen=${4:-$OSS_MAX_LEN} seqs=${5:-$MAX_NUM_SEQS}
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port "$port" --tensor-parallel-size 1 \
    --gpu-memory-utilization "$util" --max-model-len "$maxlen" --max-num-seqs "$seqs" \
    --served-model-name gpt-oss-20b \
    > "logs/oss-oracle4k-gpu${gpu}-${port}.log" 2>&1 &
  echo "  OSS gpu$gpu :$port pid=$! util=$util maxlen=$maxlen"
}

{
  echo "=== $(date -Is) start: IID oracle @${TOKENS} then GRPO resume ==="
  df -h /home/ivaning | tail -1
  free -h | head -2
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$MASTER_LOG"

echo "=== $(date -Is) stop old vLLM / leftover GPU apps ===" | tee -a "$MASTER_LOG"
for port in 8006 8007 8008 8009 8086; do kill_port "$port"; done
pkill -f 'python run.py ff-train --init-adapter checkpoints/blind_ff_grpo_17k_virtual_positive_adapter' 2>/dev/null || true
sleep 4
for gpu in 0 1 2 3; do
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
done
sleep 4

echo "=== $(date -Is) phase1: 4x OSS for oracle_hint @${TOKENS} ===" | tee -a "$MASTER_LOG"
# Start one-by-one to avoid host-RAM spikes while other users hog CPU RAM.
serve_oss 0 8006
wait_model http://127.0.0.1:8006/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
serve_oss 1 8007
wait_model http://127.0.0.1:8007/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
serve_oss 2 8008
wait_model http://127.0.0.1:8008/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
serve_oss 3 8009
wait_model http://127.0.0.1:8009/v1 gpt-oss-20b | tee -a "$MASTER_LOG"

echo "=== $(date -Is) iid oracle_hint n=1809 max_tokens=${TOKENS} ===" | tee -a "$MASTER_LOG" | tee "$ORACLE_LOG"
python eval_idist.py \
  --labels-file "$LABELS" \
  --out-dir "$ORACLE_OUT" \
  --router-model qwen3-4b-blind-ff-17k \
  --router-url http://127.0.0.1:8086/v1 \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1 \
  --max-tokens "$TOKENS" \
  --workers "$WORKERS" \
  --modes oracle_hint \
  2>&1 | tee -a "$ORACLE_LOG" | tee -a "$MASTER_LOG"

python3 - <<PY | tee -a "$MASTER_LOG"
import json
from pathlib import Path
p = Path("$ORACLE_OUT/summary.json")
s = json.loads(p.read_text())
em = s["oracle_hint"]["em"]
n = s["meta"]["n_questions"]
tok = s["meta"]["max_tokens"]
print(f"ORACLE_DONE @ {tok}: EM={em*100:.2f}% n={n}")
# contrast with 20k oracle if present
p20 = Path("checkpoints/eval_idist_oraclehint_20k/summary.json")
if p20.exists():
    e20 = json.loads(p20.read_text())["oracle_hint"]["em"]
    print(f"ref oracle@20k: {e20*100:.2f}%  delta(4k-20k)={(em-e20)*100:+.2f}pp")
PY

echo "=== $(date -Is) phase2: reconfigure OSS for GRPO (GPU0 free for policy) ===" | tee -a "$MASTER_LOG"
for port in 8006 8007 8008 8009; do kill_port "$port"; done
sleep 5
for gpu in 0 1 2 3; do
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
done
sleep 5

# GPU1/2/3 = reward OSS; GPU0 reserved for Qwen policy (no shared OSS needed).
bash scripts/serve_oss_grpo_reward.sh 2>&1 | tee -a "$MASTER_LOG"

echo "=== $(date -Is) phase3: resume virtual-positive GRPO from step40 ===" | tee -a "$MASTER_LOG"
python3 - <<'PY' | tee -a "$MASTER_LOG"
import json
from pathlib import Path
st = json.loads(Path("checkpoints/blind_ff_grpo_17k_virtual_positive_adapter/grpo_train_state.json").read_text())
print(f"grpo_state step={st['step']} cursor={st['cursor']} -> resume start_step={st['step']+1}")
PY

bash "$GRPO_SCRIPT" 2>&1 | tee -a "$MASTER_LOG"

echo "=== $(date -Is) GRPO tmux launched; chain launcher exiting ===" | tee -a "$MASTER_LOG"
echo "monitor oracle:  tail -f $ORACLE_LOG"
echo "monitor grpo:    tmux attach -t $GRPO_SESSION  OR  tail -f logs/blind-ff-grpo-17k-vp320-train.log"
echo "master:          tail -f $MASTER_LOG"
