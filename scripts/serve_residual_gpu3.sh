#!/usr/bin/env bash
# Serve OSS + residual feedback model on physical GPU3 ONLY.
set -euo pipefail
cd /home/ivaning/PAgent
mkdir -p logs

GPU=3
TARGET_GPU_UUID="$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader)"
OSS_MODEL="${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}"
if [[ -n "${MERGED:-}" ]]; then
  ORCH_MODEL="$MERGED"
elif [[ -f checkpoints/residual_evolution/promoted.json ]]; then
  ORCH_MODEL="$(python -c 'import json; print(json.load(open("checkpoints/residual_evolution/promoted.json"))["model"])')"
else
  ORCH_MODEL=checkpoints/residual_feedback_merged
fi
OSS_PORT="${OSS_PORT:-8006}"
ORCH_PORT="${ORCH_PORT:-8086}"
OSS_UTIL="${OSS_UTIL:-0.58}"
ORCH_UTIL="${ORCH_UTIL:-0.18}"
OSS_MAX_LEN="${OSS_MAX_LEN:-16384}"
ORCH_MAX_LEN="${ORCH_MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

if [[ ! -d "$OSS_MODEL" ]]; then
  echo "missing OSS model: $OSS_MODEL" >&2
  exit 1
fi
if [[ ! -d "$ORCH_MODEL" ]]; then
  echo "missing residual feedback model: $ORCH_MODEL" >&2
  exit 1
fi

kill_port() {
  local port=$1
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
}

wait_model() {
  local port=$1 name=$2
  for _ in $(seq 1 180); do
    local body
    body="$(curl -sf "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)"
    if [[ "$body" == *"$name"* ]]; then
      echo "  :${port} OK ($name)"
      return 0
    fi
    sleep 5
  done
  return 1
}

kill_port "$OSS_PORT"
kill_port "$ORCH_PORT"
sleep 4

echo "Starting OSS on physical GPU3 -> :${OSS_PORT}"
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" \
  --port "$OSS_PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$OSS_UTIL" \
  --max-model-len "$OSS_MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --served-model-name gpt-oss-20b \
  > "logs/residual-oss-gpu3-${OSS_PORT}.log" 2>&1 &
OSS_PID=$!
echo "  pid=$OSS_PID"
wait_model "$OSS_PORT" gpt-oss-20b

echo "Starting residual feedback model on physical GPU3 -> :${ORCH_PORT}"
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$ORCH_MODEL" \
  --port "$ORCH_PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$ORCH_UTIL" \
  --max-model-len "$ORCH_MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --served-model-name qwen3-4b \
  > "logs/residual-orch-gpu3-${ORCH_PORT}.log" 2>&1 &
ORCH_PID=$!
echo "  pid=$ORCH_PID"
wait_model "$ORCH_PORT" qwen3-4b

for pid in "$OSS_PID" "$ORCH_PID"; do
  PROC_ENV="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null || true)"
  if [[ "$PROC_ENV" != *"CUDA_VISIBLE_DEVICES=3"* ]]; then
    echo "GPU binding verification failed for PID $pid" >&2
    exit 1
  fi
done
python - "$OSS_PID" "$ORCH_PID" "$TARGET_GPU_UUID" <<'PY'
import csv
import io
import subprocess
import sys
from pathlib import Path

roots = [int(sys.argv[1]), int(sys.argv[2])]
target_uuid = sys.argv[3].strip()
parents = {}
for status in Path("/proc").glob("[0-9]*/status"):
    try:
        fields = {}
        for line in status.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        parents[int(fields["Pid"])] = int(fields["PPid"])
    except (OSError, KeyError, ValueError):
        pass

def descends(pid, root):
    seen = set()
    while pid in parents and pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        pid = parents[pid]
    return pid == root

raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader",
    ],
    text=True,
)
apps = [(int(row[0].strip()), row[1].strip()) for row in csv.reader(io.StringIO(raw))]
for root in roots:
    matches = [(pid, uuid) for pid, uuid in apps if descends(pid, root)]
    if not matches or any(uuid != target_uuid for _, uuid in matches):
        raise SystemExit(f"GPU UUID verification failed for process tree {root}: {matches}")
print(f"GPU UUID verified: {target_uuid}")
PY
nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total --format=csv
