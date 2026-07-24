#!/usr/bin/env bash
# HintFlow_one on iid 1809 @ fair 20k.
set -euo pipefail
cd /home/ivaning/PAgent

LOG=logs/eval-hintflow-one-idist-20k.log
OUT=checkpoints/eval_hintflow_one_idist_1809_20k
LABELS=checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl

mkdir -p logs "$OUT"

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 60); do
    curl -sf "$url" | grep -q "$name" && { echo "  $url OK"; return 0; }
    sleep 2
  done
  echo "FAIL $url"; return 1
}

echo "=== $(date -Is) HintFlow_one iid@20k ===" | tee "$LOG"
wait_model http://127.0.0.1:8086/v1/models qwen3-4b-blind-ff-17k | tee -a "$LOG"
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8008/v1/models gpt-oss-20b | tee -a "$LOG"

python HintFlow_one/eval_one.py \
  --data-file "$LABELS" \
  --out-dir "$OUT" \
  --orch-url http://127.0.0.1:8086/v1 \
  --orch-model qwen3-4b-blind-ff-17k \
  --solver-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
  --solver-model gpt-oss-20b \
  --solver-max-tokens 20000 \
  --selector-mode orch \
  --replace-threshold 0.90 \
  --seed 41 \
  --workers 48 \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
cat "$OUT/summary.json" | tee -a "$LOG"

python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
print("\n=== compare iid HF1 ===")
for tag,p in [("4k","checkpoints/eval_hintflow_one_idist_1809_4k/summary.json"),
              ("8k","checkpoints/eval_hintflow_one_idist_1809_8k/summary.json"),
              ("20k","checkpoints/eval_hintflow_one_idist_1809_20k/summary.json")]:
    if not Path(p).exists():
        print(tag, "MISSING"); continue
    h=json.load(open(p))["hintflow_one"]
    print(f"{tag}: final={h['em']*100:.1f}% base={h['baseline_em']*100:.1f}% chal={h['challenger_em']*100:.1f}% "
          f"d={h['paired_delta']*100:+.1f}pp rec={h['recovered']} harm={h['harmed']}")
o=json.load(open("checkpoints/eval_idist_oraclehint_20k/summary.json"))["oracle_hint"]["em"]
print(f"oracle@20k: {o*100:.1f}%")
PY
