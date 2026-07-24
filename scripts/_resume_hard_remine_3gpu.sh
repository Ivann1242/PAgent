#!/usr/bin/env bash
# Resume hard-remine on GPU0-2 only. Do NOT touch GPU3 (leave other users alone).
set -euo pipefail
cd /home/ivaning/PAgent

LOG=logs/hard-remine-overnight.log
OUT=checkpoints/blind_hint_hard_remine_8k_k12
IDS=checkpoints/blind_hint_17k/hard_noflip_ids.txt
# 3 OSS @ max-num-seqs=16 → keep workers under ~48
WORKERS=36
K=12
TOKENS=8192

mkdir -p logs "$OUT"

for url in \
  http://127.0.0.1:8006/v1/models \
  http://127.0.0.1:8007/v1/models \
  http://127.0.0.1:8008/v1/models
do
  curl -sf "$url" | grep -q gpt-oss-20b || { echo "FAIL $url"; exit 1; }
done
# Explicitly refuse to use :8009 / GPU3
if curl -sf --max-time 2 http://127.0.0.1:8009/v1/models >/dev/null 2>&1; then
  echo "WARN: :8009 is up; this resume script still will NOT use it." | tee -a "$LOG"
fi

echo "=== $(date -Is) resume hard remine on 3 GPUs (no GPU3) workers=${WORKERS} ===" | tee -a "$LOG"
python run.py oracle-hint \
  --data-file data/train.jsonl \
  --only-ids-file "$IDS" \
  --out-dir "$OUT" \
  --workers "$WORKERS" \
  --k "$K" \
  --hint-temp 0.9 \
  --max-tokens "$TOKENS" \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
  --protocol native \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
cat "$OUT/stats.json" | tee -a "$LOG"
