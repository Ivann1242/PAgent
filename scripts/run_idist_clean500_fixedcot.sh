#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

SOURCE=checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean_train_sample500.jsonl
LABELS=checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean_train_sample500_fixedcot.jsonl
OUT=checkpoints/eval_idist_qwen3_14b_nothink_sft_train500_fixedcot
LOG=logs/eval-idist-clean500-fixedcot.log
COT='Think step by step. Write a clear chain of reasoning before the final answer. Double-check arithmetic and algebraic manipulations. Do not skip intermediate steps.'

mkdir -p "$OUT" logs

python3 -c 'import json, sys
source, target, cot = sys.argv[1:]
with open(source) as src, open(target, "w") as dst:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        row["label_hint"] = cot
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
' "$SOURCE" "$LABELS" "$COT"

echo "=== $(date -Is) fixed_cot ablation n=500 (no baseline) ===" | tee "$LOG"
python eval_idist.py \
  --labels-file "$LABELS" \
  --out-dir "$OUT" \
  --router-model unused \
  --answer-urls http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
  --modes oracle_hint \
  --workers 64 \
  --max-tokens 8192 \
  --protocol native \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) ALL DONE ===" | tee -a "$LOG"
