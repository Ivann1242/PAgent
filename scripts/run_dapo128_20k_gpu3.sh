#!/usr/bin/env bash
# DAPO-128 @ 20k on GPU3 serves: classic Blind FF + HintFlow_one.
# Does not restart vLLM; GPU3 only.
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=${SESSION:-pagent-dapo128-20k-gpu3}
LOG=${LOG:-logs/eval-dapo128-20k-gpu3.log}
TOKENS=${TOKENS:-20000}
WORKERS=${WORKERS:-1}
ORCH_MODEL=${ORCH_MODEL:-qwen3-4b-blind-ff-17k}
SEED=${SEED:-42}
CLASSIC_OUT=${CLASSIC_OUT:-checkpoints/eval_blind_ff_dapo128_20k}
HFONE_OUT=${HFONE_OUT:-checkpoints/eval_hintflow_one_dapo128_20k_seed${SEED}}

mkdir -p logs "$CLASSIC_OUT" "$HFONE_OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing $SESSION"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  exec > >(tee $LOG) 2>&1
  echo \"=== \$(date -Is) DAPO-128 @ ${TOKENS} tokens ===\"
  curl -sf http://127.0.0.1:8006/v1/models | grep -q gpt-oss-20b
  curl -sf http://127.0.0.1:8086/v1/models | grep -q $ORCH_MODEL

  echo '=== 1/2 classic live_baseline + ff_router ==='
  python - <<'PY'
import json
from pathlib import Path
from openai import OpenAI
from config import EVAL_PARQUET, Config
from core import load_dapo_rows, write_jsonl
from eval import _metrics, eval_ff_router, eval_live_baseline

tokens = $TOKENS
workers = $WORKERS
out = Path('$CLASSIC_OUT')
out.mkdir(parents=True, exist_ok=True)
rows = load_dapo_rows(EVAL_PARQUET)[:128]
cfg = Config()
answer = OpenAI(base_url='http://127.0.0.1:8006/v1', api_key='EMPTY')
router = OpenAI(base_url='http://127.0.0.1:8086/v1', api_key='EMPTY')

base = eval_live_baseline(
    rows, answer, cfg.answer_model, protocol='native',
    workers=workers, max_tokens=tokens,
)
write_jsonl(out / 'live_baseline.jsonl', base)
ff = eval_ff_router(
    rows, router, answer,
    router_model='$ORCH_MODEL', answer_model=cfg.answer_model,
    protocol='native', workers=workers, max_tokens=tokens,
)
write_jsonl(out / 'ff_router.jsonl', ff)
summary = {
    'meta': {
        'data': str(EVAL_PARQUET), 'n': len(rows),
        'max_tokens': tokens, 'workers': workers,
        'router_model': '$ORCH_MODEL',
    },
    'live_baseline': _metrics(base),
    'ff_router': _metrics(ff),
    'ff_router_vs_live_baseline': _metrics(ff)['em'] - _metrics(base)['em'],
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY

  echo '=== 2/2 HintFlow_one ==='
  python HintFlow_one/eval_one.py \\
    --data-file data/DAPO-Math.parquet \\
    --limit 128 \\
    --workers $WORKERS \\
    --solver-urls http://127.0.0.1:8006/v1 \\
    --orch-url http://127.0.0.1:8086/v1 \\
    --orch-model $ORCH_MODEL \\
    --solver-model gpt-oss-20b \\
    --solver-max-tokens $TOKENS \\
    --selector-mode orch \\
    --replace-threshold 0.90 \\
    --seed $SEED \\
    --out-dir $HFONE_OUT

  echo '=== DONE ==='
  echo classic: $CLASSIC_OUT/summary.json
  echo hfone:   $HFONE_OUT/summary.json
"

echo "started tmux: $SESSION"
echo "  log: tail -f $LOG"
echo "  classic out: $CLASSIC_OUT/summary.json"
echo "  hfone out:   $HFONE_OUT/summary.json"
echo
echo "Note: prior HintFlow_one 128@20k seed41 already exists at"
echo "  checkpoints/eval_hintflow_one_orch_blindff17k_gpu3_128_20k/ (EM 57.0%)"
