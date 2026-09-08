#!/usr/bin/env bash
# Virtual-positive GRPO from the current Blind FF-SFT 17k HF1 challenger.
# GPU0=policy + shared OSS, GPU1/2=OSS reward; GPU3 is untouched.
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=pagent-blind-ff-grpo-vp320
LOG=logs/blind-ff-grpo-17k-vp320-train.log
SFT_INIT=checkpoints/blind_ff_sft_17k_adapter
REF=checkpoints/blind_ff_sft_17k_adapter
OUT=checkpoints/blind_ff_grpo_17k_virtual_positive_adapter
MERGED=checkpoints/blind_ff_grpo_17k_virtual_positive_merged
ROLLOUT=checkpoints/blind_ff_grpo_17k_virtual_positive_rollouts.jsonl
HF_REPO=ivaning0919/pagent-blind-ff-grpo-17k-virtual-positive

mkdir -p logs "$OUT"

STATE="$OUT/grpo_train_state.json"
if [[ -f "$STATE" ]]; then
  SAVED_STEP=$(python -c "import json; print(json.load(open('$STATE'))['step'])")
  START_STEP=$((SAVED_STEP + 1))
  INIT="$OUT"
else
  SAVED_STEP=0
  START_STEP=1
  INIT="$SFT_INIT"
fi

wait_model() {
  local url=$1 name=$2
  for _ in $(seq 1 120); do
    if curl -sf "$url/models" | python -c \
      "import json,sys; assert any(x['id']=='$name' for x in json.load(sys.stdin)['data'])" \
      >/dev/null 2>&1; then
      echo "  $url $name OK"
      return 0
    fi
    sleep 3
  done
  echo "FAIL $url $name"
  return 1
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

wait_model http://127.0.0.1:8006/v1 gpt-oss-20b
wait_model http://127.0.0.1:8007/v1 gpt-oss-20b
wait_model http://127.0.0.1:8008/v1 gpt-oss-20b

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo \"=== \$(date -Is) virtual-positive GRPO 320 ===\" | tee -a '$LOG'
  echo 'layout: GPU0=policy+:8008 GPU1=:8006 GPU2=:8007; GPU3 untouched' | tee -a '$LOG'
  echo 'batch=64 K=8 repeats=3 max_tokens=8192 steps=320' | tee -a '$LOG'
  echo 'resume: saved_step=$SAVED_STEP start_step=$START_STEP init=$INIT' | tee -a '$LOG'
  df -h /home/ivaning | tee -a '$LOG'

  python run.py ff-train \
    --init-adapter '$INIT' \
    --reference-adapter '$REF' \
    --out-dir '$OUT' \
    --rollout-log '$ROLLOUT' \
    --answer-urls 'http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1' \
    --data-file data/train.jsonl \
    --batch-size 64 \
    --max-steps 320 \
    --k 8 \
    --gen-batch-size 8 \
    --gpu 0 \
    --rollout-workers 96 \
    --reward-repeats 3 \
    --reward-max-tokens 8192 \
    --reward-temperature 0.0 \
    --lr 1e-6 \
    --start-step '$START_STEP' \
    --checkpoint-every 10 \
    --hf-repo '$HF_REPO' \
    2>&1 | tee -a '$LOG'

  echo \"=== \$(date -Is) final merge ===\" | tee -a '$LOG'
  python run.py ff-merge \
    --adapter-dir '$OUT' \
    --merged-dir '$MERGED' \
    2>&1 | tee -a '$LOG'

  echo \"=== \$(date -Is) upload final adapter + merged ===\" | tee -a '$LOG'
  python - <<'PY' 2>&1 | tee -a '$LOG'
from pathlib import Path
from huggingface_hub import HfApi

env_file = Path('/home/ivaning/prompt-r1r/Prompt-R1/.env')
values = {}
for line in env_file.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    key, value = line.split('=', 1)
    values[key.strip()] = value.strip().strip('\"').strip(\"'\")
token = values.get('HF_WRITE_TOKEN') or values.get('HF_TOKEN')
if not token:
    raise SystemExit('no HF token')

api = HfApi(token=token)
repo = 'ivaning0919/pagent-blind-ff-grpo-17k-virtual-positive'
for local, remote in [
    ('checkpoints/blind_ff_grpo_17k_virtual_positive_adapter', 'final-adapter'),
    ('checkpoints/blind_ff_grpo_17k_virtual_positive_merged', 'final-merged'),
]:
    api.upload_folder(
        folder_path=local,
        repo_id=repo,
        repo_type='model',
        path_in_repo=remote,
        ignore_patterns=['README.md'],
        commit_message=f'upload {remote}',
        token=token,
    )
    print(f'uploaded {local} -> {repo}/{remote}', flush=True)
PY

  # The final merged weights are archived on HF; retain only the latest local
  # adapter/resume state as requested.
  rm -f '$MERGED/model.safetensors'
  echo \"=== \$(date -Is) DONE ===\" | tee -a '$LOG'
  echo 'HF: https://huggingface.co/$HF_REPO' | tee -a '$LOG'
"

echo "started tmux: $SESSION"
echo "log: tail -f $LOG"
echo "HF checkpoints: https://huggingface.co/$HF_REPO"
