#!/usr/bin/env bash
# Prepare 2048 train rows and run 4-endpoint labeling.
set -euo pipefail
cd /home/ivaning/PAgent

echo "[1/3] prepare train=2048 val=256"
python run.py prepare --train-size 2048 --val-size 256

echo "[2/3] check answerers"
python - <<'PY'
import urllib.request, json, sys
from config import ANSWER_URLS, ANSWER_MODEL
ok = []
for url in ANSWER_URLS:
    req = url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        if ANSWER_MODEL not in ids:
            print(f"FAIL {req}: missing {ANSWER_MODEL}, got {ids}")
            sys.exit(1)
        ok.append(url)
        print(f"OK {url}")
    except Exception as e:
        print(f"DOWN {url}: {e}")
if len(ok) < 1:
    sys.exit("No answerer up. Run scripts/serve_oss_4gpu.sh first.")
if len(ok) < 4:
    print(f"WARNING: only {len(ok)}/4 endpoints up; continuing anyway")
PY

echo "[3/3] label 2048 questions (6 actions each, workers=32)"
python run.py label \
  --limit 2048 \
  --workers 32 \
  --out-dir checkpoints/label_2048 \
  2>&1 | tee logs/label-2048.log

echo "Done. See checkpoints/label_2048/stats.json"
