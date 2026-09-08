#!/usr/bin/env bash
# Backward-compatible entrypoint: serve default solver on 4 GPUs.
# Now points to Qwen3-14B (vLLM batch + fixed seed). See serve_qwen3_14b_4gpu.sh.
set -euo pipefail
cd /home/ivaning/PAgent
exec bash scripts/serve_qwen3_14b_4gpu.sh "$@"
