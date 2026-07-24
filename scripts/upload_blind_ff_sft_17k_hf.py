#!/usr/bin/env python3
"""Upload current HF1 Blind FF-SFT 17k (full) to Hugging Face."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path("/home/ivaning/PAgent")
ENV_FILE = Path("/home/ivaning/prompt-r1r/Prompt-R1/.env")
HF_USER = "ivaning0919"

JOBS = [
    (
        ROOT / "checkpoints/blind_ff_sft_17k_merged",
        f"{HF_USER}/pagent-blind-ff-sft-17k-merged",
        "archive Blind FF-SFT 17k full merged (current HF1 router)",
    ),
    (
        ROOT / "checkpoints/blind_ff_sft_17k_adapter",
        f"{HF_USER}/pagent-blind-ff-sft-17k-adapter",
        "archive Blind FF-SFT 17k full LoRA adapter",
    ),
]


def load_token() -> str:
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    found: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            found[k.strip()] = v.strip().strip('"').strip("'")
    token = found.get("HF_WRITE_TOKEN") or found.get("HF_TOKEN")
    if not token:
        raise SystemExit("No HF token found")
    return token


def main() -> None:
    token = load_token()
    api = HfApi(token=token)
    user = api.whoami(token=token)["name"]
    if user != HF_USER:
        raise SystemExit(f"expected {HF_USER}, got {user}")
    print(f"logged in as: {user}", flush=True)

    for local, repo_id, msg in JOBS:
        if not local.exists():
            print(f"[skip] missing {local}", flush=True)
            continue
        print(f"\n=== {local.name} -> {repo_id} ===", flush=True)
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=token)
        api.upload_folder(
            folder_path=str(local),
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
            token=token,
        )
        print(f"[ok] https://huggingface.co/{repo_id}", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
