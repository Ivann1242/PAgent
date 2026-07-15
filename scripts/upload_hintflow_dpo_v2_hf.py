#!/usr/bin/env python3
"""Upload HintFlow DPO v2 checkpoints to Hugging Face (public archive)."""

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
        ROOT / "checkpoints/hintflow_dpo_v2_merged",
        f"{HF_USER}/pagent-hintflow-dpo-v2-merged",
        "backup hintflow dpo v2 merged",
    ),
    (
        ROOT / "checkpoints/hintflow_dpo_v2_adapter",
        f"{HF_USER}/pagent-hintflow-dpo-v2-adapter",
        "backup hintflow dpo v2 adapter",
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
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in {"HF_WRITE_TOKEN", "HF_TOKEN"}:
                found[k] = v
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
        api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)
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
