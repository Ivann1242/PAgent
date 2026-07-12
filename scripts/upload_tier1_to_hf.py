#!/usr/bin/env python3
"""Upload Tier-1 obsolete PAgent merged routers to Hugging Face (public)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path("/home/ivaning/PAgent")
CHECKPOINTS = ROOT / "checkpoints"
ENV_CANDIDATES = [
    ROOT / ".env",
    Path("/home/ivaning/prompt-r1r/Prompt-R1/.env"),
]
HF_USER = os.environ.get("HF_USER", "ivaning0919")

UPLOADS: list[tuple[str, str, str]] = [
    (
        "merged",
        f"{HF_USER}/pagent-router-grpo-merged",
        "Early DAPO-Math action-router GRPO merged Qwen3-4B.",
    ),
    (
        "sft_merged",
        f"{HF_USER}/pagent-router-sft-merged",
        "Early DAPO-Math action-router SFT merged Qwen3-4B.",
    ),
    (
        "dpo_merged",
        f"{HF_USER}/pagent-router-dpo-merged",
        "Early DAPO-Math action-router DPO merged Qwen3-4B.",
    ),
    (
        "ff_sft_merged",
        f"{HF_USER}/pagent-ff-sft-merged",
        "Free-form prompt optimizer SFT merged Qwen3-4B (template hints).",
    ),
    (
        "blind_ff_sft_merged",
        f"{HF_USER}/pagent-blind-ff-sft-merged",
        "Blind-hint free-form SFT merged Qwen3-4B (2048-label run).",
    ),
    (
        "blind_ff_sft_v2_merged",
        f"{HF_USER}/pagent-blind-ff-sft-v2-merged",
        "Blind-hint free-form SFT v2 merged Qwen3-4B.",
    ),
    (
        "blind_ff_sft_v3_merged",
        f"{HF_USER}/pagent-blind-ff-sft-v3-merged",
        "Blind-hint free-form SFT v3 merged Qwen3-4B.",
    ),
    (
        "blind_ff_grpo_merged",
        f"{HF_USER}/pagent-blind-ff-grpo-merged",
        "Blind-hint free-form GRPO merged Qwen3-4B.",
    ),
    (
        "blind_ff_sft_17k_dedup_merged",
        f"{HF_USER}/pagent-blind-ff-sft-17k-dedup-merged",
        "Blind-hint 17K dedup-label free-form SFT merged Qwen3-4B.",
    ),
]


def load_token() -> str:
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    for env_file in ENV_CANDIDATES:
        if not env_file.exists():
            continue
        env: dict[str, str] = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
        token = env.get("HF_WRITE_TOKEN") or env.get("HF_TOKEN")
        if token:
            return token
    raise SystemExit("No HF token found. Set HF_WRITE_TOKEN or add it to .env")


def local_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def remote_size(api: HfApi, repo_id: str) -> int:
    total = 0
    for info in api.list_repo_tree(repo_id, repo_type="model", recursive=True):
        if getattr(info, "size", None):
            total += info.size
    return total


def ensure_readme(local: Path, description: str) -> None:
    readme = local / "README.md"
    if readme.exists():
        return
    readme.write_text(
        f"---\n"
        f"license: apache-2.0\n"
        f"base_model: Qwen/Qwen3-4B\n"
        f"tags:\n"
        f"- prompt-optimization\n"
        f"- lora-merged\n"
        f"---\n\n"
        f"# {local.name}\n\n"
        f"{description}\n\n"
        f"Merged full-weight Qwen3-4B router checkpoint from the PAgent project.\n",
        encoding="utf-8",
    )


def upload_one(api: HfApi, local_name: str, repo_id: str, description: str) -> None:
    local = CHECKPOINTS / local_name
    if not local.exists():
        print(f"[skip] missing: {local}")
        return

    ensure_readme(local, description)
    loc_bytes = local_size(local)
    print(f"\n=== upload {local_name} -> {repo_id} ({loc_bytes / 1e9:.2f} GB) ===")

    create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=api.token)
    api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)

    rem_bytes = remote_size(api, repo_id)
    if loc_bytes > 0 and rem_bytes >= loc_bytes * 0.95:
        print(f"[skip upload] remote already has ~{rem_bytes / 1e9:.2f} GB")
        return

    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"backup {local_name}",
        token=api.token,
    )
    rem_bytes = remote_size(api, repo_id)
    if rem_bytes < loc_bytes * 0.95:
        raise RuntimeError(
            f"verify failed for {repo_id}: local={loc_bytes} remote={rem_bytes}"
        )
    print(f"[ok upload] https://huggingface.co/{repo_id} (~{rem_bytes / 1e9:.2f} GB)")


def main() -> None:
    token = load_token()
    api = HfApi(token=token)
    user = api.whoami(token=token)["name"]
    if HF_USER != user:
        raise SystemExit(f"HF_USER={HF_USER} does not match logged-in user {user}")

    print(f"logged in as: {user}")
    print(f"uploading {len(UPLOADS)} tier-1 merged models (public)")

    for local_name, repo_id, description in UPLOADS:
        upload_one(api, local_name, repo_id, description)

    print("\nAll tier-1 uploads done. Local checkpoints were NOT deleted.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
