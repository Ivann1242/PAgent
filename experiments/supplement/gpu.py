"""Physical-device discipline. This module has no ML dependencies."""
from __future__ import annotations

import os
import subprocess
import sys

ALLOWED = (1, 2)


def inventory() -> dict[int, str]:
    if sys.platform != "linux":
        raise RuntimeError("Server execution only; model workloads are forbidden on this host")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    found = {}
    for line in result.stdout.splitlines():
        index, uuid = (x.strip() for x in line.split(","))
        found[int(index)] = uuid
    if not all(i in found for i in ALLOWED):
        raise RuntimeError("Physical GPUs 1 and 2 must both exist; no fallback allowed")
    return {i: found[i] for i in ALLOWED}


def require_idle(uuids: dict[int, str]) -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    busy = [line for line in result.stdout.splitlines()
            if line.split(",")[0].strip() in uuids.values()]
    if busy:
        raise RuntimeError("GPU 1/2 already occupied; no processes will be killed: " + "; ".join(busy))


def bind(physical: int, uuids: dict[int, str]) -> dict[str, str]:
    if physical not in ALLOWED:
        raise ValueError("Only physical GPU 1 or 2 is allowed")
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = uuids[physical]
    env["PAGENT_PHYSICAL_GPU"] = str(physical)
    env["PAGENT_GPU_UUID"] = uuids[physical]
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def verify_worker() -> str:
    physical = int(os.environ.get("PAGENT_PHYSICAL_GPU", "-1"))
    if physical not in ALLOWED:
        raise RuntimeError("Use the supplementary launcher: physical GPU must be 1 or 2")
    uuid = inventory()[physical]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or os.environ.get("PAGENT_GPU_UUID") != uuid:
        raise RuntimeError("GPU binding mismatch; refusing to initialize CUDA")
    return uuid
