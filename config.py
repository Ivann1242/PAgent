"""Paths and hyperparameters for GRPO Action Router."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

ROUTER_BASE = Path("/home/ivaning/prompt-r1r/Prompt-R1/Qwen/Qwen3-4B")
_EVAL_CANDIDATES = (
    ROOT / "data" / "DAPO-Math.parquet",
    Path("/home/ivaning/prompt-r1r/Prompt-R1/dataset/eval_data/DAPO-Math.parquet"),
)
EVAL_PARQUET = next((p for p in _EVAL_CANDIDATES if p.exists()), _EVAL_CANDIDATES[0])
BASELINE_RES = Path(
    "/home/ivaning/prompt-r1r/Prompt-R1/baseline-results/gpt-oss-20b/DAPO-Math/res.json"
)

ANSWER_URL = "http://127.0.0.1:8006/v1"
ANSWER_URLS = [
    "http://127.0.0.1:8006/v1",
    "http://127.0.0.1:8007/v1",
    "http://127.0.0.1:8008/v1",
    "http://127.0.0.1:8009/v1",
]
ANSWER_MODEL = "gpt-oss-20b"
OSS_MODEL_PATH = Path("/home/ivaning/models/gpt-oss-20b")
ROUTER_URL = "http://127.0.0.1:8083/v1"
ROUTER_MODEL = "qwen3-4b"
TRAINED_ROUTER_MODEL = "qwen3-4b-router"
TRAINED_FF_MODEL = "qwen3-4b-ff"

# README defaults
K = 8
BATCH_SIZE = 32
LR = 1e-6
CLIP_RANGE = 0.2
KL_BETA = 0.02
SMALL_TEMP_TRAIN = 1.0
SMALL_TEMP_EVAL = 0.0
LARGE_TEMP = 0.0
GRAD_ACCUM_GROUPS = 4
MIN_UNIQUE_ACTIONS = 2
ROLLOUT_WORKERS = 8
EVAL_WORKERS = 8

SANITY = dict(train_size=32, val_size=8, limit=32, max_steps=10, k=4, precheck_limit=8, grad_accum_groups=2)
QUICK = dict(train_size=128, val_size=32, limit=128, max_steps=128, k=K, precheck_limit=8, grad_accum_groups=GRAD_ACCUM_GROUPS)
FULL = dict(train_size=4096, val_size=256, limit=None, max_steps=None, k=K, precheck_limit=16, grad_accum_groups=GRAD_ACCUM_GROUPS)

PRESETS = {"sanity": SANITY, "quick": QUICK, "full": FULL}


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    ckpt_dir: Path = field(default_factory=lambda: CKPT_DIR)
    train_file: Path = field(default_factory=lambda: DATA_DIR / "train.jsonl")
    val_file: Path = field(default_factory=lambda: DATA_DIR / "val.jsonl")
    adapter_dir: Path = field(default_factory=lambda: CKPT_DIR / "adapter")
    merged_dir: Path = field(default_factory=lambda: CKPT_DIR / "merged")
    sft_adapter_dir: Path = field(default_factory=lambda: CKPT_DIR / "sft_adapter")
    sft_merged_dir: Path = field(default_factory=lambda: CKPT_DIR / "sft_merged")
    dpo_adapter_dir: Path = field(default_factory=lambda: CKPT_DIR / "dpo_adapter")
    dpo_merged_dir: Path = field(default_factory=lambda: CKPT_DIR / "dpo_merged")
    ff_adapter_dir: Path = field(default_factory=lambda: CKPT_DIR / "ff_adapter")
    ff_merged_dir: Path = field(default_factory=lambda: CKPT_DIR / "ff_merged")
    ff_sft_adapter_dir: Path = field(default_factory=lambda: CKPT_DIR / "ff_sft_adapter")
    ff_sft_merged_dir: Path = field(default_factory=lambda: CKPT_DIR / "ff_sft_merged")
    ff_rollout_log: Path = field(default_factory=lambda: CKPT_DIR / "rollouts_ff.jsonl")
    rollout_log: Path = field(default_factory=lambda: CKPT_DIR / "rollouts.jsonl")
    router_base: Path = field(default_factory=lambda: ROUTER_BASE)
    answer_url: str = ANSWER_URL
    answer_urls: list[str] = field(default_factory=lambda: list(ANSWER_URLS))
    answer_model: str = ANSWER_MODEL
    router_url: str = ROUTER_URL
    router_model: str = ROUTER_MODEL
