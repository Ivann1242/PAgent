"""Internal worker. GPU attestation happens before importing any ML library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from .gpu import verify_worker
from .io import write_json


def train(cfg, condition, seed, gpu_uuid):
    from openai import OpenAI
    from config import Config
    import core
    from train_ff import train_ff
    from .runner import latest_checkpoint
    from .trace import TracedClient

    out = Path(cfg["output"]) / "train" / condition / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint(out)
    step = json.loads((checkpoint / "grpo_train_state.json").read_text())["step"] if checkpoint else 0
    if step >= cfg["max_steps"]:
        print(f"Already complete: {condition}/{seed} step={step}", flush=True)
        return
    attempt = uuid.uuid4().hex
    trace_path = out / "attempts" / attempt / "calls.jsonl"
    write_json(trace_path.parent / "manifest.json", {"attempt": attempt, "resume_step": step,
               "condition": condition, "seed": seed, "checkpoint": str(checkpoint) if checkpoint else None})

    def client_factory(url, **kwargs):
        return TracedClient(OpenAI(base_url=url, api_key="EMPTY", max_retries=0,
                                  timeout=cfg["request_timeout"]), trace_path, role="solver")

    core.make_openai_client = client_factory
    configuration = Config(router_base=Path(cfg["policy_base"]), answer_model="qwen3-14b")
    train_ff(configuration, batch_size=cfg["batch_size"], max_steps=cfg["max_steps"],
             k=cfg["k"], lr=cfg["lr"], gpu=gpu_uuid, seed=seed,
             rollout_workers=cfg["rollout_workers"], gen_batch_size=cfg["gen_batch_size"],
             answer_urls=[f"http://127.0.0.1:{cfg['solver_port']}/v1"],
             data_file=Path(cfg["train_data"]),
             init_adapter_dir=checkpoint or Path(cfg["sft_adapter"]), adapter_dir=out,
             rollout_log=trace_path.parent / "rollouts.jsonl", start_step=step + 1,
             reward_repeats=cfg["reward_repeats"], reward_max_tokens=cfg["reward_max_tokens"],
             reward_temperature=cfg["reward_temperature"],
             reference_adapter_dir=Path(cfg["sft_adapter"]),
             checkpoint_every=cfg["checkpoint_every"], all_wrong_mode=condition,
             resume_checkpoint=checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["train", "eval", "hfm"], required=True)
    parser.add_argument("--condition", choices=["sft", "vp", "off"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    gpu_uuid = verify_worker()
    cfg = json.loads(Path(args.config).read_text())
    # core.call_llm has its own retry loop, separate from OpenAI max_retries.
    import core
    core.LLM_MAX_RETRIES = 1
    if args.phase == "train":
        train(cfg, args.condition, args.seed, gpu_uuid)
    else:
        from .evaluate import evaluate_hf1, evaluate_hfm
        (evaluate_hf1 if args.phase == "eval" else evaluate_hfm)(cfg, args.condition, args.seed)


if __name__ == "__main__":
    main()
