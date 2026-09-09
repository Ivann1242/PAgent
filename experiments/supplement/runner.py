"""Own all GPU subprocesses; never connect to pre-existing model services."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from .gpu import bind, inventory, require_idle
from .io import dataset, digest, file_hash, model_hash, write_json

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    required = {"policy_base", "solver_path", "sft_adapter", "train_data", "eval_data", "output"}
    if not required <= cfg.keys():
        raise ValueError(f"Missing config fields: {sorted(required - cfg.keys())}")
    cfg = {
        "seeds": [41, 42, 43], "eval_seeds": [101, 102, 103],
        "max_steps": 200, "batch_size": 32, "k": 8, "gen_batch_size": 1,
        "reward_repeats": 1, "reward_max_tokens": 8192,
        "checkpoint_every": 25, "lr": 1e-6, "reward_temperature": 0.0,
        "challenger_temperature": 0.7, "solver_temperature": 0.0,
        "solver_max_tokens": 8192, "budgets": [8192, 16384, 32768],
        "solver_port": 18202, "policy_port": 18201,
        "max_model_len": 32768, "max_num_seqs": 8,
        "gpu_memory_utilization": 0.85, "rollout_workers": 4,
        "startup_timeout": 900, "request_timeout": 1800,
        "replace_threshold": 0.90, "max_steps_hfm": 8,
        **cfg,
    }
    for forbidden in ("gpu", "gpus", "policy_gpu", "solver_gpu", "answer_urls", "solver_url", "orch_url"):
        if forbidden in cfg:
            raise ValueError(f"{forbidden} is not configurable: GPU1 policy / GPU2 solver only")
    for key in ("policy_base", "solver_path", "sft_adapter", "train_data", "eval_data", "output"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    for key in ("seeds", "eval_seeds", "budgets"):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"{key} must be nonempty and unique")
    if cfg["solver_port"] == cfg["policy_port"]:
        raise ValueError("Policy and solver ports must differ")
    for key in ("max_steps", "batch_size", "k", "gen_batch_size", "reward_repeats", "checkpoint_every"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    return cfg


def latest_checkpoint(out: Path) -> Path | None:
    candidates = sorted((out / "snapshots").glob("step-*"))
    for path in reversed(candidates):
        if ((path / "COMPLETE").exists() and (path / "optimizer_state.pt").exists()
                and (path / "adapter_config.json").exists()):
            state = json.loads((path / "grpo_train_state.json").read_text())
            if int((path / "COMPLETE").read_text()) == state["step"]:
                return path
    return None


@contextlib.contextmanager
def exclusive_lock():
    # Cross-checkout lock shared by this Unix account. A busy GPU check catches
    # other users/processes, but is not a replacement for a cluster allocation.
    path = Path(f"/tmp/pagent-gpu12-{os.getuid()}.lock")
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Processes:
    def __init__(self, uuids: dict, output: Path):
        self.uuids = uuids
        self.output = output
        self.children = []

    def start(self, args: list[str], gpu: int, name: str):
        path = self.output / "logs" / f"{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a")
        try:
            proc = subprocess.Popen(args, cwd=ROOT, env=bind(gpu, self.uuids),
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            stream.close()
        self.children.append(proc)
        print(f"Started {name}: pid={proc.pid}, physical GPU={gpu}, log={path}", flush=True)
        return proc

    def stop(self, proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def close(self):
        for proc in reversed(self.children):
            self.stop(proc)

    def serve(self, cfg: dict, *, policy=False, adapters=None):
        port = cfg["policy_port" if policy else "solver_port"]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))  # Fail before spawning if occupied.
        args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", cfg["policy_base" if policy else "solver_path"],
                "--served-model-name", "policy-base" if policy else "qwen3-14b",
                "--host", "127.0.0.1", "--port", str(port),
                "--tensor-parallel-size", "1", "--seed", "0",
                "--max-model-len", str(cfg["max_model_len"]),
                "--max-num-seqs", str(cfg["max_num_seqs"]),
                "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"])]
        if policy:
            args += ["--enable-lora", "--max-lora-rank", "64", "--max-loras", "2",
                     "--lora-modules", *[f"{name}={path}" for name, path in adapters.items()]]
        proc = self.start(args, 1 if policy else 2, "policy-server" if policy else "solver-server")
        deadline = time.monotonic() + cfg["startup_timeout"]
        expected = set(adapters) if policy else {"qwen3-14b"}
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("Owned model server exited; inspect its log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                    names = {item["id"] for item in json.load(response)["data"]}
                if expected <= names:
                    return proc
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError("Owned model server startup timed out")

    def worker(self, config_path: Path, phase: str, condition: str, seed: int):
        proc = self.start([sys.executable, "-m", "experiments.supplement.worker",
                           "--config", str(config_path), "--phase", phase,
                           "--condition", condition, "--seed", str(seed)],
                          1, f"{phase}-{condition}-{seed}")
        if proc.wait() != 0:
            raise RuntimeError(f"Worker failed: {phase}/{condition}/{seed}; inspect logs")


def run(cfg: dict, phase: str, execute: bool):
    if not execute:
        print(json.dumps({"phase": phase, "physical_gpus": {"policy": 1, "solver": 2},
                          "config": cfg, "execute": False}, indent=2))
        return
    if phase == "all":
        for stage in ("train", "eval", "hfm"):
            run(cfg, stage, True)
        from .analyze import analyze
        analyze(cfg["output"], plots=True)
        return
    uuids = inventory()
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock():
        require_idle(uuids)
        from .prepare import normalize
        train_rows = dataset(cfg["train_data"])
        eval_rows = dataset(cfg["eval_data"])
        train_problems = {normalize(r["problem"]) for r in train_rows}
        if any(normalize(r["problem"]) in train_problems for r in eval_rows):
            raise RuntimeError("Training/evaluation exact problem overlap; refusing experiment")
        # Content hashes, including weights, make cached candidates and resumes
        # invalid when a model directory changes in place. Hashing is CPU-only.
        manifest = {"config": cfg, "data": {k: file_hash(cfg[k]) for k in ("train_data", "eval_data")},
                    "models": {k: model_hash(cfg[k]) for k in ("policy_base", "solver_path", "sft_adapter")},
                    "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Protocol/source/model/data changed: use a new output directory")
        if subprocess.check_output(["git", "diff", "HEAD", "--", "*.py"], cwd=ROOT):
            raise RuntimeError("Commit Python changes before running a reproducible experiment")
        write_json(manifest_path, manifest)
        config_path = output / "config.json"
        write_json(config_path, cfg)
        write_json(output / "hardware.json", {"physical_gpu_uuids": uuids})
        processes = Processes(uuids, output)
        old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        status = {"phase": phase, "state": "running", "manifest_hash": digest(manifest)}
        write_json(output / "status.json", status)
        try:
            processes.serve(cfg)
            if phase == "train":
                for seed in cfg["seeds"]:
                    for condition in ("off", "vp"):
                        status.update(condition=condition, seed=seed)
                        write_json(output / "status.json", status)
                        processes.worker(config_path, phase, condition, seed)
            elif phase in {"eval", "hfm"}:
                pairs = [("sft", cfg["seeds"][0])]
                pairs += [(c, s) for s in cfg["seeds"] for c in ("off", "vp")]
                if phase == "hfm":
                    pairs = [("vp", cfg["seeds"][0])]
                for condition, seed in pairs:
                    adapter = Path(cfg["sft_adapter"]) if condition == "sft" else latest_checkpoint(output / "train" / condition / str(seed))
                    if adapter is None:
                        raise RuntimeError(f"Missing complete checkpoint for {condition}/{seed}")
                    if condition != "sft" and json.loads((adapter / "grpo_train_state.json").read_text())["step"] != cfg["max_steps"]:
                        raise RuntimeError("Final evaluation requires the preselected final step")
                    adapter_record = output / "adapter_hashes" / f"{condition}-{seed}.json"
                    identity = {"path": str(adapter), "hash": model_hash(adapter)}
                    if adapter_record.exists() and json.loads(adapter_record.read_text()) != identity:
                        raise RuntimeError("Evaluation adapter changed; refusing cached result reuse")
                    write_json(adapter_record, identity)
                    policy = processes.serve(cfg, policy=True, adapters={"sft": cfg["sft_adapter"], "current": str(adapter)})
                    status.update(condition=condition, seed=seed)
                    write_json(output / "status.json", status)
                    processes.worker(config_path, phase, condition, seed)
                    processes.stop(policy)
            status["state"] = "complete"
        except BaseException as exc:
            status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise
        finally:
            processes.close()
            signal.signal(signal.SIGTERM, old_handler)
            write_json(output / "status.json", status)
