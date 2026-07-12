#!/usr/bin/env python3
"""GRPO Action Router — unified entry point."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import datasets

from config import ANSWER_URLS, EVAL_WORKERS, PRESETS, Config
from core import build_small_prompt, extract_raw_question, write_jsonl
from eval import run_eval, run_precheck
from dpo_train import merge_dpo, train_dpo
from label import run_label
from oracle_hint import run_oracle_hint
from sft_train import merge_sft, train_sft
from sft_ff_train import merge_sft_ff, train_sft_ff
from train import merge, train
from train_ff import merge_ff, train_ff

DATA_SOURCE = "BytedTsinghua-SIA/DAPO-Math-17k"


def check_answerer(url: str, model: str) -> None:
    req_url = url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(req_url, timeout=10) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise SystemExit(f"Answerer unreachable at {req_url}: {exc}") from exc
    ids = [m["id"] for m in data["data"]]
    if model not in ids:
        raise SystemExit(f"{model} not served at {url}; available: {ids}")
    print(f"answerer OK ({url}): {ids}")


def check_answerers(urls: list[str], model: str) -> list[str]:
    alive: list[str] = []
    for url in urls:
        req_url = url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(req_url, timeout=10) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            print(f"answerer skip {url}: {exc}")
            continue
        ids = [m["id"] for m in data["data"]]
        if model not in ids:
            print(f"answerer skip {url}: {model} not in {ids}")
            continue
        alive.append(url)
        print(f"answerer OK ({url}): {ids}")
    if not alive:
        raise SystemExit(f"no answerer endpoints available for {model}")
    return alive


def prepare(cfg: Config, *, train_size=4096, val_size=256, seed=42) -> None:
    print(f"Loading {DATA_SOURCE} ...")
    full = datasets.load_dataset(DATA_SOURCE, split="train").shuffle(seed=seed)
    val_n = min(val_size, len(full))
    train_n = min(train_size, len(full) - val_n)

    def to_row(ex, idx):
        q = extract_raw_question(ex["prompt"][0]["content"])
        gold = ex["reward_model"]["ground_truth"]
        if hasattr(gold, "__len__") and not isinstance(gold, str):
            gold = gold[0]
        return {
            "id": idx, "data_source": DATA_SOURCE,
            "problem": q, "gold": str(gold),
            "router_prompt": build_small_prompt(q),
        }

    train_rows = [to_row(ex, i) for i, ex in enumerate(full.select(range(train_n)))]
    val_rows = [to_row(ex, train_n + i) for i, ex in enumerate(
        full.select(range(train_n, train_n + val_n))
    )]
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cfg.train_file, train_rows)
    write_jsonl(cfg.val_file, val_rows)
    print(f"train={len(train_rows)} -> {cfg.train_file}")
    print(f"val={len(val_rows)} -> {cfg.val_file}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Download DAPO subset -> jsonl")
    sp.add_argument("--train-size", type=int, default=4096)
    sp.add_argument("--val-size", type=int, default=256)

    sp = sub.add_parser("precheck", help="Per-action + oracle headroom check")
    sp.add_argument("--limit", type=int, default=16)

    sp = sub.add_parser("label", help="Exhaustive per-action rollout -> supervised labels")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--data-file", default=None, help="default: data/train.jsonl")
    sp.add_argument("--out-dir", default=None)
    sp.add_argument("--workers", type=int, default=32)
    sp.add_argument("--answer-urls", default=None,
                    help="comma-separated vLLM URLs (default: 4 endpoints from config)")
    sp.add_argument("--protocol", choices=["native", "paper"], default="native")

    sp = sub.add_parser("oracle-hint", help="Blind OSS hints (problem only) -> SFT labels if flip")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--data-file", default=None)
    sp.add_argument("--out-dir", default=None)
    sp.add_argument("--workers", type=int, default=32)
    sp.add_argument("--k", type=int, default=6, help="hint samples per baseline-wrong question")
    sp.add_argument("--hint-temp", type=float, default=0.8)
    sp.add_argument("--answer-urls", default=None)
    sp.add_argument("--protocol", choices=["native", "paper"], default="native")

    sp = sub.add_parser("dpo-build", help="Build offline DPO pairs from rollouts.jsonl")
    sp.add_argument("--rollouts-file", default=None)
    sp.add_argument("--out-file", default=None)

    sp = sub.add_parser("dpo-train", help="Standard offline DPO from preference pairs")
    sp.add_argument("--pairs-file", default=None)
    sp.add_argument("--rollouts-file", default=None)
    sp.add_argument("--out-dir", default=None)
    sp.add_argument("--gpu", default="3")
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--batch-size", type=int, default=2)
    sp.add_argument("--grad-accum", type=int, default=8)
    sp.add_argument("--lr", type=float, default=5e-7)
    sp.add_argument("--beta", type=float, default=0.1)
    sp.add_argument("--rebuild-pairs", action="store_true")

    sp = sub.add_parser("dpo-merge", help="Merge DPO LoRA adapter")

    sp = sub.add_parser("sft-train", help="Supervised router training from labels.jsonl")
    sp.add_argument("--labels-file", default=None)
    sp.add_argument("--out-dir", default=None, help="adapter output dir")
    sp.add_argument("--gpu", default="1")
    sp.add_argument("--epochs", type=int, default=3)
    sp.add_argument("--batch-size", type=int, default=8)
    sp.add_argument("--lr", type=float, default=2e-5)

    sp = sub.add_parser("sft-merge", help="Merge SFT LoRA adapter")

    sp = sub.add_parser("train", help="GRPO LoRA training")
    sp.add_argument("--mode", choices=["sanity", "quick", "full"], default="quick")
    sp.add_argument("--gpu", default="1")
    sp.add_argument("--rollout-workers", type=int, default=None,
                    help="parallel answerer requests per step (default: K)")

    sp = sub.add_parser("ff-train", help="Free-form prompt optimizer GRPO training")
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--max-steps", type=int, default=10)
    sp.add_argument("--k", type=int, default=8)
    sp.add_argument("--gpu", default="1")
    sp.add_argument("--rollout-workers", type=int, default=32)
    sp.add_argument("--gen-batch-size", type=int, default=4,
                    help="parallel Qwen hint prompts per generate batch")
    sp.add_argument("--start-step", type=int, default=1,
                    help="resume GRPO from this step (requires prior checkpoint in --out-dir)")
    sp.add_argument("--lr", type=float, default=1e-6)
    sp.add_argument("--data-file", default=None)
    sp.add_argument("--init-adapter", default=None, help="continue from existing LoRA adapter")
    sp.add_argument("--out-dir", default=None, help="output adapter dir")
    sp.add_argument("--rollout-log", default=None)
    sp.add_argument("--answer-urls", default=None,
                    help="comma-separated OSS URLs (default: all 4 endpoints)")

    sp = sub.add_parser("ff-merge", help="Merge free-form GRPO LoRA adapter")
    sp.add_argument("--adapter-dir", default=None)
    sp.add_argument("--merged-dir", default=None)

    sp = sub.add_parser("ff-sft-build", help="Build free-form SFT labels from action labels")
    sp.add_argument("--labels-file", default=None)
    sp.add_argument("--out-file", default=None)

    sp = sub.add_parser("ff-dedup-blind", help="Pick one flip hint per question via repeat OSS eval")
    sp.add_argument("--labels-file", default=None)
    sp.add_argument("--out-file", default=None)
    sp.add_argument("--repeats", type=int, default=3,
                    help="OSS re-tests per candidate hint (multi-candidate questions only)")
    sp.add_argument("--workers", type=int, default=32)
    sp.add_argument("--answer-urls", default=None)
    sp.add_argument("--protocol", choices=["native", "paper"], default="native")
    sp.add_argument("--retest-singles", action="store_true",
                    help="also re-test questions that only have one flip hint")

    sp = sub.add_parser("ff-sft-mix-blind", help="Mix blind flip hints + empty for baseline-correct")
    sp.add_argument("--baselines-file", default=None)
    sp.add_argument("--oracle-labels-file", default=None)
    sp.add_argument("--out-file", default=None)
    sp.add_argument("--empty-ratio", type=float, default=0.5,
                    help="target fraction of empty-hint rows (default: 0.5)")
    sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("ff-sft-train", help="SFT free-form prompt optimizer from ff labels")
    sp.add_argument("--labels-file", default=None)
    sp.add_argument("--out-dir", default=None)
    sp.add_argument("--gpu", default="1")
    sp.add_argument("--epochs", type=int, default=3)
    sp.add_argument("--batch-size", type=int, default=8)
    sp.add_argument("--lr", type=float, default=2e-5)

    sp = sub.add_parser("ff-sft-merge", help="Merge free-form SFT LoRA adapter")
    sp.add_argument("--adapter-dir", default=None)
    sp.add_argument("--merged-dir", default=None)

    sub.add_parser("merge", help="Merge LoRA adapter")

    sp = sub.add_parser("eval", help="Run eval modes")
    sp.add_argument("--modes", nargs="+",
                    default=["baseline", "random", "router", "oracle"])
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--router-model", default=None)
    sp.add_argument("--router-url", default=None)
    sp.add_argument("--protocol", choices=["native", "paper"], default="native",
                    help="native=Final Answer prompt; paper=Prompt-R1 <answer> protocol")
    sp.add_argument("--out-dir", default=None, help="eval output dir (default: checkpoints/eval or eval_paper)")
    sp.add_argument("--eval-workers", type=int, default=EVAL_WORKERS,
                    help="parallel HTTP workers for live_baseline/router eval")

    sp = sub.add_parser("pipeline", help="prepare -> precheck -> train -> merge -> eval")
    sp.add_argument("--mode", choices=["sanity", "quick", "full"], default="quick")
    sp.add_argument("--skip-precheck", action="store_true")
    sp.add_argument("--skip-eval", action="store_true")
    sp.add_argument("--gpu", default="1")
    sp.add_argument("--rollout-workers", type=int, default=None,
                    help="parallel answerer requests per step (default: K)")

    args = p.parse_args()
    cfg = Config()

    if args.cmd == "prepare":
        prepare(cfg, train_size=args.train_size, val_size=args.val_size)
    elif args.cmd == "precheck":
        check_answerer(cfg.answer_url, cfg.answer_model)
        run_precheck(cfg, limit=args.limit)
    elif args.cmd == "label":
        urls = [u.strip() for u in args.answer_urls.split(",")] if args.answer_urls else ANSWER_URLS
        urls = check_answerers(urls, cfg.answer_model)
        run_label(
            cfg,
            limit=args.limit,
            data_file=Path(args.data_file) if args.data_file else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            workers=args.workers,
            answer_urls=urls,
            protocol=args.protocol,
        )
    elif args.cmd == "oracle-hint":
        urls = [u.strip() for u in args.answer_urls.split(",")] if args.answer_urls else ANSWER_URLS
        urls = check_answerers(urls, cfg.answer_model)
        run_oracle_hint(
            cfg,
            limit=args.limit,
            data_file=Path(args.data_file) if args.data_file else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            workers=args.workers,
            answer_urls=urls,
            protocol=args.protocol,
            k=args.k,
            hint_temp=args.hint_temp,
        )
    elif args.cmd == "dpo-build":
        from dpo_data import build_dpo_pairs
        build_dpo_pairs(
            Path(args.rollouts_file or cfg.ckpt_dir / "label_2048" / "rollouts.jsonl"),
            Path(args.out_file or cfg.ckpt_dir / "label_2048" / "dpo_pairs.jsonl"),
        )
    elif args.cmd == "dpo-train":
        train_dpo(
            cfg,
            pairs_file=Path(args.pairs_file) if args.pairs_file else None,
            rollouts_file=Path(args.rollouts_file) if args.rollouts_file else None,
            adapter_dir=Path(args.out_dir) if args.out_dir else cfg.dpo_adapter_dir,
            gpu=args.gpu,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            beta=args.beta,
            rebuild_pairs=args.rebuild_pairs,
        )
    elif args.cmd == "dpo-merge":
        merge_dpo(cfg)
    elif args.cmd == "sft-train":
        train_sft(
            cfg,
            labels_file=Path(args.labels_file) if args.labels_file else None,
            adapter_dir=Path(args.out_dir) if args.out_dir else cfg.sft_adapter_dir,
            gpu=args.gpu,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.cmd == "sft-merge":
        merge_sft(cfg)
    elif args.cmd == "train":
        check_answerer(cfg.answer_url, cfg.answer_model)
        preset = PRESETS[args.mode]
        if not cfg.train_file.exists():
            prepare(cfg, train_size=preset["train_size"], val_size=preset["val_size"])
        train(cfg, limit=preset["limit"], max_steps=preset["max_steps"],
              k=preset.get("k", 8), gpu=args.gpu,
              grad_accum_groups=preset.get("grad_accum_groups", 4),
              rollout_workers=args.rollout_workers or preset.get("k", 8))
    elif args.cmd == "ff-train":
        urls = (
            [u.strip() for u in args.answer_urls.split(",")]
            if args.answer_urls else cfg.answer_urls
        )
        urls = check_answerers(urls, cfg.answer_model)
        cfg.answer_urls = urls
        if not cfg.train_file.exists():
            prepare(cfg, train_size=2048, val_size=256)
        train_ff(
            cfg,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            k=args.k,
            gpu=args.gpu,
            rollout_workers=args.rollout_workers,
            gen_batch_size=args.gen_batch_size,
            lr=args.lr,
            data_file=Path(args.data_file) if args.data_file else None,
            init_adapter_dir=Path(args.init_adapter) if args.init_adapter else None,
            adapter_dir=Path(args.out_dir) if args.out_dir else None,
            rollout_log=Path(args.rollout_log) if args.rollout_log else None,
            start_step=args.start_step,
        )
    elif args.cmd == "ff-merge":
        merge_ff(
            cfg,
            adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
            merged_dir=Path(args.merged_dir) if args.merged_dir else None,
        )
    elif args.cmd == "ff-sft-build":
        from ff_data import build_ff_labels
        build_ff_labels(
            Path(args.labels_file or cfg.ckpt_dir / "label_2048" / "labels.jsonl"),
            Path(args.out_file or cfg.ckpt_dir / "label_2048" / "ff_labels.jsonl"),
        )
    elif args.cmd == "ff-dedup-blind":
        from dedup_blind_labels import dedup_blind_labels
        urls = [u.strip() for u in args.answer_urls.split(",")] if args.answer_urls else ANSWER_URLS
        urls = check_answerers(urls, cfg.answer_model)
        blind_dir = cfg.ckpt_dir / "blind_hint_17k"
        dedup_blind_labels(
            cfg,
            labels_file=Path(args.labels_file or blind_dir / "oracle_labels.jsonl"),
            out_file=Path(args.out_file or blind_dir / "oracle_labels_dedup.jsonl"),
            repeats=args.repeats,
            workers=args.workers,
            answer_urls=urls,
            protocol=args.protocol,
            retest_singles=args.retest_singles,
        )
    elif args.cmd == "ff-sft-mix-blind":
        from ff_data import build_blind_mixed_labels
        blind_dir = cfg.ckpt_dir / "blind_hint_2048"
        build_blind_mixed_labels(
            Path(args.baselines_file or blind_dir / "baselines.jsonl"),
            Path(args.oracle_labels_file or blind_dir / "oracle_labels.jsonl"),
            Path(args.out_file or blind_dir / "mixed_labels.jsonl"),
            empty_ratio=args.empty_ratio,
            seed=args.seed,
        )
    elif args.cmd == "ff-sft-train":
        train_sft_ff(
            cfg,
            labels_file=Path(args.labels_file) if args.labels_file else None,
            adapter_dir=Path(args.out_dir) if args.out_dir else None,
            gpu=args.gpu,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.cmd == "ff-sft-merge":
        merge_sft_ff(
            cfg,
            adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
            merged_dir=Path(args.merged_dir) if args.merged_dir else None,
        )
    elif args.cmd == "merge":
        merge(cfg)
    elif args.cmd == "eval":
        if args.router_url:
            cfg.router_url = args.router_url
        run_eval(cfg, modes=args.modes, limit=args.limit,
                 router_model=args.router_model, protocol=args.protocol,
                 out_dir=Path(args.out_dir) if args.out_dir else None,
                 workers=args.eval_workers)
    elif args.cmd == "pipeline":
        check_answerer(cfg.answer_url, cfg.answer_model)
        preset = PRESETS[args.mode]
        print(f"mode={args.mode}: train={preset['train_size']} steps={preset.get('max_steps') or 'all'} k={preset.get('k', 8)}")
        prepare(cfg, train_size=preset["train_size"], val_size=preset["val_size"])
        if not args.skip_precheck:
            run_precheck(cfg, limit=preset.get("precheck_limit", 8))
        train(cfg, limit=preset["limit"], max_steps=preset["max_steps"],
              k=preset.get("k", 8), gpu=args.gpu,
              grad_accum_groups=preset.get("grad_accum_groups", 4),
              rollout_workers=args.rollout_workers or preset.get("k", 8))
        merge(cfg)
        print(
            f"\nServe trained router:\n"
            f"  CUDA_VISIBLE_DEVICES=2 vllm serve {cfg.merged_dir} "
            f"--served-model-name qwen3-4b-router --port 8083 "
            f"--tensor-parallel-size 1 --gpu-memory-utilization 0.50"
        )
        if not args.skip_eval:
            print("\n[eval] baseline/random need answerer; router needs merged model served on :8083")
            run_eval(cfg, modes=["baseline", "random"], limit=64)


if __name__ == "__main__":
    main()
