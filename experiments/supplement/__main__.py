from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="HintFlow supplements: physical GPU 1/2 ONLY")
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run", help="Dry-run by default; execute only on the Linux GPU server")
    launch.add_argument("phase", choices=["all", "train", "eval", "hfm"])
    launch.add_argument("--config", required=True)
    launch.add_argument("--execute", action="store_true")
    analysis = commands.add_parser("analyze", help="CPU analysis of server-generated results")
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--plots", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--output", required=True)
    prepare = commands.add_parser("freeze-data")
    prepare.add_argument("--candidates", required=True)
    prepare.add_argument("--exclude", action="append", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--count", type=int, default=1024)
    prepare.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if args.command == "run":
        from .runner import load_config, run
        run(load_config(args.config), args.phase, args.execute)
    elif args.command == "analyze":
        from .analyze import analyze
        analyze(args.output, args.plots)
    elif args.command == "freeze-data":
        from .prepare import freeze
        freeze(args.candidates, args.exclude, args.output, args.count, args.seed)
    else:
        from .runner import latest_checkpoint
        root = Path(args.output)
        status_path = root / "status.json"
        result = {"run": json.loads(status_path.read_text()) if status_path.exists() else "not started", "training": {}}
        for condition in ("off", "vp"):
            for out in sorted((root / "train" / condition).glob("*")):
                checkpoint = latest_checkpoint(out)
                result["training"][f"{condition}/{out.name}"] = (
                    json.loads((checkpoint / "grpo_train_state.json").read_text())["step"] if checkpoint else 0)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
