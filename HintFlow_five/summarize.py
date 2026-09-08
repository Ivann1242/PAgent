#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import load_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="checkpoints/eval_hintflow_five_iid512")
    args = p.parse_args()
    out = Path(args.out_dir)
    by_mode = {m: {r["id"]: r for r in load_jsonl(out / f"{m}.jsonl")}
               for m in ("baseline", "router", "oracle")}
    ids = sorted(set.intersection(*(set(v) for v in by_mode.values())))
    result = {"n_paired": len(ids)}
    for mode, rows in by_mode.items():
        result[mode] = {"em": sum(int(rows[i].get("em") or 0) for i in ids) / max(len(ids), 1)}
    base, router = by_mode["baseline"], by_mode["router"]
    result["router_vs_baseline"] = {
        "delta": result["router"]["em"] - result["baseline"]["em"],
        "recovered": sum(not base[i].get("em") and bool(router[i].get("em")) for i in ids),
        "harmed": sum(bool(base[i].get("em")) and not router[i].get("em") for i in ids),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
