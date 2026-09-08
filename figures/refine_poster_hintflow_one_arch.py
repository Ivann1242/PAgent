#!/usr/bin/env python3
"""Second-pass AutoFigure refine: cleaner poster layout + optional enhancement."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OUT = ROOT / "figures" / "autofigure_hintflow_one_v2"
OUT.mkdir(parents=True, exist_ok=True)

DESCRIPTION = r"""
Design a CLEAN, publication-ready wide architecture figure for a research POSTER
Method panel. Title: HintFlow_one.

CRITICAL LAYOUT RULES (must obey — previous version failed aesthetics):
- Wide canvas ~1600×620, generous margins, NO overlapping text/boxes/titles.
- One clear left-to-right story with TWO parallel middle tracks that merge.
- Strict alignment grid; stage headers share the same baseline.
- Short straight or gently curved arrows only; no long zigzag paths.
- Short labels; avoid paragraph text inside boxes.
- Model name must read exactly: GPT-OSS-20B (never GPT-0S or broken LaTeX).
- Trainable router color = warm amber; frozen solver = cool blue; KEEP = green; REPLACE = coral.
- White background, print-friendly, ICLR poster aesthetic.

CONTENT (accurate):

Top title: HintFlow_one — Blind Free-Form Routing with Fail-Closed Selection

Row layout:
[ Problem x ]
     |\
     | \-------------------------------> [ Small Router Qwen3-4B Blind FF-SFT ]
     |                                           |
     |                                           v
     |                                      free-form hint h
     |                                           |
     v                                           v
[ Frozen Solver GPT-OSS-20B ]           [ Frozen Solver GPT-OSS-20B ]
     no hint                                   problem + hint
     |                                           |
     v                                           v
   baseline y0                              challenger y1
     \                                           /
      \____________ [ Conservative Selector ] __/
                    Qwen3-4B orch, conf≥0.90
                    KEEP (default) | REPLACE
                              |
                              v
                         Final ŷ
            Recover / Harm / No-change legend

Bottom compact safety strip (one horizontal bar, 5 short chips):
baseline-first KEEP · unparseable→KEEP · same ans→KEEP · REPLACE iff conf≥0.90 · fail→KEEP

Caption (one line):
One-step Blind FF challenger + conservative KEEP/REPLACE; strong solver frozen.

Do NOT show multi-step plan/review, trees, DPO, or GRPO.
"""


def main() -> None:
    from autofigure import AutoFigureAgent, Config

    key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Missing OPEN_ROUTER_API_KEY")

    model = os.environ.get("AUTOFIGURE_MODEL", "google/gemini-2.5-pro")
    config = Config(
        generation_api_key=key,
        generation_provider="openrouter",
        generation_model=model,
        enhancement_api_key=key,
        enhancement_provider="openrouter",
        max_iterations=5,
        quality_threshold=9.0,
        output_dir=str(OUT / "workdir"),
        art_style=(
            "Clean modern scientific poster diagram, flat vector, "
            "balanced whitespace, crisp alignment, no clutter"
        ),
    )
    agent = AutoFigureAgent(config)
    print(f"refine model={model} out={OUT}", flush=True)
    result = agent.generate(
        description=DESCRIPTION.strip(),
        max_iterations=5,
        output_format="svg",
        topic="paper",
        enable_enhancement=True,
        enhancement_count=2,
        art_style=config.art_style,
        enhancement_input_type="code2prompt",
    )
    print(
        f"success={getattr(result, 'success', None)} "
        f"score={getattr(result, 'final_score', None)} "
        f"svg={getattr(result, 'svg_path', None)} "
        f"preview={getattr(result, 'preview_path', None)} "
        f"enhanced={getattr(result, 'enhanced_paths', None)}",
        flush=True,
    )
    mapping = [
        ("svg_path", "fig_poster_hintflow_one_architecture_v2.svg"),
        ("preview_path", "fig_poster_hintflow_one_architecture_v2.png"),
    ]
    for attr, name in mapping:
        src = getattr(result, attr, None)
        if src and Path(src).exists():
            dst = ROOT / "figures" / name
            shutil.copy2(src, dst)
            print(f"copied {dst}", flush=True)

    enhanced = getattr(result, "enhanced_paths", None) or []
    for i, src in enumerate(enhanced):
        p = Path(src)
        if p.exists():
            dst = ROOT / "figures" / f"fig_poster_hintflow_one_architecture_v2_enh{i+1}{p.suffix}"
            shutil.copy2(p, dst)
            print(f"copied {dst}", flush=True)


if __name__ == "__main__":
    main()
