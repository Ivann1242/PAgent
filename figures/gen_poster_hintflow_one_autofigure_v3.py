#!/usr/bin/env python3
"""AutoFigure regenerate: HintFlow_one poster Method with consistent LLM icons."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OUT = ROOT / "figures" / "autofigure_hintflow_one_v3"
OUT.mkdir(parents=True, exist_ok=True)

DESCRIPTION = r"""
Create a CLEAN, publication-ready WIDE research-poster Method figure.

Title:
HintFlow_one — Blind Free-Form Routing with Fail-Closed Selection

Subtitle (one line):
Train hint policy π_θ(h|x) · freeze strong solver π_S · residual KEEP/REPLACE

CRITICAL VISUAL REQUIREMENTS:
1) Every LLM module MUST show a LARGE, SAME-SIZE cute robot / AI emoji-style icon
   (friendly flat sticker look, like 🤖 or a rounded robot head). Icons must be
   IDENTICAL in size across modules — no tiny badges, no uneven scales.
2) Place the icon ABOVE the module title (centered), then title, then model line,
   then formula. Generous padding. Do NOT squeeze text beside a side icon.
3) No overlapping text. No clipped / half words. No overflowing boxes.
4) Wide landscape ~2.4:1. ICLR poster aesthetic, white background.
5) Color: amber = Hint Policy (trainable); teal-blue = frozen solvers; green = selector KEEP; coral = REPLACE.

PIPELINE (left → right):

[1 Problem]
  Document icon + big "x" (math query)

[2 Hint Policy]  ← rename from Blind router
  LARGE robot emoji icon (same size as other LLM icons)
  Qwen3-4B · Blind FF-SFT
  Formula: h ~ π_θ(h | x)
  Note: no solver trace / no y0

Then TWO parallel frozen solvers (stacked):

[3a Challenger solve]
  LARGE robot emoji icon + small lock badge (frozen)
  Frozen π_S · GPT-OSS-20B
  y1 ~ π_S(y | x, h)
  Inputs: x and hint h

[3b Bare baseline]
  LARGE robot emoji icon + small lock badge (frozen) — SAME icon size as 3a
  Frozen π_S · same weights
  y0 ~ π_S(y | x)

[4 Conservative selector]
  LARGE robot emoji icon (same size)
  Qwen3-4B · fail-closed
  d = 1[REPLACE]
  REPLACE iff conf ≥ τ , τ = 0.90 (default KEEP)
  Two clear chips: KEEP (green) | REPLACE (coral)

[5 Output]
  ŷ = (1-d) y0 + d y1

BOTTOM PANELS (must fit fully inside boxes; use wrapping / 2×3 cards):

A) Distribution change induced by the hint
   Bare: π_S(y|x) → y0   vs   Hinted: π_S(y|x,h) → y1 (shifted mass)
   Show two simple density curves (emoji-friendly flat illustration OK)
   Formula: Δe = ||e(y1) − e(y0)||_2 ; selector gates acceptance

B) Paired outcomes vs baseline (FULL words, no clipping)
   ● Recover: EM(y0)=0, EM(ŷ)=1
   ● Harm: EM(y0)=1, EM(ŷ)=0
   ● No-change: EM(y0)=EM(ŷ)

C) Fail-closed rules as 6 equal cards (2 rows × 3 cols), no overflow:
   R1 baseline-first → KEEP
   R2 unparseable y1 → KEEP
   R3 same answer → KEEP
   R4 REPLACE only if conf ≥ τ
   R5 selector failure → KEEP
   R6 ŷ = y1 iff d=1 else y0

Footer:
Inference-only residual pipeline · one-step Blind FF challenger · no multi-turn tree credit assignment

Do NOT invent multi-step plan/review trees, DPO, or GRPO.
Model name must be exactly GPT-OSS-20B (never GPT-0S).
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
        quality_threshold=8.8,
        output_dir=str(OUT / "workdir"),
        art_style=(
            "Cute flat scientific poster diagram with consistent large robot emoji icons "
            "for every LLM block, generous whitespace, no overflow, print-ready"
        ),
    )
    agent = AutoFigureAgent(config)
    print(f"AutoFigure v3 model={model} out={OUT}", flush=True)
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
        ("svg_path", "fig_poster_hintflow_one_architecture_af.svg"),
        ("preview_path", "fig_poster_hintflow_one_architecture_af.png"),
    ]
    for attr, name in mapping:
        src = getattr(result, attr, None)
        if src and Path(src).exists():
            dst = ROOT / "figures" / name
            shutil.copy2(src, dst)
            print(f"copied {dst}", flush=True)

    for i, src in enumerate(getattr(result, "enhanced_paths", None) or []):
        p = Path(src)
        if p.exists():
            dst = ROOT / "figures" / f"fig_poster_hintflow_one_architecture_af_enh{i+1}{p.suffix}"
            shutil.copy2(p, dst)
            print(f"copied {dst}", flush=True)


if __name__ == "__main__":
    main()
