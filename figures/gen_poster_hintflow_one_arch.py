#!/usr/bin/env python3
"""Generate HintFlow_one architecture figure for poster Method via AutoFigure."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OUT = ROOT / "figures" / "autofigure_hintflow_one"
OUT.mkdir(parents=True, exist_ok=True)

DESCRIPTION = """
Create a clean, publication-ready METHOD architecture figure for a research poster
titled HintFlow_one (one-step Blind Free-Form routing with conservative selection).

SCIENTIFIC STORY (must be accurate):
HintFlow_one is a residual, baseline-first inference pipeline for hard math.
A frozen strong solver (GPT-OSS-20B) is never fine-tuned. A small router
(Qwen3-4B Blind FF-SFT) proposes a free-form natural-language hint from the
problem alone (blind: no solver trace). The same solver then produces a
challenger solution conditioned on that hint. A conservative KEEP/REPLACE
selector compares bare baseline vs challenger and only replaces the incumbent
when confidence is high (default ≥ 0.90), otherwise fail-closed KEEP.
Goal: keep recoveries from Blind FF while suppressing harm to correct baselines.

LAYOUT (left-to-right horizontal pipeline, poster-friendly, single row of stages
with a bottom safety-rules callout):

Stage 0 — Input
  Box: Math problem x

Stage 1 — Bare Baseline (always-on)
  Model badge: Frozen Solver  GPT-OSS-20B
  Arrow from problem → Solver (no hint)
  Output box: Incumbent / Baseline solution y0

Stage 2 — Blind Free-Form Router
  Model badge: Small Router  Qwen3-4B (Blind FF-SFT)
  Arrow from problem only (emphasize Blind: problem → hint, no y0)
  Output box: Free-form hint h

Stage 3 — Challenger Solve
  Model badge: Frozen Solver  GPT-OSS-20B (same weights)
  Inputs: problem + hint h
  Output box: Challenger solution y1

Stage 4 — Conservative Selector
  Model badge: Orch selector (Qwen3-4B) or rule fallback
  Inputs: y0 and y1 (+ problem)
  Decision diamond/box: KEEP | REPLACE
  Hard constraints shown as small guard icons / bullets:
    • baseline-first default KEEP
    • unparseable challenger → KEEP
    • same answer → KEEP
    • REPLACE only if conf ≥ 0.90
    • selector failure → KEEP

Stage 5 — Final
  Output: Final answer ŷ  (ŷ = y1 if REPLACE else y0)
  Tiny outcome tags: Recover / Harm / No-change (paired vs baseline)

VISUAL STYLE (ICLR / NeurIPS poster aesthetic):
- Clean vector SVG, flat modern scientific illustration
- Soft blue/teal for frozen solver modules; warm amber/coral for the trainable router
- Green for KEEP / safety; muted orange for REPLACE
- Thin elegant arrows; rounded rectangles; clear typography
- Minimal text; no paragraph blocks; labels short
- White/light background; high contrast for print
- Aspect ratio ~2.4:1 (wide poster method panel)
- Optional small caption under figure:
  "HintFlow_one: one-step Blind FF challenger + fail-closed KEEP/REPLACE"

DO NOT invent multi-step plan/review loops, trees, DPO, or GRPO in this figure.
This figure is inference-time HintFlow_one only.
"""


def main() -> None:
    from autofigure import AutoFigureAgent, Config

    key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Missing OPEN_ROUTER_API_KEY in .env")

    # Prefer a widely available OpenRouter Gemini; fall back handled by API errors.
    model = os.environ.get(
        "AUTOFIGURE_MODEL",
        "google/gemini-2.5-pro",
    )

    config = Config(
        generation_api_key=key,
        generation_provider="openrouter",
        generation_model=model,
        max_iterations=4,
        quality_threshold=8.5,
        output_dir=str(OUT / "workdir"),
    )
    agent = AutoFigureAgent(config)
    print(f"AutoFigure model={model} out={OUT}", flush=True)
    result = agent.generate(
        description=DESCRIPTION.strip(),
        max_iterations=4,
        output_format="svg",
        topic="paper",
        enable_enhancement=False,
    )
    print(
        f"success={getattr(result, 'success', None)} "
        f"score={getattr(result, 'final_score', None)} "
        f"svg={getattr(result, 'svg_path', None)} "
        f"preview={getattr(result, 'preview_path', None)}",
        flush=True,
    )
    # Copy stable deliverables
    for attr, name in [
        ("svg_path", "fig_poster_hintflow_one_architecture.svg"),
        ("preview_path", "fig_poster_hintflow_one_architecture.png"),
    ]:
        src = getattr(result, attr, None)
        if src and Path(src).exists():
            dst = OUT.parent / name
            shutil.copy2(src, dst)
            print(f"copied {src} -> {dst}", flush=True)

    # Also dump any xml/drawio if present in workdir
    work = OUT / "workdir"
    if work.exists():
        for p in sorted(work.rglob("*")):
            if p.suffix.lower() in {".svg", ".png", ".xml", ".json"} and p.is_file():
                print(f"artifact: {p} ({p.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
