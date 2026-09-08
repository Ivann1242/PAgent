#!/usr/bin/env python3
"""Poster Method figure: HintFlow_one (clean layout + research notation).

Fixes AutoFigure overflow / clipped legend text, and adds distribution-shift
formulas that match the project's Blind-FF + fail-closed selector story.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "figures" / "fig_poster_hintflow_one_architecture_clean.pdf"
OUT_PNG = ROOT / "figures" / "fig_poster_hintflow_one_architecture_clean.png"

# Palette (print-friendly, not purple-default)
C_BG = "none"  # transparent canvas for slides / overlay
C_BOX = "#FFFFFF"
C_EDGE = "#2F3A44"
C_SOLVER = "#1F6F8B"      # frozen solver teal-blue
C_SOLVER_FILL = "#E8F4F8"
C_ROUTER = "#C47B2B"      # trainable amber
C_ROUTER_FILL = "#FFF4E6"
C_KEEP = "#1B7F5A"
C_REPLACE = "#C0392B"
C_MUTED = "#6B7280"
C_PANEL = "#F3F4F6"
C_REC = "#009E73"
C_HARM = "#D55E00"
C_NC = "#9CA3AF"


def round_box(ax, xy, w, h, *, fc, ec, lw=1.2, r=0.03, z=2):
    p = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
        mutation_aspect=0.6,
    )
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, *, color=C_EDGE, lw=1.35, style="-|>", rad=0.0, z=3):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
            shrinkA=2, shrinkB=2,
        )
    )


def text(ax, x, y, s, *, size=8.5, color=C_EDGE, weight="normal", ha="center", va="center", family=None):
    kw = dict(
        fontsize=size, color=color, fontweight=weight,
        ha=ha, va=va, zorder=5,
    )
    if family:
        kw["fontfamily"] = family
    ax.text(x, y, s, **kw)


_EMOJI_DIR = ROOT / "figures" / "_emoji_assets"
_ICON_CACHE: dict[str, object] = {}


def _icon_img(name: str):
    if name not in _ICON_CACHE:
        _ICON_CACHE[name] = plt.imread(_EMOJI_DIR / name)
    return _ICON_CACHE[name]


def queue_llm_icon(
    jobs: list,
    cx: float,
    cy: float,
    *,
    color: str,
    scale: float = 1.0,
    frozen: bool = False,
) -> None:
    """Record icon placement; pasted as true pixel circles after savefig."""
    jobs.append({"cx": cx, "cy": cy, "color": color, "scale": scale, "frozen": frozen})


def paste_llm_icons(ax, fig, png_path: Path, jobs: list, *, dpi: int) -> None:
    """Paste circular badges onto the saved PNG (avoids axes-aspect squash)."""
    from PIL import Image

    # Force the same dpi used by savefig so data→pixel mapping matches the PNG.
    fig.set_dpi(dpi)
    fig.canvas.draw()
    base = Image.open(png_path).convert("RGBA")
    W, H = base.size
    canvas_w = fig.get_figwidth() * dpi
    canvas_h = fig.get_figheight() * dpi
    sx = W / canvas_w
    sy = H / canvas_h

    for job in jobs:
        if job["color"].upper() in {C_ROUTER.upper(), "#C47B2B"}:
            badge = Image.open(_EMOJI_DIR / "badge_amber.png").convert("RGBA")
        elif job["color"].upper() in {C_KEEP.upper(), "#1B7F5A"}:
            badge = Image.open(_EMOJI_DIR / "badge_green.png").convert("RGBA")
        else:
            badge = Image.open(_EMOJI_DIR / "badge_teal.png").convert("RGBA")

        # Perfect square in PNG pixels → true circle.
        side = int(round(0.58 * dpi * job["scale"]))  # ~0.58" at scale=1
        badge = badge.resize((side, side), Image.Resampling.LANCZOS)

        disp_x, disp_y = ax.transData.transform((job["cx"], job["cy"]))
        px = int(round(disp_x * sx - side / 2))
        py = int(round((canvas_h - disp_y) * sy - side / 2))
        px = max(0, min(px, W - side))
        py = max(0, min(py, H - side))
        base.alpha_composite(badge, (px, py))

        if job["frozen"]:
            lock = Image.open(_EMOJI_DIR / "lock_badge.png").convert("RGBA")
            ls = max(int(round(side * 0.36)), 20)
            lock = lock.resize((ls, ls), Image.Resampling.LANCZOS)
            lx = min(px + side - ls // 3, W - ls)
            ly = max(py - ls // 6, 0)
            base.alpha_composite(lock, (lx, ly))

    base.save(png_path)
    print(f"pasted {len(jobs)} circular LLM badges → {png_path}")



def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 320,
    })

    fig_w, fig_h = 13.4, 7.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 108)
    ax.axis("off")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    # Title
    text(ax, 50, 104.5, r"HintFlow$_{\mathrm{one}}$: Blind Free-Form Routing with Fail-Closed Selection",
         size=13.5, weight="bold")
    text(ax, 50, 100.6,
         r"Train hint policy $\pi_\theta(h\!\mid\!x)$  ·  freeze strong solver $\pi_{\mathrm{S}}$  ·  residual KEEP/REPLACE",
         size=8.6, color=C_MUTED)

    # ---------------- Stage columns ----------------
    # Identical large LLM icons centered ABOVE titles (pasted as true circles).
    ICON = 1.05
    icon_jobs: list = []

    # Stage 1: problem
    round_box(ax, (2.0, 64.0), 11.0, 26.0, fc=C_BOX, ec=C_EDGE, lw=1.15)
    text(ax, 7.5, 85.5, "1  Problem", size=8.2, weight="bold", color=C_MUTED)
    text(ax, 7.5, 78.0, r"$x$", size=16, weight="bold")
    text(ax, 7.5, 70.5, "math query", size=7.8, color=C_MUTED)

    # Stage 2: Hint Policy
    round_box(ax, (16.0, 62.0), 18.0, 30.0, fc=C_ROUTER_FILL, ec=C_ROUTER, lw=1.4)
    queue_llm_icon(icon_jobs, 25.0, 86.5, color=C_ROUTER, scale=ICON, frozen=False)
    text(ax, 25.0, 78.6, "2  Hint Policy", size=8.5, weight="bold", color=C_ROUTER)
    text(ax, 25.0, 74.6, "Qwen3-4B  ·  Blind FF-SFT", size=7.4, color=C_EDGE)
    text(ax, 25.0, 70.0, r"$h\sim\pi_\theta(h\!\mid\!x)$", size=9.8, weight="bold")
    text(ax, 25.0, 65.4, "no solver trace / no $y_0$", size=7.1, color=C_MUTED)

    # Stage 3a: Challenger
    round_box(ax, (39.0, 73.0), 21.5, 22.5, fc=C_SOLVER_FILL, ec=C_SOLVER, lw=1.35)
    queue_llm_icon(icon_jobs, 49.75, 90.5, color=C_SOLVER, scale=ICON, frozen=True)
    text(ax, 49.75, 82.6, "3a  Challenger solve", size=8.2, weight="bold", color=C_SOLVER)
    text(ax, 49.75, 78.8, r"Frozen $\pi_{\mathrm{S}}$  ·  GPT-OSS-20B", size=7.3)
    text(ax, 49.75, 74.8, r"$y_1\sim\pi_{\mathrm{S}}(y\!\mid\!x,h)$", size=9.4, weight="bold")

    # Stage 3b: Baseline
    round_box(ax, (39.0, 48.0), 21.5, 22.5, fc=C_SOLVER_FILL, ec=C_SOLVER, lw=1.35)
    queue_llm_icon(icon_jobs, 49.75, 65.5, color=C_SOLVER, scale=ICON, frozen=True)
    text(ax, 49.75, 57.6, "3b  Bare baseline", size=8.2, weight="bold", color=C_SOLVER)
    text(ax, 49.75, 53.8, r"Frozen $\pi_{\mathrm{S}}$  ·  same weights", size=7.3)
    text(ax, 49.75, 49.8, r"$y_0\sim\pi_{\mathrm{S}}(y\!\mid\!x)$", size=9.4, weight="bold")

    # Stage 4: selector
    round_box(ax, (65.5, 52.0), 18.5, 37.0, fc="#F7FAF8", ec=C_KEEP, lw=1.35)
    queue_llm_icon(icon_jobs, 74.75, 83.5, color=C_KEEP, scale=ICON, frozen=False)
    text(ax, 74.75, 75.6, "4  Conservative selector", size=8.1, weight="bold", color=C_KEEP)
    text(ax, 74.75, 71.4, "Qwen3-4B  ·  fail-closed", size=7.3, color=C_MUTED)
    text(ax, 74.75, 66.6, r"$d=\mathbb{1}[\mathrm{REPLACE}]$", size=9.2, weight="bold")
    text(ax, 74.75, 62.2, r"REPLACE iff $\mathrm{conf}\,\geq\,\tau$", size=7.7)
    text(ax, 74.75, 58.2, r"$\tau=0.90$ (default KEEP)", size=7.3, color=C_MUTED)
    round_box(ax, (68.0, 53.4), 6.0, 3.3, fc="#E6F6EE", ec=C_KEEP, lw=1.0, r=0.08)
    text(ax, 71.0, 55.05, "KEEP", size=7.4, weight="bold", color=C_KEEP)
    round_box(ax, (75.4, 53.4), 6.8, 3.3, fc="#FDECEA", ec=C_REPLACE, lw=1.0, r=0.08)
    text(ax, 78.8, 55.05, "REPLACE", size=7.2, weight="bold", color=C_REPLACE)

    # Stage 5: output
    round_box(ax, (87.5, 62.0), 10.5, 26.0, fc=C_BOX, ec=C_EDGE, lw=1.2)
    text(ax, 92.75, 83.5, "5  Output", size=8.2, weight="bold", color=C_MUTED)
    text(ax, 92.75, 76.0, r"$\hat{y}$", size=16, weight="bold")
    text(ax, 92.75, 68.5, r"$\hat{y}=(1\!-\!d)\,y_0+d\,y_1$", size=7.3)

    # ---------------- Arrows ----------------
    arrow(ax, (13.1, 76.0), (15.9, 77.5), color=C_ROUTER, lw=1.4)
    arrow(ax, (7.5, 63.8), (7.5, 46.0), color=C_SOLVER, lw=1.15, style="-")
    arrow(ax, (7.5, 46.0), (39.0, 56.0), color=C_SOLVER, lw=1.15, rad=-0.02)
    arrow(ax, (13.1, 84.0), (39.0, 85.0), color=C_SOLVER, lw=1.05, rad=0.14)
    text(ax, 26.0, 94.5, r"$x$", size=7.5, color=C_SOLVER)
    arrow(ax, (34.1, 76.5), (38.9, 83.0), color=C_ROUTER, lw=1.35)
    text(ax, 35.6, 83.5, r"$h$", size=7.8, color=C_ROUTER, weight="bold")

    arrow(ax, (60.6, 84.0), (65.4, 76.0), color=C_EDGE, lw=1.2)
    arrow(ax, (60.6, 59.0), (65.4, 66.0), color=C_EDGE, lw=1.2)
    arrow(ax, (84.1, 70.5), (87.4, 70.5), color=C_EDGE, lw=1.35)

    text(ax, 61.4, 87.5, r"$y_1$", size=9, weight="bold", color=C_SOLVER, ha="left")
    text(ax, 61.4, 55.5, r"$y_0$", size=9, weight="bold", color=C_SOLVER, ha="left")

    # ---------------- Distribution-change panel ----------------
    round_box(ax, (2.2, 28.5), 58.8, 16.5, fc="#FFFFFF", ec="#D1D5DB", lw=1.0, r=0.02)
    text(ax, 31.6, 42.8, "Distribution change induced by the hint", size=9.0, weight="bold", ha="center")

    ax.add_patch(Rectangle((5.5, 30.8), 22.0, 9.8, facecolor=C_SOLVER_FILL, edgecolor=C_SOLVER, lw=1.0, zorder=2))
    text(ax, 12.2, 38.6, r"Bare", size=7.4, weight="bold", color=C_SOLVER, ha="left")
    text(ax, 12.2, 35.8, r"$\pi_{\mathrm{S}}(y\!\mid\!x)$", size=8.4, ha="left")
    text(ax, 12.2, 32.8, r"sample $y_0$", size=7.0, color=C_MUTED, ha="left")
    xs = np.linspace(18.2, 26.2, 60)
    ys = 32.4 + 4.8 * np.exp(-0.5 * ((xs - 20.8) / 1.35) ** 2)
    ax.plot(xs, ys, color=C_SOLVER, lw=1.4, zorder=4)
    ax.fill_between(xs, 32.4, ys, color=C_SOLVER, alpha=0.18, zorder=3)

    arrow(ax, (28.3, 35.5), (32.8, 35.5), color=C_ROUTER, lw=1.6, style="-|>")
    text(ax, 30.55, 37.6, r"$h\sim\pi_\theta$", size=7.2, color=C_ROUTER)

    ax.add_patch(Rectangle((33.8, 30.8), 24.8, 9.8, facecolor=C_ROUTER_FILL, edgecolor=C_ROUTER, lw=1.0, zorder=2))
    text(ax, 35.2, 38.6, r"Hinted", size=7.4, weight="bold", color=C_ROUTER, ha="left")
    text(ax, 35.2, 35.8, r"$\pi_{\mathrm{S}}(y\!\mid\!x,h)$", size=8.4, ha="left")
    text(ax, 35.2, 32.8, r"sample $y_1$", size=7.0, color=C_MUTED, ha="left")
    xs2 = np.linspace(45.0, 56.5, 70)
    ys_old = 32.4 + 3.4 * np.exp(-0.5 * ((xs2 - 48.0) / 1.4) ** 2)
    ys_new = 32.4 + 4.6 * np.exp(-0.5 * ((xs2 - 52.2) / 1.25) ** 2)
    ax.plot(xs2, ys_old, color=C_SOLVER, lw=1.0, ls=(0, (3, 2)), zorder=4, alpha=0.7)
    ax.plot(xs2, ys_new, color=C_ROUTER, lw=1.45, zorder=4)
    ax.fill_between(xs2, 32.4, ys_new, color=C_ROUTER, alpha=0.16, zorder=3)
    text(ax, 52.4, 38.8, "shift", size=6.8, color=C_ROUTER)

    text(ax, 31.6, 29.3,
         r"Solution embedding shift $\Delta e=\|e(y_1)-e(y_0)\|_2$  ·  selector gates acceptance of the shift",
         size=7.2, color=C_MUTED, ha="center")

    # ---------------- Paired outcomes ----------------
    round_box(ax, (63.0, 28.5), 35.0, 16.5, fc="#FFFFFF", ec="#D1D5DB", lw=1.0, r=0.02)
    text(ax, 80.5, 42.8, "Paired outcomes vs baseline", size=9.0, weight="bold")

    rows = [
        (C_REC, "Recover", r"$\mathrm{EM}(y_0)=0,\ \mathrm{EM}(\hat{y})=1$"),
        (C_HARM, "Harm", r"$\mathrm{EM}(y_0)=1,\ \mathrm{EM}(\hat{y})=0$"),
        (C_NC, "No-change", r"$\mathrm{EM}(y_0)=\mathrm{EM}(\hat{y})$"),
    ]
    y0 = 38.8
    for i, (c, name, form) in enumerate(rows):
        y = y0 - i * 3.3
        # markersize is in display points → stays circular (unlike data-space Circle)
        ax.plot(
            66.2, y, "o",
            markersize=9.5, color=c, markeredgecolor="none",
            zorder=4, clip_on=False,
        )
        text(ax, 68.2, y, name, size=8.2, weight="bold", color=c, ha="left")
        text(ax, 78.0, y, form, size=8.0, color=C_EDGE, ha="left")

    # ---------------- Fail-closed rules ----------------
    round_box(ax, (2.2, 6.0), 95.8, 19.5, fc=C_PANEL, ec="#D1D5DB", lw=1.0, r=0.015)
    text(ax, 50, 23.0, "Fail-closed selection rules", size=9.2, weight="bold")

    rules = [
        (C_KEEP, "R1  Baseline-first", "default decision = KEEP"),
        (C_KEEP, "R2  Unparseable $y_1$", r"$\Rightarrow$ KEEP"),
        (C_KEEP, "R3  Same answer", r"$y_1\equiv y_0\ \Rightarrow$ KEEP"),
        (C_REPLACE, "R4  High-confidence gate", r"REPLACE only if $\mathrm{conf}\,\geq\,\tau$"),
        (C_KEEP, "R5  Selector failure", r"exception / parse fail $\Rightarrow$ KEEP"),
        (C_EDGE, "R6  Acceptance", r"$\hat{y}=y_1$ iff $d=1$, else $y_0$"),
    ]
    for i, (c, title, body) in enumerate(rules):
        col, row = i % 3, i // 3
        x = 4.0 + col * 31.5
        y = 14.4 - row * 7.2
        round_box(ax, (x, y), 30.0, 6.2, fc="#FFFFFF", ec=c, lw=1.05, r=0.04)
        text(ax, x + 15.0, y + 4.15, title, size=7.6, weight="bold", color=c)
        text(ax, x + 15.0, y + 1.85, body, size=7.4, color=C_EDGE)

    text(ax, 50, 2.8,
         r"Inference-only residual pipeline  ·  one-step Blind FF challenger  ·  no multi-turn tree credit assignment",
         size=7.3, color=C_MUTED)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    # Fixed canvas (no tight crop) so data→pixel mapping matches the PNG.
    fig.subplots_adjust(left=0.03, right=0.995, top=0.97, bottom=0.035)
    dpi = 320
    fig.set_dpi(dpi)
    fig.savefig(
        OUT_PNG, dpi=dpi, bbox_inches=None,
        facecolor="none", edgecolor="none", transparent=True,
    )
    paste_llm_icons(ax, fig, OUT_PNG, icon_jobs, dpi=dpi)
    # PDF: keep alpha when possible (Google Slides prefers the PNG).
    try:
        import fitz  # PyMuPDF

        doc = fitz.open()
        pix = fitz.Pixmap(str(OUT_PNG))
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, filename=str(OUT_PNG))
        doc.save(OUT_PDF)
        doc.close()
    except Exception:
        # Fallback: matplotlib vector PDF (no raster icons) with transparent page.
        fig.savefig(
            OUT_PDF, dpi=dpi, bbox_inches=None,
            facecolor="none", edgecolor="none", transparent=True,
        )
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
