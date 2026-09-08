#!/usr/bin/env python3
"""IID HintFlow solution-embedding UMAP arrows + ||Δ|| violin.

Unified 2×3 figure (default):

  python figures/gen_fig_idist_solution_umap_arrows.py --budget 8k
  python figures/gen_fig_idist_solution_umap_arrows.py --budget 4k --target both
  python figures/gen_fig_idist_solution_umap_arrows.py --budget 20k

Single-row figures still available:

  python figures/gen_fig_idist_solution_umap_arrows.py --budget 8k --target challenger
  python figures/gen_fig_idist_solution_umap_arrows.py --budget 8k --target hf1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
GPU = os.environ.get("EMBED_GPU", "1")
MAX_CHARS = 6000
SEED = 41

C_REC = "#009E73"
C_HARM = "#D55E00"
C_NC = "#999999"
C_BASE = "#9AA0A6"

BUDGETS = ("4k", "8k", "20k")


def budget_paths(budget: str) -> tuple[Path, Path]:
    data = ROOT / f"checkpoints/eval_hintflow_one_idist_1809_{budget}/hintflow_one.jsonl"
    cache = ROOT / f"figures/cache_idist_{budget}_solution_embeds.npz"
    return data, cache


def row_meta(budget: str) -> dict[str, dict[str, str]]:
    return {
        "challenger": {
            "row_label": "Always-on\nchallenger",
            "end_label": "challenger ○",
            "out_stem": f"fig_idist_{budget}_solution_umap_arrows",
        },
        "hf1": {
            "row_label": "HintFlow_one\n(HF1)",
            "end_label": "HF1 final ○",
            "out_stem": f"fig_idist_{budget}_hf1_solution_umap_arrows",
        },
    }


def trunc(text: str) -> str:
    text = (text or "").strip()
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


def decision(row: dict) -> str:
    sel = row.get("selection") or {}
    if isinstance(sel, dict):
        return str(sel.get("decision") or "KEEP").upper()
    return str(sel).upper()


def paired_outcome(baseline_em: int, other_em: int) -> str:
    b = int(baseline_em) >= 1
    o = int(other_em) >= 1
    if (not b) and o:
        return "recovery"
    if b and (not o):
        return "harm"
    return "no_change"


def load_rows(data_path: Path) -> list[dict]:
    rows = []
    with data_path.open() as f:
        for line in f:
            r = json.loads(line)
            r["base_sol"] = trunc(r["baseline"].get("solution") or "")
            r["chal_sol"] = trunc(r["challenger"].get("solution") or "")
            rows.append(r)
    return rows


def embed_all(rows: list[dict], cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if z["model_name"].item() == MODEL_NAME and len(z["base"]) == len(rows):
            print(f"loaded cache {cache}", flush=True)
            return z["base"], z["chal"]

    from sentence_transformers import SentenceTransformer

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    print(f"embedding with {MODEL_NAME} on GPU {GPU}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    base = model.encode(
        [r["base_sol"] for r in rows],
        batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    chal = model.encode(
        [r["chal_sol"] for r in rows],
        batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, base=base, chal=chal,
        model_name=np.array(MODEL_NAME),
        ids=np.array([r["id"] for r in rows], dtype=object),
    )
    print(f"saved cache {cache}", flush=True)
    return base, chal


def resolve_target(
    target: str, rows: list[dict], base_e: np.ndarray, chal_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if target == "challenger":
        labels = np.array([
            paired_outcome(r["baseline_em"], r["challenger_em"]) for r in rows
        ])
        return chal_e, labels
    if target == "hf1":
        decs = np.array([decision(r) for r in rows])
        end = np.where(decs[:, None] == "REPLACE", chal_e, base_e)
        labels = np.array([
            paired_outcome(r["baseline_em"], r["em"]) for r in rows
        ])
        return end, labels
    raise ValueError(f"unknown target: {target}")


def _set_rc() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _style_umap(ax, xlim, ylim, *, xlabel: bool, ylabel: str | None) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel("UMAP-1")
    if ylabel:
        ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.6)
    ax.set_facecolor("#FAFAFA")


def _draw_arrows(ax, xb, yb, xe, ye, color: str) -> None:
    if len(xb) == 0:
        return
    ax.quiver(
        xb, yb, xe - xb, ye - yb,
        angles="xy", scale_units="xy", scale=1.0,
        width=0.0030, headwidth=3.2, headlength=3.8, headaxislength=3.2,
        color=color, alpha=0.35, zorder=2,
    )
    ax.scatter(xb, yb, s=8, c=C_BASE, alpha=0.55, linewidths=0, zorder=3)
    ax.scatter(
        xe, ye, s=32, facecolors="none", edgecolors=color,
        linewidths=1.1, alpha=0.85, zorder=4,
    )


def _draw_violin(ax, labels: np.ndarray, delta_norm: np.ndarray, ylim=None) -> None:
    order = ["recovery", "harm", "no_change"]
    colors = [C_REC, C_HARM, C_NC]
    counts = [int((labels == k).sum()) for k in order]
    data = [delta_norm[labels == k] for k in order]
    parts = ax.violinplot(
        data, positions=[0, 1, 2],
        showmeans=False, showextrema=False, showmedians=False,
    )
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.35)
        body.set_linewidth(0.8)
    bp = ax.boxplot(
        data, positions=[0, 1, 2], widths=0.18, showfliers=False,
        patch_artist=True,
        medianprops=dict(color="#222222", linewidth=1.2),
        whiskerprops=dict(color="#555555", linewidth=0.8),
        capprops=dict(color="#555555", linewidth=0.8),
        boxprops=dict(facecolor="white", edgecolor="#555555", linewidth=0.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.9)
    means = [float(np.mean(d)) if len(d) else 0.0 for d in data]
    ax.scatter(
        [0, 1, 2], means, marker="D", s=22, c=colors, zorder=5,
        edgecolors="white", linewidths=0.4,
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        [
            f"Recovery\n($n$={counts[0]})",
            f"Harm\n($n$={counts[1]})",
            f"No change\n($n$={counts[2]})",
        ],
        fontsize=7.5,
    )
    ax.set_ylabel(r"$\|\Delta\|_2$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    y1 = ax.get_ylim()[1]
    for i, m in enumerate(means):
        ax.text(
            i, m + 0.03 * y1, f"{m:.2f}",
            ha="center", va="bottom", fontsize=7, color="#333333",
        )
    return means, counts


def fit_umap_for_one_view(
    base_e: np.ndarray,
    end_e: np.ndarray,
    labels: np.ndarray,
    *,
    tag: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    """UMAP fit on this view's Recovery/Harm endpoints only (old single-fig behavior)."""
    n = len(base_e)
    flip = (labels == "recovery") | (labels == "harm")
    idx = np.where(flip)[0]
    pts = np.vstack([base_e[idx], end_e[idx]])
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.25, metric="cosine",
        random_state=SEED, n_jobs=1,
    )
    xy = reducer.fit_transform(pts)
    k = len(idx)
    base_xy = np.full((n, 2), np.nan, dtype=np.float64)
    end_xy = np.full((n, 2), np.nan, dtype=np.float64)
    base_xy[idx] = xy[:k]
    end_xy[idx] = xy[k:]
    all_xy = xy
    x0, x1 = all_xy[:, 0].min(), all_xy[:, 0].max()
    y0, y1 = all_xy[:, 1].min(), all_xy[:, 1].max()
    pad_x = 0.06 * (x1 - x0 + 1e-6)
    pad_y = 0.06 * (y1 - y0 + 1e-6)
    xlim = (x0 - pad_x, x1 + pad_x)
    ylim = (y0 - pad_y, y1 + pad_y)
    print(f"[{tag}] UMAP fit on {k} flip questions ({2 * k} endpoints)", flush=True)
    return base_xy, end_xy, xlim, ylim


def plot_combined(
    rows: list[dict],
    base_e: np.ndarray,
    chal_e: np.ndarray,
    out_pdf: Path,
    out_png: Path,
    *,
    budget: str = "8k",
    include_displacement: bool = True,
) -> None:
    """2×N: rows=challenger/HF1, cols=Recovery/Harm[ / Displacement].

    Each row fits its own UMAP on that row's flip endpoints — same as the
    original separate figures — so HF1 recovery geometry is not warped by
    pooling with challenger flips. Violin y-limits are still shared when shown.
    """
    _set_rc()

    views = {}
    for name in ("challenger", "hf1"):
        end_e, labels = resolve_target(name, rows, base_e, chal_e)
        base_xy, end_xy, xlim, ylim = fit_umap_for_one_view(
            base_e, end_e, labels, tag=name,
        )
        views[name] = {
            "labels": labels,
            "base_xy": base_xy,
            "end_xy": end_xy,
            "xlim": xlim,
            "ylim": ylim,
            "delta": np.linalg.norm(end_e - base_e, axis=1),
            "n_rec": int((labels == "recovery").sum()),
            "n_harm": int((labels == "harm").sum()),
            "n_nc": int((labels == "no_change").sum()),
        }
        print(
            f"[{name}] recovery={views[name]['n_rec']} "
            f"harm={views[name]['n_harm']} no_change={views[name]['n_nc']}",
            flush=True,
        )

    from matplotlib.gridspec import GridSpec

    n_data_cols = 3 if include_displacement else 2
    if include_displacement:
        fig = plt.figure(figsize=(11.2, 6.0))
        gs = GridSpec(
            2, 4, figure=fig,
            width_ratios=[1.15, 1.15, 1.0, 0.72],
            wspace=0.22, hspace=0.30,
            left=0.09, right=0.98, top=0.90, bottom=0.07,
        )
        dmax = max(float(v["delta"].max()) for v in views.values())
        vlim = (0.0, dmax * 1.12)
    else:
        fig = plt.figure(figsize=(8.6, 6.0))
        gs = GridSpec(
            2, 3, figure=fig,
            width_ratios=[1.15, 1.15, 0.78],
            wspace=0.22, hspace=0.30,
            left=0.10, right=0.98, top=0.90, bottom=0.07,
        )
        vlim = None

    axes = np.empty((2, n_data_cols), dtype=object)
    for i in range(2):
        for j in range(n_data_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])
    ax_side = fig.add_subplot(gs[:, n_data_cols])
    ax_side.axis("off")

    col_titles = ["Recovery", "Harm"]
    if include_displacement:
        col_titles.append("Displacement magnitude")
    for j, title in enumerate(col_titles):
        color = C_REC if j == 0 else (C_HARM if j == 1 else "#333333")
        axes[0, j].set_title(title, pad=8, color=color, fontsize=10.5)

    meta_by = row_meta(budget)
    for i, name in enumerate(("challenger", "hf1")):
        v = views[name]
        meta = meta_by[name]
        labels = v["labels"]
        base_xy = v["base_xy"]
        end_xy = v["end_xy"]
        xlim, ylim = v["xlim"], v["ylim"]

        for j, (cls, color) in enumerate(
            [("recovery", C_REC), ("harm", C_HARM)]
        ):
            ax = axes[i, j]
            m = labels == cls
            _draw_arrows(
                ax,
                base_xy[m, 0], base_xy[m, 1],
                end_xy[m, 0], end_xy[m, 1],
                color,
            )
            n_cls = int(m.sum())
            ax.text(
                0.03, 0.03, f"$n$={n_cls}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color=color,
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white",
                    edgecolor="#E5E5E5", alpha=0.9, linewidth=0.6,
                ),
            )
            _style_umap(
                ax, xlim, ylim,
                xlabel=(i == 1),
                ylabel=("UMAP-2" if j == 0 else None),
            )

        if include_displacement:
            axv = axes[i, 2]
            _draw_violin(axv, labels, v["delta"], ylim=vlim)
            if i == 0:
                axv.set_xticklabels([])
            axv.set_title("")

        axes[i, 0].annotate(
            meta["row_label"],
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.28, 0.5), textcoords="axes fraction",
            ha="center", va="center", fontsize=9.5, color="#222222",
            rotation=90,
            fontweight="semibold",
        )

    # Right-side legend + notes (spans both rows)
    legend_elems = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=C_BASE, markeredgecolor=C_BASE,
            markersize=5.5, label="baseline ●",
        ),
        Line2D(
            [0], [0], marker="o", color="#444444",
            markerfacecolor="none", markeredgecolor="#444444",
            markersize=7, markeredgewidth=1.2,
            label="target ○ (chal. / HF1)",
        ),
        Line2D([0], [0], color=C_REC, lw=2.0, label="Recovery arrow"),
        Line2D([0], [0], color=C_HARM, lw=2.0, label="Harm arrow"),
    ]
    leg = ax_side.legend(
        handles=legend_elems, loc="upper left",
        frameon=True, fancybox=False, edgecolor="#DDDDDD",
        fontsize=8, borderpad=0.55, labelspacing=0.7,
        handlelength=1.6, bbox_to_anchor=(0.0, 1.0),
    )
    leg.get_frame().set_linewidth(0.8)

    c = views["challenger"]
    h = views["hf1"]
    notes = [
        "Notes",
        "",
        "Each row fits its own UMAP",
        "on that row's Recovery/Harm",
        "endpoints (same as separate",
        "single-row figures).",
        "",
        "Arrows: baseline ● → target ○.",
        "",
        "HF1 target:",
        "  KEEP = baseline embed",
        "  REPLACE = challenger.",
    ]
    if include_displacement:
        notes += [
            "",
            "Violin y-limits shared across",
            "rows for fair ‖Δ‖₂ compare.",
        ]
    notes += [
        "",
        "Counts (rec / harm / nc)",
        f"  Challenger  {c['n_rec']} / {c['n_harm']} / {c['n_nc']}",
        f"  HF1         {h['n_rec']} / {h['n_harm']} / {h['n_nc']}",
    ]
    ax_side.text(
        0.0, 0.72, "\n".join(notes),
        transform=ax_side.transAxes, ha="left", va="top",
        fontsize=7.5, color="#444444", linespacing=1.40,
    )

    fig.suptitle(
        "Solution embedding shifts  ·  Always-on challenger vs HintFlow_one  ·  "
        f"BGE-large · UMAP  ·  IID 1809 @{budget}",
        fontsize=10.5, y=0.97, color="#222222",
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {out_pdf}", flush=True)
    print(f"wrote {out_png}", flush=True)


def plot_single(
    target: str,
    rows: list[dict],
    base_e: np.ndarray,
    chal_e: np.ndarray,
    out_pdf: Path,
    out_png: Path,
    *,
    budget: str = "8k",
) -> None:
    """1×3 single-row export; UMAP fit on this target's flips only."""
    end_e, labels = resolve_target(target, rows, base_e, chal_e)
    base_xy, end_xy, xlim, ylim = fit_umap_for_one_view(
        base_e, end_e, labels, tag=target,
    )
    delta = np.linalg.norm(end_e - base_e, axis=1)

    _set_rc()
    fig, axes = plt.subplots(
        1, 3, figsize=(9.6, 3.2),
        gridspec_kw={"width_ratios": [1.15, 1.15, 1.0], "wspace": 0.22},
    )
    meta = row_meta(budget)[target]
    for j, (cls, color) in enumerate([("recovery", C_REC), ("harm", C_HARM)]):
        ax = axes[j]
        m = labels == cls
        _draw_arrows(
            ax, base_xy[m, 0], base_xy[m, 1], end_xy[m, 0], end_xy[m, 1], color,
        )
        n_cls = int(m.sum())
        ax.set_title(f"{cls.capitalize()} ($n$={n_cls})", color=color, pad=6)
        _style_umap(ax, xlim, ylim, xlabel=True, ylabel=("UMAP-2" if j == 0 else None))
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=C_BASE, markersize=4.5, label="baseline ●"),
                Line2D([0], [0], marker="o", color=color, markerfacecolor="none",
                       markeredgecolor=color, markersize=7, markeredgewidth=1.2,
                       label=meta["end_label"]),
            ],
            loc="upper right", fontsize=7, frameon=True, fancybox=False,
            edgecolor="#DDDDDD",
        )
    axes[2].set_title("Displacement magnitude", pad=6)
    _draw_violin(axes[2], labels, delta)
    fig.suptitle(
        f"{meta['row_label'].replace(chr(10), ' ')}  ·  BGE-large · UMAP  ·  IID 1809 @{budget}",
        fontsize=10, y=1.03,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_pdf}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--budget",
        choices=BUDGETS,
        default="8k",
        help="IID HintFlow_one eval budget (sets data/cache/output stems)",
    )
    ap.add_argument(
        "--target",
        choices=["both", "challenger", "hf1"],
        default="both",
        help="both = unified 2×3 figure (default)",
    )
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    ap.add_argument(
        "--umap-only",
        action="store_true",
        help="2×2 Recovery/Harm UMAP only (no displacement violin)",
    )
    args = ap.parse_args()

    data_default, cache_default = budget_paths(args.budget)
    data_path = args.data or data_default
    cache_path = args.cache or cache_default

    rows = load_rows(data_path)
    base_e, chal_e = embed_all(rows, cache_path)

    if args.target == "both":
        if args.umap_only:
            stem = f"fig_idist_{args.budget}_solution_umap_challenger_vs_hf1_distonly"
        else:
            stem = f"fig_idist_{args.budget}_solution_umap_challenger_vs_hf1"
        plot_combined(
            rows, base_e, chal_e,
            out_pdf=args.out_dir / f"{stem}.pdf",
            out_png=args.out_dir / f"{stem}.png",
            budget=args.budget,
            include_displacement=not args.umap_only,
        )
    else:
        stem = row_meta(args.budget)[args.target]["out_stem"]
        plot_single(
            args.target, rows, base_e, chal_e,
            out_pdf=args.out_dir / f"{stem}.pdf",
            out_png=args.out_dir / f"{stem}.png",
            budget=args.budget,
        )


if __name__ == "__main__":
    main()
