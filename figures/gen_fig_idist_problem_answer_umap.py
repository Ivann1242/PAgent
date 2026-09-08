#!/usr/bin/env python3
"""IID HintFlow: problem→answer UMAP (contrast to solution-shift figure).

Unlike fig_idist_8k_solution_umap_*:
  - old: baseline solution ● → target solution ○
  - this: problem ■ → answer ○  (green=recovery, red=harm)

Outputs new filenames only; does not overwrite the solution-shift figures.

  python figures/gen_fig_idist_problem_answer_umap.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "checkpoints/eval_hintflow_one_idist_1809_8k/hintflow_one.jsonl"
SOL_CACHE = ROOT / "figures/cache_idist_8k_solution_embeds.npz"
PROB_CACHE = ROOT / "figures/cache_idist_8k_problem_embeds.npz"
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
GPU = os.environ.get("EMBED_GPU", "0")
MAX_CHARS = 6000
SEED = 41

C_REC = "#009E73"
C_HARM = "#E41A1C"  # red, as requested
C_PROB = "#5B6B7A"
C_NC = "#999999"

ROW_META = {
    "challenger": {"row_label": "Always-on\nchallenger"},
    "hf1": {"row_label": "HintFlow_one\n(HF1)"},
}


def trunc(text: str) -> str:
    text = (text or "").strip()
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


def extract_math_problem(prompt: str) -> str:
    """Strip shared instruction wrapper so the problem point is the math stem."""
    text = (prompt or "").strip()
    # Drop leading instruction block before the first blank line after header.
    m = re.search(
        r"(?:step by step\.[^\n]*\n\n)(.+?)(?:\n\nRemember to put|\Z)",
        text,
        flags=re.S | re.I,
    )
    if m:
        return trunc(m.group(1).strip())
    # Fallback: drop first/last instructional paragraphs if present.
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) >= 3 and parts[0].lower().startswith("solve"):
        return trunc("\n\n".join(parts[1:-1]))
    return trunc(text)


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
            r["prob_text"] = extract_math_problem(r.get("problem") or "")
            r["base_sol"] = trunc(r["baseline"].get("solution") or "")
            r["chal_sol"] = trunc(r["challenger"].get("solution") or "")
            rows.append(r)
    return rows


def load_solution_embeds(rows: list[dict], cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if not cache.exists():
        raise FileNotFoundError(
            f"missing solution cache {cache}; run gen_fig_idist_solution_umap_arrows.py first"
        )
    z = np.load(cache, allow_pickle=True)
    if len(z["base"]) != len(rows):
        raise ValueError("solution cache length mismatch")
    if z["model_name"].item() != MODEL_NAME:
        raise ValueError(
            f"solution cache model {z['model_name'].item()} != {MODEL_NAME}"
        )
    print(f"loaded solution cache {cache}", flush=True)
    return z["base"], z["chal"]


def embed_problems(rows: list[dict], cache: Path) -> np.ndarray:
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if (
            z["model_name"].item() == MODEL_NAME
            and len(z["prob"]) == len(rows)
        ):
            print(f"loaded problem cache {cache}", flush=True)
            return z["prob"]

    from sentence_transformers import SentenceTransformer

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    print(f"embedding problems with {MODEL_NAME} on GPU {GPU}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    prob = model.encode(
        [r["prob_text"] for r in rows],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        prob=prob,
        model_name=np.array(MODEL_NAME),
        ids=np.array([r["id"] for r in rows], dtype=object),
        texts=np.array([r["prob_text"] for r in rows], dtype=object),
    )
    print(f"saved problem cache {cache}", flush=True)
    return prob


def resolve_answer_embeds(
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


def _draw_prob_ans_arrows(ax, xp, yp, xa, ya, color: str) -> None:
    if len(xp) == 0:
        return
    ax.quiver(
        xp, yp, xa - xp, ya - yp,
        angles="xy", scale_units="xy", scale=1.0,
        width=0.0030, headwidth=3.2, headlength=3.8, headaxislength=3.2,
        color=color, alpha=0.35, zorder=2,
    )
    ax.scatter(
        xp, yp, s=18, c=C_PROB, marker="s", alpha=0.70,
        linewidths=0, zorder=3,
    )
    ax.scatter(
        xa, ya, s=32, facecolors="none", edgecolors=color,
        linewidths=1.1, alpha=0.85, zorder=4,
    )


def _draw_violin(ax, labels: np.ndarray, delta: np.ndarray, ylim=None) -> None:
    order = ["recovery", "harm", "no_change"]
    colors = [C_REC, C_HARM, C_NC]
    counts = [int((labels == k).sum()) for k in order]
    data = [delta[labels == k] for k in order]
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
    ax.set_ylabel(r"$\|e_{\mathrm{ans}}-e_{\mathrm{prob}}\|_2$")
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


def fit_umap_problem_answer(
    prob_e: np.ndarray,
    ans_e: np.ndarray,
    labels: np.ndarray,
    *,
    tag: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    """UMAP on this row's Recovery/Harm problem+answer endpoints."""
    n = len(prob_e)
    flip = (labels == "recovery") | (labels == "harm")
    idx = np.where(flip)[0]
    pts = np.vstack([prob_e[idx], ans_e[idx]])
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.25, metric="cosine",
        random_state=SEED, n_jobs=1,
    )
    xy = reducer.fit_transform(pts)
    k = len(idx)
    prob_xy = np.full((n, 2), np.nan, dtype=np.float64)
    ans_xy = np.full((n, 2), np.nan, dtype=np.float64)
    prob_xy[idx] = xy[:k]
    ans_xy[idx] = xy[k:]
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    pad_x = 0.06 * (x1 - x0 + 1e-6)
    pad_y = 0.06 * (y1 - y0 + 1e-6)
    print(f"[{tag}] UMAP fit on {k} flips ({2 * k} endpoints: problem+answer)", flush=True)
    return prob_xy, ans_xy, (x0 - pad_x, x1 + pad_x), (y0 - pad_y, y1 + pad_y)


def plot_combined(
    rows: list[dict],
    prob_e: np.ndarray,
    base_e: np.ndarray,
    chal_e: np.ndarray,
    out_pdf: Path,
    out_png: Path,
) -> None:
    _set_rc()
    views = {}
    for name in ("challenger", "hf1"):
        ans_e, labels = resolve_answer_embeds(name, rows, base_e, chal_e)
        prob_xy, ans_xy, xlim, ylim = fit_umap_problem_answer(
            prob_e, ans_e, labels, tag=name,
        )
        views[name] = {
            "labels": labels,
            "prob_xy": prob_xy,
            "ans_xy": ans_xy,
            "xlim": xlim,
            "ylim": ylim,
            "delta": np.linalg.norm(ans_e - prob_e, axis=1),
            "n_rec": int((labels == "recovery").sum()),
            "n_harm": int((labels == "harm").sum()),
            "n_nc": int((labels == "no_change").sum()),
        }
        print(
            f"[{name}] recovery={views[name]['n_rec']} "
            f"harm={views[name]['n_harm']} no_change={views[name]['n_nc']}",
            flush=True,
        )

    dmax = max(float(v["delta"].max()) for v in views.values())
    vlim = (0.0, dmax * 1.12)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(11.2, 6.0))
    gs = GridSpec(
        2, 4, figure=fig,
        width_ratios=[1.15, 1.15, 1.0, 0.78],
        wspace=0.22, hspace=0.30,
        left=0.09, right=0.98, top=0.90, bottom=0.07,
    )
    axes = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j])
    ax_side = fig.add_subplot(gs[:, 3])
    ax_side.axis("off")

    for j, (title, color) in enumerate([
        ("Recovery", C_REC),
        ("Harm", C_HARM),
        ("Problem–answer distance", "#333333"),
    ]):
        axes[0, j].set_title(title, pad=8, color=color, fontsize=10.5)

    for i, name in enumerate(("challenger", "hf1")):
        v = views[name]
        labels = v["labels"]
        px, ax_xy = v["prob_xy"], v["ans_xy"]
        xlim, ylim = v["xlim"], v["ylim"]

        for j, (cls, color) in enumerate(
            [("recovery", C_REC), ("harm", C_HARM)]
        ):
            ax = axes[i, j]
            m = labels == cls
            _draw_prob_ans_arrows(
                ax,
                px[m, 0], px[m, 1],
                ax_xy[m, 0], ax_xy[m, 1],
                color,
            )
            ax.text(
                0.03, 0.03, f"$n$={int(m.sum())}",
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

        axv = axes[i, 2]
        _draw_violin(axv, labels, v["delta"], ylim=vlim)
        if i == 0:
            axv.set_xticklabels([])

        axes[i, 0].annotate(
            ROW_META[name]["row_label"],
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.28, 0.5), textcoords="axes fraction",
            ha="center", va="center", fontsize=9, color="#333333",
            rotation=90,
        )

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_PROB,
               markersize=7, label="problem ■"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#444444", markersize=8, label="answer ○"),
        Line2D([0], [0], color=C_REC, lw=2.0, label="Recovery arrow"),
        Line2D([0], [0], color=C_HARM, lw=2.0, label="Harm arrow"),
    ]
    leg = ax_side.legend(
        handles=legend_handles, loc="upper left", frameon=True,
        fontsize=8, borderpad=0.55, labelspacing=0.7,
        handlelength=1.6, bbox_to_anchor=(0.0, 1.0),
    )
    leg.get_frame().set_linewidth(0.8)

    c, h = views["challenger"], views["hf1"]
    side_text = (
        "Notes\n"
        "\n"
        "Contrast figure (new):\n"
        "  problem ■ → answer ○\n"
        "Old figure (unchanged):\n"
        "  baseline sol → target sol\n"
        "\n"
        "Each row fits its own UMAP\n"
        "on that row's Recovery/Harm\n"
        "problem+answer endpoints.\n"
        "\n"
        "HF1 answer:\n"
        "  KEEP = baseline sol\n"
        "  REPLACE = challenger\n"
        "\n"
        "Violin: ‖e_ans − e_prob‖₂\n"
        "in BGE space (not UMAP).\n"
        "\n"
        "Counts (rec / harm / nc)\n"
        f"  Challenger  {c['n_rec']} / {c['n_harm']} / {c['n_nc']}\n"
        f"  HF1         {h['n_rec']} / {h['n_harm']} / {h['n_nc']}"
    )
    ax_side.text(
        0.0, 0.72, side_text,
        transform=ax_side.transAxes, ha="left", va="top",
        fontsize=7.5, color="#444444", linespacing=1.40,
    )

    fig.suptitle(
        "Problem→answer embedding  ·  Always-on challenger vs HintFlow_one  ·  "
        "BGE-large · UMAP  ·  IID 1809 @8k",
        fontsize=10.5, y=0.97, color="#222222",
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}", flush=True)
    print(f"wrote {out_png}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--sol-cache", type=Path, default=SOL_CACHE)
    ap.add_argument("--prob-cache", type=Path, default=PROB_CACHE)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = ap.parse_args()

    rows = load_rows(args.data)
    print(f"loaded {len(rows)} rows from {args.data}", flush=True)
    # sanity on problem extraction
    n_short = sum(1 for r in rows if len(r["prob_text"]) < 20)
    print(f"problem extract: median_len="
          f"{int(np.median([len(r['prob_text']) for r in rows]))} "
          f"short(<20)={n_short}", flush=True)

    base_e, chal_e = load_solution_embeds(rows, args.sol_cache)
    prob_e = embed_problems(rows, args.prob_cache)
    plot_combined(
        rows, prob_e, base_e, chal_e,
        out_pdf=args.out_dir / "fig_idist_8k_problem_answer_umap_challenger_vs_hf1.pdf",
        out_png=args.out_dir / "fig_idist_8k_problem_answer_umap_challenger_vs_hf1.png",
    )


if __name__ == "__main__":
    main()
