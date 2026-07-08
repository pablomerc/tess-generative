"""Render the latent-space UMAP figure for the paper from cached encoder embeddings.

Inputs (this dir):
  - umap_base_n4096_s314_data.npz       (UMAP embeddings for 4096 paired galaxies, seed=314)
  - umap_base_n4096_s314_metadata.json  (run metadata: ckpt path, params, distance ratios)

Outputs (this dir, one PDF + PNG per pair-seed):
  - umap_base_n4096_s314_science_{N_PAIRS}pairs_pseed{seed}.{pdf,png}

The pair-seed only changes which N galaxy pairs are highlighted; the underlying
UMAP embedding (the cloud of all 4096 points) is identical across seeds.

The cache was produced by:
  galaxy_images/galaxy_model/visualization_scripts/regenerate_umap_base.py
on checkpoint galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "umap_base_n4096_s314_data.npz"
META_PATH = HERE / "umap_base_n4096_s314_metadata.json"

N_PAIRS = 12
# Each pair-seed reshuffles which N_PAIRS galaxy pairs are highlighted; the
# underlying UMAP cloud (4096 points) is identical across seeds.
# 9999 is the picked one for the paper. To regenerate the earlier sweep, extend
# this list, e.g. [7, 42, 123, 314, 1234, 2025, 31415, 9999, 271828].
PAIR_SEEDS = [31415]
OUT_STEM_FMT = HERE / "umap_base_n4096_s314_science_{n}pairs_pseed{seed}"

MAX_Y_TICKS = 5

PAIR_LINE_WIDTH = 1.9
PAIR_LINE_ALPHA = 0.85
PAIR_LINE_STYLE = "-"

# --- Styling ----------------------------------------------------------------
FIG_SIZE = (20, 8)
DPI = 300
POINT_SIZE = 20
ALPHA_E1 = 0.55
ALPHA_E2 = 0.35

# Earlier palettes (kept here in case we want to revert):
#   5-color: "#70a845" green, "#b460bd" purple, "#4aac8d" teal, "#c85979" pink, "#b49041" gold
#   3-color (gold variant): "#70a845" green, "#b49041" gold, "#c85979" pink
#   gold dropped because it blended with the cream HSC background.
PALETTE = [
    "#e8c4a0",  # HSC
    "#8eb8e8",  # Legacy
    "#70a845",  # green
    "#9b7cc5",  # muted purple (slightly darker pastel)
    "#c85979",  # pink
]
COLOR_HSC = PALETTE[0]
COLOR_LEGACY = PALETTE[1]
# 3 colors x 4 shapes = 12 unique (color, marker) combos for N_PAIRS=12.
PAIR_COLORS = PALETTE[2:]
PAIR_MARKERS = ["x", "s", "o", "^"]
PAIR_MARKER_SIZE = 200
PAIR_LINEWIDTHS = 3

TITLE_FONTSIZE = 28
TITLE_PAD = 12
AXIS_FONTSIZE = 23
LEGEND_FONTSIZE = 21
TICK_FONTSIZE = 17
LEGEND_MARKER_SIZE = 8


class _PairsLegendHandle:
    pass


class _HandlerPairs(HandlerBase):
    def __init__(self, pair_markers, **kwargs):
        super().__init__(**kwargs)
        self.pair_markers = pair_markers

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        y_center = ydescent + height / 2
        n = len(self.pair_markers)
        for i, mk in enumerate(self.pair_markers):
            x = xdescent + width * (i + 0.5) / n
            artists.append(
                Line2D(
                    [x], [y_center], marker=mk, color="white", markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=1.2,
                    markersize=fontsize * 0.8, linestyle="None", transform=trans,
                )
            )
        return artists


def _draw_panel(ax, hsc_xy, legacy_xy, selected_indices, *, alpha, title, show_ylabel):
    ax.scatter(hsc_xy[:, 0], hsc_xy[:, 1], s=POINT_SIZE, alpha=alpha, c=COLOR_HSC, label="HSC")
    ax.scatter(legacy_xy[:, 0], legacy_xy[:, 1], s=POINT_SIZE, alpha=alpha, c=COLOR_LEGACY, label="Legacy")

    n_pair_colors = len(PAIR_COLORS)
    for i, idx in enumerate(selected_indices):
        color = PAIR_COLORS[i % n_pair_colors]
        marker = PAIR_MARKERS[i % len(PAIR_MARKERS)]
        lw = PAIR_LINEWIDTHS
        lw_outline = lw + 2 if marker == "x" else lw
        ax.plot(
            [hsc_xy[idx, 0], legacy_xy[idx, 0]],
            [hsc_xy[idx, 1], legacy_xy[idx, 1]],
            color=color, linewidth=PAIR_LINE_WIDTH, alpha=PAIR_LINE_ALPHA,
            linestyle=PAIR_LINE_STYLE, zorder=3, solid_capstyle="round",
        )
        if marker == "x":
            ax.scatter(hsc_xy[idx, 0], hsc_xy[idx, 1], marker=marker, s=PAIR_MARKER_SIZE,
                       c=["black"], linewidths=lw_outline, zorder=4)
            ax.scatter(legacy_xy[idx, 0], legacy_xy[idx, 1], marker=marker, s=PAIR_MARKER_SIZE,
                       c=["black"], linewidths=lw_outline, zorder=4)
        ax.scatter(hsc_xy[idx, 0], hsc_xy[idx, 1], marker=marker, s=PAIR_MARKER_SIZE,
                   c=[color], linewidths=lw, edgecolors="black", zorder=5)
        ax.scatter(legacy_xy[idx, 0], legacy_xy[idx, 1], marker=marker, s=PAIR_MARKER_SIZE,
                   c=[color], linewidths=lw, edgecolors="black", zorder=5)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=TITLE_PAD)
    ax.set_xlabel("UMAP Component 1", fontsize=AXIS_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("UMAP Component 2", fontsize=AXIS_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_locator(plt.MaxNLocator(MAX_Y_TICKS))

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_HSC,
               markeredgecolor="black", markersize=LEGEND_MARKER_SIZE, label="HSC"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_LEGACY,
               markeredgecolor="black", markersize=LEGEND_MARKER_SIZE, label="Legacy"),
        _PairsLegendHandle(),
    ]
    legend_labels = ["HSC", "Legacy", "Pairs"]
    ax.legend(
        handles=legend_handles, labels=legend_labels, fontsize=LEGEND_FONTSIZE,
        handlelength=4, handler_map={_PairsLegendHandle: _HandlerPairs(PAIR_MARKERS)},
    )


def _render_one(hsc_umap_1, legacy_umap_1, hsc_umap_2, legacy_umap_2, selected_indices, out_stem):
    import scienceplots  # noqa: F401  (registers styles on import)

    with plt.style.context(["science", "no-latex"]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
        _draw_panel(ax1, hsc_umap_1, legacy_umap_1, selected_indices,
                    alpha=ALPHA_E1, title="Physics Latent Space", show_ylabel=True)
        _draw_panel(ax2, hsc_umap_2, legacy_umap_2, selected_indices,
                    alpha=ALPHA_E2, title="Instrument Latent Space", show_ylabel=False)
        plt.tight_layout()

        pdf_path = out_stem.with_suffix(".pdf")
        png_path = out_stem.with_suffix(".png")
        plt.savefig(pdf_path)
        plt.savefig(png_path, dpi=DPI)
        plt.close()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    return pdf_path, png_path


def main():
    with np.load(DATA_PATH, allow_pickle=False) as z:
        hsc_umap_1 = z["hsc_umap_1"]
        legacy_umap_1 = z["legacy_umap_1"]
        hsc_umap_2 = z["hsc_umap_2"]
        legacy_umap_2 = z["legacy_umap_2"]

    with open(META_PATH) as f:
        meta = json.load(f)
    num_hsc = int(meta["num_hsc"])
    print(f"[make_figure] meta: tag={meta.get('tag')} num_anchors={meta.get('num_anchors')} "
          f"physics_ratio={meta.get('pair_distance_ratio_physics'):.3f} "
          f"instrument_ratio={meta.get('pair_distance_ratio_instrument'):.3f} "
          f"num_hsc={num_hsc}")

    outputs = []
    for seed in PAIR_SEEDS:
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(num_hsc, size=N_PAIRS, replace=False).astype(np.int64)
        out_stem = Path(str(OUT_STEM_FMT).format(n=N_PAIRS, seed=seed))
        outputs.append(_render_one(
            hsc_umap_1, legacy_umap_1, hsc_umap_2, legacy_umap_2,
            selected_indices, out_stem,
        ))
    return outputs


if __name__ == "__main__":
    main()
