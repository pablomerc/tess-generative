"""
Paper-style SNR-traversal figure.

Reads the bundled `_cache/snr_traversal_arrays.h5` (produced upstream by
`galaxy_images/galaxy_model/latent_traversal/snr_traversal_full.py`) and emits
a clean per-target figure showing only:

    [HSC target]  [Legacy pair]  ||  [recon p_low SNR ... p_high SNR]

with a big SNR low->high arrow under the reconstruction columns and per-bucket
SNR labels. NO sameIns / conditioning column.

Style mirrors visualization_scripts/for_paper/replot_reconstruction.py
(green=Ground truth, gray=Model input, blue=Output highlights, inner bold
labels, horizontal group headers above each column group).

Defaults reproduce the bundled paper figures. Only numpy + matplotlib + h5py
are required.

Usage:
    cd figures_for_paper/snr_traversal_figure
    python paper_snr_traversal.py                      # all targets, both vis, PNG+PDF
    python paper_snr_traversal.py --target-idxs 53601
    python paper_snr_traversal.py --vis indep          # indep only
    python paper_snr_traversal.py --vis rowscale       # rowscale only
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ---- defaults: run from this directory and everything Just Works ----
HERE = Path(__file__).resolve().parent
DEFAULT_ARRAYS = (
    HERE.parents[1]
    / "galaxy_images/galaxy_model/latent_traversal/snr_traversal_paper_fixed_noise"
    / "snr_traversal_arrays.h5"
)
DEFAULT_OUTDIR = HERE
DEFAULT_TARGETS = [53601, 55427, 83907, 15780, 33241, 58095, 34953, 62309, 62218, 92028]

# ---- paper-style colors ----
COLOR_TRUTH  = "#d0f0c0"  # green  — ground truth (HSC target)
COLOR_INPUT  = "#d9d9d9"  # gray   — model input  (Legacy pair)
COLOR_OUTPUT = "#d1efff"  # light blue — reconstructions

Z_GRAY  = -3
Z_BLUE  = -2
Z_GREEN = -1


# ---- visualisation helpers ----

def vis_indep(img_chw):
    rgb = img_chw[:3].astype(np.float32).copy()
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def vis_rowscale(img_chw, vmin, vmax):
    rgb = img_chw[:3].astype(np.float32).copy()
    rgb = (rgb - vmin[:, None, None]) / (vmax[:, None, None] - vmin[:, None, None] + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


# ---- label / highlight helpers (paper style) ----

def add_inner_label(ax, text, fontsize=10):
    ax.text(
        0.5, 0.96, text,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", color="black",
        verticalalignment="top", horizontalalignment="center",
        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.7, linewidth=0),
    )


def add_bottom_label(ax, text, fontsize=8):
    ax.text(
        0.5, 0.04, text,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", color="white",
        verticalalignment="bottom", horizontalalignment="center",
        alpha=0.85,
    )


def add_highlight(ax, color, zorder_bg):
    rect = patches.Rectangle(
        (-0.05, -0.05), 1.1, 1.1,
        transform=ax.transAxes,
        facecolor=color, edgecolor="none", zorder=zorder_bg, clip_on=False,
    )
    ax.add_patch(rect)


def _draw_header(fig, ax_lo, ax_hi, y_top, label, fontsize=12, line_width=1.5):
    """Draw a label + underline spanning ax_lo..ax_hi at figure-coord y_top."""
    bb_lo = ax_lo.get_position()
    bb_hi = ax_hi.get_position()
    x_mid = (bb_lo.x0 + bb_hi.x1) / 2.0
    fig.text(x_mid, y_top + 0.012, label,
             ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    fig.add_artist(plt.Line2D(
        [bb_lo.x0, bb_hi.x1], [y_top, y_top],
        transform=fig.transFigure, color="black", linewidth=line_width,
    ))


def draw_group_headers(fig, axes, n_recon):
    """
    Three separate headers:
      col 0  → "Ground truth"
      col 1  → "Model input"
      cols 2+ → "Reconstructions: ..."
    All at the same y level.
    """
    y_top = axes[0].get_position().y1 + 0.012

    _draw_header(fig, axes[0], axes[0], y_top, "Ground truth")
    _draw_header(fig, axes[1], axes[1], y_top, "Model input")
    _draw_header(
        fig, axes[2], axes[2 + n_recon - 1], y_top,
        r"Reconstructions: $p(x_{\mathrm{HSC}} \mid z_{\mathrm{phy}}, z_{\mathrm{ins}}^{\mathrm{SNR}})$",
    )


def draw_snr_arrow(fig, axes_row_recon):
    bb_lo = axes_row_recon[0].get_position()
    bb_hi = axes_row_recon[-1].get_position()

    y = bb_lo.y0 - 0.045
    x0 = bb_lo.x0 + 0.005
    x1 = bb_hi.x1 - 0.005

    arrow = patches.FancyArrowPatch(
        (x0, y), (x1, y),
        transform=fig.transFigure,
        arrowstyle="->,head_width=8,head_length=12",
        mutation_scale=1.0,
        linewidth=3.0, color="black",
    )
    fig.add_artist(arrow)

    fig.text((x0 + x1) / 2.0, y - 0.028, "SNR low  →  SNR high",
             ha="center", va="top", fontsize=12, fontweight="bold")


# ---- core plot ----

def plot_target(arr_path: Path, target_idx: int, mode: str, vis_method: str,
                output_dir: Path, formats=("png", "pdf")):
    grp_name = f"target_{target_idx:05d}/{mode}"
    with h5py.File(arr_path, "r") as f:
        if grp_name not in f:
            raise KeyError(f"{grp_name} not in {arr_path}")
        g = f[grp_name]
        target_hsc    = np.asarray(g["target_hsc"], dtype=np.float32)
        target_legacy = np.asarray(g["target_legacy"], dtype=np.float32)
        generated     = np.asarray(g["generated"], dtype=np.float32)
        pcts    = np.asarray(g["buckets/percentiles"], dtype=np.float32)
        snr_avg = np.asarray(g["buckets/snr_avgs"], dtype=np.float32)
        target_snr = float(g.attrs["target_snr"])
        target_psf = float(g.attrs["target_psf"])

    n_recon = generated.shape[0]
    n_cols = 2 + n_recon

    if vis_method == "rowscale":
        rgb_ref = target_hsc[:3]
        vmin = rgb_ref.min(axis=(1, 2))
        vmax = rgb_ref.max(axis=(1, 2))
        apply = lambda im: vis_rowscale(im, vmin, vmax)
    elif vis_method == "indep":
        apply = vis_indep
    else:
        raise ValueError(vis_method)

    panel_w = 2.2
    panel_h = 2.4
    header_pad_in = 0.55
    arrow_pad_in  = 0.85
    fig_w = panel_w * n_cols + 0.4
    fig_h = panel_h + header_pad_in + arrow_pad_in

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=True)
    axes = np.atleast_1d(axes)

    # Col 0: HSC target — ground truth (not a model input)
    axes[0].imshow(apply(target_hsc)); axes[0].set_axis_off()
    add_inner_label(axes[0], "HSC (target)")
    add_bottom_label(axes[0], f"SNR={target_snr:.0f}  PSF={target_psf:.2f}\"")
    add_highlight(axes[0], COLOR_TRUTH, Z_GREEN)

    # Col 1: Legacy pair — the actual model input providing z_phy
    axes[1].imshow(apply(target_legacy)); axes[1].set_axis_off()
    add_inner_label(axes[1], "Legacy (input)")
    add_bottom_label(axes[1], "same galaxy")
    add_highlight(axes[1], COLOR_INPUT, Z_GRAY)

    # Cols 2+: reconstructions sorted low→high SNR
    order = np.argsort(snr_avg)
    for col, k in enumerate(order):
        ax = axes[2 + col]
        ax.imshow(apply(generated[k])); ax.set_axis_off()
        p_pos = 100 - float(pcts[k])  # neg-pct → pos-pct (high = high SNR)
        add_inner_label(ax, f"SNR p{p_pos:.0f}")
        add_bottom_label(ax, f"avg SNR={snr_avg[k]:.0f}")
        add_highlight(ax, COLOR_OUTPUT, Z_BLUE)

    plt.subplots_adjust(
        wspace=0.05,
        left=0.02, right=0.99,
        bottom=arrow_pad_in / fig_h,
        top=1.0 - header_pad_in / fig_h,
    )

    draw_group_headers(fig, axes, n_recon)
    draw_snr_arrow(fig, axes[2:2 + n_recon])

    fig.suptitle(
        f"target stats_idx={target_idx}  |  mode={mode}  |  vis={vis_method}",
        fontsize=10, y=0.995,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"paper_snr_traversal_{target_idx:05d}_{mode}_{vis_method}"
    for ext in formats:
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved {out}")
    plt.close(fig)


# ---- main ----

def parse_csv_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays-file", type=Path, default=DEFAULT_ARRAYS,
                        help="snr_traversal_arrays.h5 produced by snr_traversal_full.py")
    parser.add_argument("--target-idxs", type=parse_csv_int, default=DEFAULT_TARGETS,
                        help="Comma-separated stats_idx values")
    parser.add_argument("--mode", type=str, default="k10")
    parser.add_argument("--vis", type=str, default="both",
                        choices=["indep", "rowscale", "both"],
                        help="Visualisation method; 'both' produces both variants (default)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--formats", type=str, default="png,pdf")
    args = parser.parse_args()

    if not args.arrays_file.exists():
        raise FileNotFoundError(
            f"Arrays file not found: {args.arrays_file}\n"
            "Re-run upstream pipeline (snr_traversal_full.py) first."
        )

    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    vis_methods = ["indep", "rowscale"] if args.vis == "both" else [args.vis]

    for tidx in args.target_idxs:
        for vis in vis_methods:
            plot_target(args.arrays_file, tidx, args.mode, vis, args.output_dir, formats)


if __name__ == "__main__":
    main()
