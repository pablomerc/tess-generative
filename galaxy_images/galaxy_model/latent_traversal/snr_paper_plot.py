"""
Paper-style SNR-traversal figure.

Reads the intermediate arrays HDF5 produced by snr_traversal_full.py and emits
a clean per-target figure showing only:

    [HSC target]  [Legacy pair]  ||  [recon p95]  [recon p75]  [recon p50]  [recon p25]  [recon p5]

with a big SNR low->high arrow above the reconstruction columns and the SNR
percentile / numeric SNR labeled per recon. NO sameIns/conditioning column.

Visual style mirrors visualization_scripts/for_paper/replot_reconstruction.py:
colored backgrounds (green=Target, gray=Input, blue=Output), inner bold labels,
horizontal group headers above the columns.

Usage:
    python snr_paper_plot.py \
        --arrays-file snr_traversal_paper/snr_traversal_arrays.h5 \
        --target-idxs 53601,55427,83907,15780,33241 \
        --mode k10 \
        --vis indep \
        --output-dir snr_traversal_paper/paper_plots
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ---- paper-style colors (match replot_reconstruction.py) ----
COLOR_TARGET = "#d0f0c0"  # green
COLOR_INPUT  = "#d9d9d9"  # gray
COLOR_OUTPUT = "#d1efff"  # light blue

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


def draw_group_headers(fig, axes_row, n_recon):
    """Headers above: 'Inputs' (HSC tgt + Legacy pair) and 'Reconstructions' (n_recon panels)."""
    y_offset_title = 0.012
    line_width = 1.5
    label_fontsize = 12

    # Inputs group: cols 0..1
    bb_in_lo = axes_row[0].get_position()
    bb_in_hi = axes_row[1].get_position()
    y_top = bb_in_lo.y1 + 0.012
    x_mid_in = (bb_in_lo.x0 + bb_in_hi.x1) / 2.0
    fig.text(x_mid_in, y_top + y_offset_title, "Inputs",
             ha="center", va="bottom", fontsize=label_fontsize, fontweight="bold")
    fig.add_artist(plt.Line2D(
        [bb_in_lo.x0, bb_in_hi.x1], [y_top, y_top],
        transform=fig.transFigure, color="black", linewidth=line_width,
    ))

    # Reconstructions group: cols 2..2+n_recon-1
    bb_out_lo = axes_row[2].get_position()
    bb_out_hi = axes_row[2 + n_recon - 1].get_position()
    x_mid_out = (bb_out_lo.x0 + bb_out_hi.x1) / 2.0
    fig.text(
        x_mid_out, y_top + y_offset_title,
        r"Reconstructions: $p(x_{\mathrm{HSC}} \mid z_{\mathrm{phy}}, z_{\mathrm{ins}}^{\mathrm{SNR}})$",
        ha="center", va="bottom", fontsize=label_fontsize, fontweight="bold",
    )
    fig.add_artist(plt.Line2D(
        [bb_out_lo.x0, bb_out_hi.x1], [y_top, y_top],
        transform=fig.transFigure, color="black", linewidth=line_width,
    ))


def draw_snr_arrow(fig, axes_row_recon):
    """Big arrow under the reconstructions row pointing left->right (low SNR -> high SNR)."""
    bb_lo = axes_row_recon[0].get_position()
    bb_hi = axes_row_recon[-1].get_position()

    # Arrow sits below the reconstructions
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
                output_path: Path):
    grp_name = f"target_{target_idx:05d}/{mode}"
    with h5py.File(arr_path, "r") as f:
        if grp_name not in f:
            raise KeyError(f"{grp_name} not in {arr_path}")
        g = f[grp_name]
        target_hsc    = np.asarray(g["target_hsc"], dtype=np.float32)     # (4,48,48)
        target_legacy = np.asarray(g["target_legacy"], dtype=np.float32)  # (4,48,48)
        generated     = np.asarray(g["generated"], dtype=np.float32)      # (n_levels,4,48,48)
        labels  = [s.decode("utf-8") if isinstance(s, bytes) else s
                   for s in np.asarray(g["buckets/labels"])]
        pcts    = np.asarray(g["buckets/percentiles"], dtype=np.float32)
        snr_avg = np.asarray(g["buckets/snr_avgs"], dtype=np.float32)
        target_snr = float(g.attrs["target_snr"])
        target_psf = float(g.attrs["target_psf"])

    n_recon = generated.shape[0]
    n_cols = 2 + n_recon

    # rowscale uses target HSC as reference
    if vis_method == "rowscale":
        rgb_ref = target_hsc[:3]
        vmin = rgb_ref.min(axis=(1, 2))
        vmax = rgb_ref.max(axis=(1, 2))
        apply = lambda im: vis_rowscale(im, vmin, vmax)
    elif vis_method == "indep":
        apply = vis_indep
    else:
        raise ValueError(vis_method)

    # figure
    panel_w = 2.2
    panel_h = 2.4
    header_pad_in = 0.55
    arrow_pad_in  = 0.85
    fig_w = panel_w * n_cols + 0.4
    fig_h = panel_h + header_pad_in + arrow_pad_in

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=True)
    axes = np.atleast_1d(axes)

    # Col 0: HSC target (green/Target)
    axes[0].imshow(apply(target_hsc)); axes[0].set_axis_off()
    add_inner_label(axes[0], "HSC target")
    add_bottom_label(axes[0], f"SNR={target_snr:.0f}  PSF={target_psf:.2f}\"")
    add_highlight(axes[0], COLOR_TARGET, Z_GREEN)

    # Col 1: Legacy pair (gray/Input)
    axes[1].imshow(apply(target_legacy)); axes[1].set_axis_off()
    add_inner_label(axes[1], "Legacy pair")
    add_bottom_label(axes[1], "same galaxy")
    add_highlight(axes[1], COLOR_INPUT, Z_GRAY)

    # Cols 2..: reconstructions ordered low SNR -> high SNR
    # SNR_LEVELS in snr_traversal_full.py is [p95(neg)=noisiest, p75, p50, p25, p5(neg)=cleanest]
    # so generated[0] = noisiest = lowest SNR -> already in low->high order. Confirm via snr_avg.
    order = np.argsort(snr_avg)  # low SNR first
    for col, k in enumerate(order):
        ax = axes[2 + col]
        ax.imshow(apply(generated[k])); ax.set_axis_off()
        # Label with neg-percentile (matches how SNR_LEVELS were defined)
        # neg-percentile p95 = noisy => it's the *low SNR* bucket;
        # show the equivalent SNR_pos percentile (100 - p_neg) for reader clarity.
        p_pos = 100 - float(pcts[k])
        add_inner_label(ax, f"SNR p{p_pos:.0f}")
        add_bottom_label(ax, f"avg SNR={snr_avg[k]:.0f}")
        add_highlight(ax, COLOR_OUTPUT, Z_BLUE)

    # spacing
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {output_path}")


# ---- main ----

def parse_csv_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays-file", type=Path, required=True,
                        help="Path to snr_traversal_arrays.h5 produced by snr_traversal_full.py")
    parser.add_argument("--target-idxs", type=parse_csv_int, required=True,
                        help="Comma-separated stats_idx values to plot")
    parser.add_argument("--mode", type=str, default="k10",
                        help="Mode group inside the arrays file (default k10)")
    parser.add_argument("--vis", type=str, default="indep", choices=["indep", "rowscale"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for tidx in args.target_idxs:
        out = args.output_dir / f"snr_paper_{tidx:05d}_{args.mode}_{args.vis}.png"
        plot_target(args.arrays_file, tidx, args.mode, args.vis, out)


if __name__ == "__main__":
    main()
