"""Combined data-driven noise model figure -- SciencePlots styling.

Same content + layout as `make_figure_plain.py`, but applies the SciencePlots
"science" style (with the "no-latex" backend so it works without a system
LaTeX install). Times-style serif fonts are still requested at a uniform
larger size, which composes cleanly on top of the SciencePlots stylesheet.

Top row    -- SNR traversal for target stats_idx=53601 (k10, vis=indep), built
              from `snr_traversal_53601_k10.npz` (sliced from
              snr_traversal_arrays.h5).
Bottom row -- 4 (Original, Corrected) pairs from the artifact-correction
              experiment, built from `artifact_correction.npz` (sliced from
              the exp_E repeat run tensors).

Outputs: datadriven_noise_model_figure_scienceplots.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  -- registers the "science" style with mpl

HERE = Path(__file__).resolve().parent

# SciencePlots base style + no-latex (we don't depend on a system LaTeX install).
plt.style.use(["science", "no-latex"])

# Override on top: Times New Roman where available, uniform larger font size.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
FS = 16          # text inside the panels (inner & bottom labels)
FS_OUT = 20      # text outside the panels (headers, arrow label, pair titles)
FS_RECON = 20    # "Reconstructions:" prefix (matches FS_OUT for visual consistency)
FS_FORMULA = 26  # the math formula sitting next to "Reconstructions:"

# --- top-row palette (paper style) ---
COLOR_TRUTH  = "#d0f0c0"
COLOR_INPUT  = "#d9d9d9"
COLOR_OUTPUT = "#d1efff"
Z_GRAY, Z_BLUE, Z_GREEN = -3, -2, -1


def vis_indep(img_chw):
    rgb = img_chw[:3].astype(np.float32).copy()
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def to_rgb_p1_99(img_chw):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, 1, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, 99, axis=(1, 2), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1).transpose(1, 2, 0)


def add_inner_label(ax, text):
    ax.text(0.5, 0.96, text, transform=ax.transAxes,
            fontsize=FS, fontweight="bold", color="black",
            verticalalignment="top", horizontalalignment="center",
            bbox=dict(boxstyle="square,pad=0.2",
                      facecolor="white", alpha=0.7, linewidth=0))


def add_bottom_label(ax, text):
    ax.text(0.5, 0.04, text, transform=ax.transAxes,
            fontsize=FS, fontweight="bold", color="white",
            verticalalignment="bottom", horizontalalignment="center", alpha=0.9)


def add_highlight(ax, color, zorder_bg):
    rect = patches.Rectangle(
        (-0.05, -0.05), 1.1, 1.1, transform=ax.transAxes,
        facecolor=color, edgecolor="none", zorder=zorder_bg, clip_on=False,
    )
    ax.add_patch(rect)


def draw_header(fig, ax_lo, ax_hi, y_top, label, fontsize=FS_OUT, line_width=1.5):
    bb_lo = ax_lo.get_position()
    bb_hi = ax_hi.get_position()
    fig.text((bb_lo.x0 + bb_hi.x1) / 2.0, y_top + 0.012, label,
             ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    fig.add_artist(plt.Line2D(
        [bb_lo.x0, bb_hi.x1], [y_top, y_top],
        transform=fig.transFigure, color="black", linewidth=line_width,
    ))


def draw_header_split(fig, ax_lo, ax_hi, y_top, prefix, formula,
                      prefix_fs=FS_RECON, formula_fs=FS_FORMULA, line_width=1.5):
    """Header with two text segments at different sizes, centered as a group."""
    bb_lo = ax_lo.get_position()
    bb_hi = ax_hi.get_position()
    x_mid = (bb_lo.x0 + bb_hi.x1) / 2.0
    y_text = y_top + 0.012

    t_pre = fig.text(0, y_text, prefix, fontsize=prefix_fs, fontweight="bold",
                     ha="left", va="bottom")
    t_for = fig.text(0, y_text, formula, fontsize=formula_fs, fontweight="bold",
                     ha="left", va="bottom")
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    bb_pre = t_pre.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(inv)
    bb_for = t_for.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(inv)
    gap = 0.005
    total_w = bb_pre.width + gap + bb_for.width
    x_start = x_mid - total_w / 2.0
    t_pre.set_position((x_start, y_text))
    t_for.set_position((x_start + bb_pre.width + gap, y_text))

    fig.add_artist(plt.Line2D(
        [bb_lo.x0, bb_hi.x1], [y_top, y_top],
        transform=fig.transFigure, color="black", linewidth=line_width,
    ))


def main():
    snr = np.load(HERE / "snr_traversal_53601_k10.npz")
    ac = np.load(HERE / "artifact_correction.npz")

    target_hsc    = snr["target_hsc"]
    target_legacy = snr["target_legacy"]
    generated     = snr["generated"]
    pcts          = snr["percentiles"]
    snr_avg       = snr["snr_avgs"]
    target_snr    = float(snr["target_snr"])
    target_psf    = float(snr["target_psf"])
    n_recon = generated.shape[0]
    n_top = 2 + n_recon

    targets = ac["ins_hsc"]
    recons  = ac["recons_mean"]
    n_pairs = len(targets)

    fig_w = 18.5
    fig_h = 7.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    top_panels_top    = 0.92
    top_panels_bottom = 0.55
    sep_y             = 0.46
    bot_panels_top    = 0.42
    bot_panels_bottom = 0.04

    # ---- TOP: SNR traversal ----
    gs_top = gridspec.GridSpec(
        1, n_top, figure=fig,
        left=0.02, right=0.99,
        top=top_panels_top, bottom=top_panels_bottom,
        wspace=0.06,
    )
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(n_top)]

    axes_top[0].imshow(vis_indep(target_hsc)); axes_top[0].set_axis_off()
    add_inner_label(axes_top[0], "HSC (target)")
    add_bottom_label(axes_top[0], f"SNR={target_snr:.0f}")
    add_highlight(axes_top[0], COLOR_TRUTH, Z_GREEN)

    axes_top[1].imshow(vis_indep(target_legacy)); axes_top[1].set_axis_off()
    add_inner_label(axes_top[1], "Legacy (input)")
    add_bottom_label(axes_top[1], "Same galaxy")
    add_highlight(axes_top[1], COLOR_INPUT, Z_GRAY)

    order = np.argsort(snr_avg)
    for col, k in enumerate(order):
        ax = axes_top[2 + col]
        ax.imshow(vis_indep(generated[k])); ax.set_axis_off()
        p_pos = 100 - float(pcts[k])
        add_inner_label(ax, f"SNR p{p_pos:.0f}")
        add_bottom_label(ax, f"avg SNR={snr_avg[k]:.0f}")
        add_highlight(ax, COLOR_OUTPUT, Z_BLUE)

    y_header = top_panels_top + 0.005
    draw_header(fig, axes_top[0], axes_top[0], y_header, "Ground truth")
    draw_header(fig, axes_top[1], axes_top[1], y_header, "Model input")
    draw_header_split(
        fig, axes_top[2], axes_top[2 + n_recon - 1], y_header,
        "Reconstructions:",
        r"$p(x_{\mathrm{HSC}} \mid z_{\mathrm{phy}}, z_{\mathrm{ins}}^{\mathrm{SNR}})$",
    )

    bb_lo = axes_top[2].get_position()
    bb_hi = axes_top[-1].get_position()
    y_arrow = bb_lo.y0 - 0.025
    x0 = bb_lo.x0 + 0.005
    x1 = bb_hi.x1 - 0.005
    fig.add_artist(patches.FancyArrowPatch(
        (x0, y_arrow), (x1, y_arrow), transform=fig.transFigure,
        arrowstyle="->,head_width=8,head_length=12",
        mutation_scale=1.0, linewidth=3.0, color="black",
    ))
    fig.text((x0 + x1) / 2.0, y_arrow - 0.025,
             "SNR low  $\\rightarrow$  SNR high",
             ha="center", va="top", fontsize=FS_OUT, fontweight="bold")

    fig.add_artist(plt.Line2D(
        [0.02, 0.99], [sep_y, sep_y],
        transform=fig.transFigure, color="0.6", linewidth=1.5,
    ))

    # ---- BOTTOM: artifact correction (4 pairs), spanning the full top width ----
    gs_bot_outer = gridspec.GridSpec(
        1, n_pairs, figure=fig,
        left=0.02, right=0.99,
        top=bot_panels_top, bottom=bot_panels_bottom,
        wspace=0.14,
    )
    pair_outer_positions = []
    for j in range(n_pairs):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_bot_outer[0, j], wspace=0.06,
        )
        ax_o = fig.add_subplot(inner[0, 0])
        ax_o.imshow(to_rgb_p1_99(targets[j])); ax_o.axis("off")
        ax_o.set_title(f"Original {j + 1}", fontsize=FS_OUT, fontweight="bold", pad=6)

        ax_c = fig.add_subplot(inner[0, 1])
        ax_c.imshow(to_rgb_p1_99(recons[j])); ax_c.axis("off")
        ax_c.set_title(f"Corrected {j + 1}", fontsize=FS_OUT, fontweight="bold", pad=6)
        pair_outer_positions.append(gs_bot_outer[0, j].get_position(fig))

    for j in range(n_pairs - 1):
        x_mid = (pair_outer_positions[j].x1 + pair_outer_positions[j + 1].x0) / 2.0
        fig.add_artist(plt.Line2D(
            [x_mid, x_mid],
            [pair_outer_positions[j].y0, pair_outer_positions[j].y1],
            transform=fig.transFigure, color="0.7", linewidth=0.8,
        ))

    out_pdf = HERE / "datadriven_noise_model_figure_scienceplots.pdf"
    out_png = HERE / "datadriven_noise_model_figure_scienceplots.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
