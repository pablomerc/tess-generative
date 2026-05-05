"""Combined UMAP + anomaly-detection paper figure.

Top band:    1x3 UMAP panels — Physics (ours) | Instrument (ours) | AION
Bottom band: 1x3 groups of 4x3 HSC anomaly stamps — same column order
             (AION moved to the right to match the UMAP arrangement).

Reads the existing cached npz files; does NOT recompute anything.

Outputs:
    combined_umap_anomaly.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"

# All inputs are local — no upstream paths required.
OURS_NPZ = CACHE / "ours_umap_base_n4096_s314.npz"
AION_NPZ = CACHE / "aion_umap_neighbors_efficient_n4096.npz"

# Anomaly column order = same as UMAP top band: Physics | Instrument | AION
ANOM_SOURCES = [
    {"label": "Ours (Physics)",     "cache": CACHE / "anomaly_ours_physics_top12.npz"},
    {"label": "Ours (Instrument)",  "cache": CACHE / "anomaly_ours_instrument_top12.npz"},
    {"label": "AION-1",             "cache": CACHE / "anomaly_aion_top12.npz"},
]

# Match plot_umap_from_file.py palette/style
COLOR_HSC = "#e8c4a0"
COLOR_LEGACY = "#8eb8e8"
PAIR_COLORS = ["#70a845", "#b460bd", "#4aac8d", "#c85979", "#b49041"]
PAIR_MARKERS = ["x", "s", "o", "^"]


# ─── Pair legend handler (copied from plot_umap_from_file.py) ────────────────
class _PairsHandle:
    pass


class _HandlerPairs(HandlerBase):
    def __init__(self, markers, **kw):
        super().__init__(**kw)
        self.markers = markers

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        artists = []
        yc = ydescent + height / 2
        n = len(self.markers)
        for i, mk in enumerate(self.markers):
            x = xdescent + width * (i + 0.5) / n
            artists.append(Line2D([x], [yc], marker=mk, color="white",
                                  markerfacecolor="white", markeredgecolor="black",
                                  markeredgewidth=1.2, markersize=fontsize * 0.8,
                                  linestyle="None", transform=trans))
        return artists


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _load_anom(src):
    with np.load(src["cache"]) as d:
        return {"hsc_imgs": d["hsc_imgs"], "top_pcts": d["top_pcts"]}


def _draw_umap_panel(ax, hsc_xy, legacy_xy, selected_indices, title,
                     point_size=20, alpha=0.55,
                     pair_marker_size=200, pair_linewidths=3):
    ax.scatter(hsc_xy[:, 0], hsc_xy[:, 1], s=point_size, c=COLOR_HSC, alpha=alpha)
    ax.scatter(legacy_xy[:, 0], legacy_xy[:, 1], s=point_size, c=COLOR_LEGACY, alpha=alpha)
    if selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = PAIR_COLORS[i % len(PAIR_COLORS)]
            marker = PAIR_MARKERS[i % len(PAIR_MARKERS)]
            lw_outline = pair_linewidths + 2 if marker == "x" else pair_linewidths
            if marker == "x":
                for xy in (hsc_xy, legacy_xy):
                    ax.scatter([xy[idx, 0]], [xy[idx, 1]], marker=marker,
                               s=pair_marker_size, c=["black"],
                               linewidths=lw_outline, zorder=4, alpha=1.0)
            for xy in (hsc_xy, legacy_xy):
                ax.scatter([xy[idx, 0]], [xy[idx, 1]], marker=marker,
                           s=pair_marker_size, c=[color], linewidths=pair_linewidths,
                           zorder=5, edgecolors="black", alpha=1.0)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("UMAP Component 1", fontsize=14)
    ax.set_ylabel("UMAP Component 2", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_box_aspect(1.0)  # force square axes box regardless of allocated cell shape
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_HSC,
               markeredgecolor="black", markersize=8, label="HSC"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_LEGACY,
               markeredgecolor="black", markersize=8, label="Legacy"),
        _PairsHandle(),
    ]
    ax.legend(handles=handles, labels=["HSC", "Legacy", "Pairs"], fontsize=12,
              handlelength=4,
              handler_map={_PairsHandle: _HandlerPairs(PAIR_MARKERS)})


# ─── Builder ────────────────────────────────────────────────────────────────
def build_figure(top_n=12, n_cols_per=3,
                 stamp_size=1.55, group_gap=0.30,
                 hspace_stamps=0.04, wspace_stamps=0.04,
                 umap_height=None, header_pad=0.55):
    n_rows_stamps = top_n // n_cols_per          # 4
    n_groups = len(ANOM_SOURCES)                  # 3

    fig_w = n_groups * n_cols_per * stamp_size + (n_groups - 1) * group_gap

    # Make the UMAP top band tall enough that each of the 3 panels can be
    # square: per-panel allocated width ≈ fig_w / (n_groups + (n_groups-1)*wspace),
    # plus a bit of padding for the title and axis labels.
    umap_wspace = 0.18
    per_panel_w = fig_w * 0.99 / (3 + 2 * umap_wspace)
    if umap_height is None:
        umap_height = per_panel_w + 0.95   # extra room for title + axis labels

    fig_h_stamps = n_rows_stamps * stamp_size + header_pad
    fig_h = umap_height + fig_h_stamps
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Outer 2-row vertical split: UMAP on top, anomaly stamps on bottom.
    height_ratios = [umap_height, fig_h_stamps]
    outer_v = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=height_ratios,
        left=0.005, right=0.995, top=0.985, bottom=0.005,
        hspace=0.10,
    )

    # ── Top band: 1x3 UMAP panels ──
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_v[0], wspace=umap_wspace,
    )

    # Load UMAP coords
    with np.load(OURS_NPZ, allow_pickle=False) as z:
        hsc_u1 = z["hsc_umap_1"]; leg_u1 = z["legacy_umap_1"]
        hsc_u2 = z["hsc_umap_2"]; leg_u2 = z["legacy_umap_2"]
        sel = z["selected_indices"]
    sel = sel if sel.size > 0 else None
    with np.load(AION_NPZ, allow_pickle=False) as z:
        hsc_ua = z["hsc_umap"]; leg_ua = z["legacy_umap"]

    ax_phys = fig.add_subplot(top_gs[0, 0])
    ax_ins = fig.add_subplot(top_gs[0, 1])
    ax_aion = fig.add_subplot(top_gs[0, 2])

    _draw_umap_panel(ax_phys, hsc_u1, leg_u1, sel, "Physics Latent Space",
                     alpha=0.55)
    _draw_umap_panel(ax_ins, hsc_u2, leg_u2, sel, "Instrument Latent Space",
                     alpha=0.35)
    _draw_umap_panel(ax_aion, hsc_ua, leg_ua, sel, "AION Latent Space",
                     alpha=0.55)

    # ── Bottom band: 1x3 stamp groups, with dashed inter-group separators ──
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_groups, subplot_spec=outer_v[1],
        wspace=group_gap / (n_cols_per * stamp_size),
    )

    # Capture bottom region for the dashed separators (in figure coords).
    bot_pos = outer_v[1].get_position(fig)
    sep_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    sep_ax.set_xlim(0, 1); sep_ax.set_ylim(0, 1)
    sep_ax.set_axis_off(); sep_ax.patch.set_visible(False)

    title_y_offset = 0.12 * (header_pad / fig_h_stamps)

    for col_idx, src in enumerate(ANOM_SOURCES):
        data = _load_anom(src)
        hsc_imgs = data["hsc_imgs"]
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_stamps, n_cols_per,
            subplot_spec=bottom_gs[0, col_idx],
            hspace=hspace_stamps, wspace=wspace_stamps,
        )
        for i in range(top_n):
            r, c = divmod(i, n_cols_per)
            ax = fig.add_subplot(inner[r, c])
            ax.set_facecolor("none")
            ax.set_zorder(2)
            ax.imshow(_to_rgb(hsc_imgs[i]), interpolation="nearest")
            ax.axis("off")
            ax.text(0.5, 0.04, f"#{i+1}",
                    transform=ax.transAxes,
                    fontsize=10, fontweight="bold", color="black",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.2",
                              facecolor="white", alpha=0.7, linewidth=0))

        # Group header at the top of the bottom band.
        gpos = bottom_gs[0, col_idx].get_position(fig)
        x_mid = (gpos.x0 + gpos.x1) / 2
        # Place header just above the stamp grid.
        fig.text(x_mid, gpos.y1 + title_y_offset,
                 src["label"], ha="center", va="bottom",
                 fontsize=17, fontweight="bold")

    # Inter-group dashed separators that span only the bottom band.
    for col_idx in range(n_groups - 1):
        p_left = bottom_gs[0, col_idx].get_position(fig)
        p_right = bottom_gs[0, col_idx + 1].get_position(fig)
        x_sep = (p_left.x1 + p_right.x0) / 2
        sep_ax.plot([x_sep, x_sep],
                    [bot_pos.y0, bot_pos.y1],
                    color="#888888", linestyle="--", linewidth=1.2,
                    dashes=(6, 4), solid_capstyle="butt")

    return fig


def _save(fig, stem):
    out_png = HERE / f"{stem}.png"
    out_pdf = HERE / f"{stem}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")


def main():
    print("Rendering scienceplots version")
    try:
        import scienceplots  # noqa: F401
        styles = ["science", "no-latex"]
    except ImportError:
        print("  scienceplots not installed; using default style")
        styles = []
    with plt.style.context(styles) if styles else plt.style.context("default"):
        fig = build_figure()
        _save(fig, "combined_umap_anomaly_science")
        plt.close(fig)


if __name__ == "__main__":
    main()
