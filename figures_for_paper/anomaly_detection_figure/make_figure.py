"""Render the 3-column anomaly-detection paper figure from the bundled cache.

Run from this directory:
    python make_figure.py

Reads `_cache/anomaly_scores_*__*__top12.npz` (one per source) and writes:
    anomaly_detection_figure.{png,pdf}            — plain matplotlib
    anomaly_detection_figure_science.{png,pdf}    — scienceplots styling

Layout (per source group):
    4 rows × 3 cols of HSC stamps (12 top deduped flow anomalies).
    Group header at the top, gray dashed vertical separator between groups.
    Rank label "#i" rendered inside each stamp at the bottom, in a white
    rounded box (style copied from the top-row labels in
    `figures_for_paper/datadriven_noise_model_figure`).

Inputs come from `build_cache.py`; this script does not touch raw HDF5s.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "_cache"

# Same SOURCES as build_cache.py — order matters (column order).
SOURCES = [
    {
        "label": "Ours (Physics)",
        "scores_file": "anomaly_scores_ours_367k.h5",
        "score_key": "ours/hsc_mean/flow",
    },
    {
        "label": "AION-1",
        "scores_file": "anomaly_scores_aion_367k.h5",
        "score_key": "aion/hsc_mean_pca64/flow",
    },
    {
        "label": "Ours (Instrument)",
        "scores_file": "anomaly_scores_ins_367k.h5",
        "score_key": "ours/hsc_flat/flow",
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _cache_path(src, top_n):
    stem = Path(src["scores_file"]).stem
    key = src["score_key"].replace("/", "_")
    return CACHE_DIR / f"{stem}__{key}__top{top_n}.npz"


def _load_source(src, top_n):
    cp = _cache_path(src, top_n)
    if not cp.exists():
        raise FileNotFoundError(
            f"Missing cache file: {cp}.\nRun `python build_cache.py` first."
        )
    with np.load(cp) as d:
        return {
            "top_raw":  d["top_raw"],
            "top_pcts": d["top_pcts"],
            "hsc_imgs": d["hsc_imgs"],
        }


# ─── Figure builder ───────────────────────────────────────────────────────────

def build_figure(top_n=12, n_cols_per=3,
                 img_size=1.30, group_gap=0.30,
                 hspace=0.04, wspace=0.04, header_pad=0.32):
    """Construct the figure (style-agnostic). Caller controls plt.style."""
    n_rows = top_n // n_cols_per
    n_groups = len(SOURCES)

    fig_w = n_groups * n_cols_per * img_size + (n_groups - 1) * group_gap
    fig_h = n_rows * img_size + header_pad
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    gs_left, gs_right = 0.005, 0.995
    wspace_frac = group_gap / (n_cols_per * img_size)
    top_frac = 1.0 - (header_pad * 0.75) / fig_h
    bottom_frac = 0.005
    outer = gridspec.GridSpec(
        1, n_groups, figure=fig,
        left=gs_left, right=gs_right,
        top=top_frac, bottom=bottom_frac,
        wspace=wspace_frac,
    )

    # Full-figure overlay axis to draw inter-group dashed separators.
    sep_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    sep_ax.set_xlim(0, 1); sep_ax.set_ylim(0, 1)
    sep_ax.set_axis_off(); sep_ax.patch.set_visible(False)
    for col_idx in range(n_groups - 1):
        p_left = outer[col_idx].get_position(fig)
        p_right = outer[col_idx + 1].get_position(fig)
        x_sep = (p_left.x1 + p_right.x0) / 2
        sep_ax.plot(
            [x_sep, x_sep],
            [bottom_frac, top_frac],
            color="#888888", linestyle="--", linewidth=1.2,
            dashes=(6, 4), solid_capstyle="butt",
        )

    title_y = 1.0 - 0.12 * (header_pad / fig_h)

    for col_idx, src in enumerate(SOURCES):
        data = _load_source(src, top_n)
        hsc_imgs = data["hsc_imgs"]
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols_per,
            subplot_spec=outer[col_idx],
            hspace=hspace, wspace=wspace,
        )
        for i in range(top_n):
            r, c = divmod(i, n_cols_per)
            ax = fig.add_subplot(inner[r, c])
            ax.set_facecolor("none")
            ax.set_zorder(2)
            ax.imshow(_to_rgb(hsc_imgs[i]), interpolation="nearest")
            ax.axis("off")
            # Rank label at the bottom, styled to match the top-row inner-label
            # convention from datadriven_noise_model_figure (alpha=0.7, white box).
            ax.text(
                0.5, 0.04, f"#{i+1}",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold", color="black",
                verticalalignment="bottom", horizontalalignment="center",
                bbox=dict(boxstyle="square,pad=0.2",
                          facecolor="white", alpha=0.7, linewidth=0),
            )

        pos = outer[col_idx].get_position(fig)
        x_mid = (pos.x0 + pos.x1) / 2
        fig.text(
            x_mid, title_y,
            src["label"],
            ha="center", va="top",
            fontsize=13, fontweight="bold",
        )

    return fig


def _save(fig, stem):
    out_png = HERE / f"{stem}.png"
    out_pdf = HERE / f"{stem}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")


def main():
    # Plain version
    print("Rendering plain version")
    fig = build_figure()
    _save(fig, "anomaly_detection_figure")
    plt.close(fig)

    # Scienceplots version (paper style)
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        print("scienceplots not installed; skipping science version. "
              "Install with `pip install scienceplots`.")
        return
    print("Rendering scienceplots version")
    with plt.style.context(["science", "no-latex"]):
        fig = build_figure()
        _save(fig, "anomaly_detection_figure_science")
        plt.close(fig)


if __name__ == "__main__":
    main()
