"""Top-12 anomalies per model with HSC cModel SNR percentile labels.

Three columns: Ours (Physics) | AION-1 | Ours (Instrument). For each anomaly,
the title shows its rank within the model and its percentile in the
dataset-wide SNR distribution from hsc_noise_metrics.h5.

Run from galaxy_model/:
  python anomaly_detection/top_anomalies_snr_percentile.py \
    --method flow --out anomaly_detection/outputs/figures_compare/top12_snr_percentile_flow.png
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from paper_anomaly_figure import (
    _to_rgb,
    _load_hsc_images,
    _top_n_with_percentiles,
    SOURCES,
    OUTPUTS_DIR,
    NEIGHBORS_HDF5,
    BG_COLORS,
)

NOISE_METRICS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5"


def build_snr_lookup(n_neighbors):
    """Return (snr_by_raw, sorted_finite) for percentile lookups keyed by raw_index."""
    with h5py.File(NOISE_METRICS_HDF5, "r") as f:
        hdf5_row_idx = np.array(f["hdf5_row_idx"], dtype=np.int64)
        snr_neg_4band = np.array(f["snr_neg_4band"], dtype=np.float64)

    snr_pos = -snr_neg_4band

    snr_by_raw = np.full(n_neighbors, np.nan, dtype=np.float64)
    snr_by_raw[hdf5_row_idx] = snr_pos

    finite = np.isfinite(snr_pos)
    sorted_finite = np.sort(snr_pos[finite])

    n_total = len(snr_pos)
    n_finite = finite.sum()
    finite_vals = snr_pos[finite]
    print(
        f"SNR map: {n_total} rows in metrics file, {n_finite} finite "
        f"({100.0 * n_finite / n_total:.1f}%); "
        f"min={finite_vals.min():.3f} median={np.median(finite_vals):.3f} max={finite_vals.max():.3f}"
    )
    return snr_by_raw, sorted_finite


def percentile_of(value, sorted_finite):
    if not np.isfinite(value):
        return float("nan")
    rank = np.searchsorted(sorted_finite, value, side="left")
    return rank / len(sorted_finite) * 100.0


def make_figure(method, out_path, top_n=12, n_cols_per=4, img_size=1.35, group_gap=0.35):
    assert top_n % n_cols_per == 0, f"top_n={top_n} must be divisible by n_cols_per={n_cols_per}"
    n_rows = top_n // n_cols_per
    n_groups = len(SOURCES)

    sources_for_method = []
    for src in SOURCES:
        new_key = src["score_key"].rsplit("/", 1)[0] + f"/{method}"
        sources_for_method.append({**src, "score_key": new_key})

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        n_neighbors = f["images_hsc"].shape[0]
    snr_by_raw, sorted_finite = build_snr_lookup(n_neighbors)

    per_source = []
    with h5py.File(NEIGHBORS_HDF5, "r") as h5_data:
        for src in sources_for_method:
            scores_path = OUTPUTS_DIR / src["scores_file"]
            print(f"  {src['label']} [{src['score_key']}]: loading top {top_n} ...")
            top_raw, _ = _top_n_with_percentiles(scores_path, src["score_key"], top_n)
            hsc_imgs = _load_hsc_images(h5_data, top_raw)
            top_snr = snr_by_raw[top_raw]
            top_snr_pct = np.array([percentile_of(s, sorted_finite) for s in top_snr])
            for i in range(min(3, top_n)):
                snr_str = "n/a" if not np.isfinite(top_snr[i]) else f"{top_snr[i]:.2f}"
                pct_str = "n/a" if not np.isfinite(top_snr_pct[i]) else f"{top_snr_pct[i]:.1f}"
                print(f"    #{i+1}: raw_idx={top_raw[i]:>8d}  SNR={snr_str:>6s}  pct={pct_str}")
            per_source.append({
                "label": src["label"],
                "top_raw": top_raw,
                "hsc_imgs": hsc_imgs,
                "top_snr_pct": top_snr_pct,
            })

    fig_w = n_groups * n_cols_per * img_size + (n_groups - 1) * group_gap
    fig_h = n_rows * img_size + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    gs_left, gs_right = 0.005, 0.995
    wspace_frac = group_gap / (n_cols_per * img_size)
    outer = gridspec.GridSpec(
        1, n_groups, figure=fig,
        left=gs_left, right=gs_right,
        top=0.82, bottom=0.03,
        wspace=wspace_frac,
    )

    col_w_fig = (gs_right - gs_left) / (n_groups + wspace_frac * (n_groups - 1))
    gap = wspace_frac * col_w_fig
    pad_x = gap * 0.40

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.set_axis_off()
    bg_ax.patch.set_visible(False)

    for col_idx in range(n_groups):
        pos = outer[col_idx].get_position(fig)
        x0 = pos.x0 - (pad_x if col_idx > 0 else 0)
        x1 = pos.x1 + (pad_x if col_idx < n_groups - 1 else 0)
        bg_ax.axvspan(x0, x1, facecolor=BG_COLORS[col_idx], linewidth=0)

    for col_idx, data in enumerate(per_source):
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols_per,
            subplot_spec=outer[col_idx],
            hspace=0.55, wspace=0.04,
        )

        for i in range(top_n):
            r, c = divmod(i, n_cols_per)
            ax = fig.add_subplot(inner[r, c])
            ax.set_facecolor("none")
            ax.set_zorder(2)
            ax.imshow(_to_rgb(data["hsc_imgs"][i]))
            ax.axis("off")

            pct = data["top_snr_pct"][i]
            pct_str = "n/a" if not np.isfinite(pct) else f"p{pct:.0f}"
            ax.set_title(
                f"#{i+1}\nSNR {pct_str}",
                fontsize=9, pad=2,
                color="#222222", fontweight="bold",
                linespacing=1.05,
            )

        pos = outer[col_idx].get_position(fig)
        x_mid = (pos.x0 + pos.x1) / 2
        fig.text(
            x_mid, 0.88,
            data["label"],
            ha="center", va="bottom",
            fontsize=15, fontweight="bold",
        )

    fig.suptitle(
        f"Top-{top_n} anomalies — detector: {method}  (label = rank / SNR percentile in full dataset)",
        fontsize=11, y=0.98,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["flow", "iforest"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()
    make_figure(args.method, args.out, top_n=args.top_n)


if __name__ == "__main__":
    main()
