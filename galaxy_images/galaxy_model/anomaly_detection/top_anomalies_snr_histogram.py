"""Histogram of top-N anomalies per model vs HSC cModel SNR percentile.

For each of Ours (Physics), AION-1, Ours (Instrument), takes the top-N anomalies
under a chosen detector and plots the distribution of their SNR percentiles
(0-100) in the dataset-wide HSC cModel SNR distribution from hsc_noise_metrics.h5.

Run from galaxy_model/:
  python anomaly_detection/top_anomalies_snr_histogram.py \
    --method flow --top-n 512 \
    --out anomaly_detection/outputs/figures_compare/snr_pct_hist_top512_flow.png
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_anomaly_figure import (
    _top_n_with_percentiles,
    SOURCES,
    OUTPUTS_DIR,
    NEIGHBORS_HDF5,
    BG_COLORS,
)

NOISE_METRICS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5"


def build_snr_lookup(n_neighbors):
    with h5py.File(NOISE_METRICS_HDF5, "r") as f:
        hdf5_row_idx = np.array(f["hdf5_row_idx"], dtype=np.int64)
        snr_neg_4band = np.array(f["snr_neg_4band"], dtype=np.float64)

    snr_pos = -snr_neg_4band
    snr_by_raw = np.full(n_neighbors, np.nan, dtype=np.float64)
    snr_by_raw[hdf5_row_idx] = snr_pos

    finite = np.isfinite(snr_pos)
    sorted_finite = np.sort(snr_pos[finite])

    print(
        f"SNR map: {len(snr_pos)} rows in metrics file, {finite.sum()} finite "
        f"({100.0 * finite.sum() / len(snr_pos):.1f}%)"
    )
    return snr_by_raw, sorted_finite


def percentile_array(values, sorted_finite):
    out = np.full_like(values, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.any():
        ranks = np.searchsorted(sorted_finite, values[finite], side="left")
        out[finite] = ranks / len(sorted_finite) * 100.0
    return out


def make_figure(method, top_n, out_path):
    sources_for_method = []
    for src in SOURCES:
        new_key = src["score_key"].rsplit("/", 1)[0] + f"/{method}"
        sources_for_method.append({**src, "score_key": new_key})

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        n_neighbors = f["images_hsc"].shape[0]
    snr_by_raw, sorted_finite = build_snr_lookup(n_neighbors)

    per_source = []
    for src in sources_for_method:
        scores_path = OUTPUTS_DIR / src["scores_file"]
        print(f"  {src['label']} [{src['score_key']}]: loading top {top_n} ...")
        top_raw, _ = _top_n_with_percentiles(scores_path, src["score_key"], top_n)
        top_snr = snr_by_raw[top_raw]
        top_pcts = percentile_array(top_snr, sorted_finite)
        finite_pcts = top_pcts[np.isfinite(top_pcts)]
        n_nan = top_n - len(finite_pcts)
        med = float(np.median(finite_pcts)) if finite_pcts.size else float("nan")
        print(
            f"    n={len(finite_pcts)} finite (NaN={n_nan})  "
            f"median pct={med:.1f}  "
            f"frac<p25={(finite_pcts < 25).mean():.2f}  "
            f"frac>p75={(finite_pcts > 75).mean():.2f}"
        )
        per_source.append({
            "label": src["label"],
            "pcts": finite_pcts,
            "n_nan": n_nan,
            "median": med,
        })

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    bins = np.linspace(0, 100, 21)

    for ax, data, bg in zip(axes, per_source, BG_COLORS):
        ax.set_facecolor(bg)
        ax.hist(data["pcts"], bins=bins, color="#222222", edgecolor="white", linewidth=0.4)
        ax.axvline(data["median"], color="#C0392B", linestyle="--", linewidth=1.4,
                   label=f"median p{data['median']:.0f}")
        ax.axvspan(0, 100, alpha=0)
        ax.set_xlim(0, 100)
        ax.set_xlabel("SNR percentile (full dataset)")
        ax.set_title(
            f"{data['label']}  (n={len(data['pcts'])}, NaN={data['n_nan']})",
            fontsize=11, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("count of top-N anomalies")
    fig.suptitle(
        f"Top-{top_n} anomalies vs SNR percentile  (detector: {method})",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["flow", "iforest"])
    parser.add_argument("--top-n", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    make_figure(args.method, args.top_n, args.out)


if __name__ == "__main__":
    main()
