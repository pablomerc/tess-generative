"""
Side-by-side comparison of top anomalies from three models.

Presets (--preset):
  physics    [default] hsc_mean/<method>:
    1. Ours (physics)      100k  — ours/hsc_mean/<method>
    2. AION                100k  — aion/hsc_mean_pca64/<method>
    3. Contrastive (phys)  <contrastive-size>  — ours/hsc_mean/<method>

  instrument  instrument_emb/<method>:
    1. Ours (instrument)   100k  — ours/hsc_flat/<method>
    2. AION                100k  — aion/hsc_mean_pca64/<method>
    3. Contrastive (inst)  <contrastive-size>  — ours/instrument_emb/<method>

Run from galaxy_model/:
  python anomaly_detection/compare_top_anomalies.py [--preset physics|instrument]
    [--method flow|iforest] [--contrastive-size 10k|100k] [--top-n 12] [--out OUT.png]
"""
import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUTS_DIR = _here / "outputs"
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"

def _build_sources(preset, method, contrastive_size):
    aion_method = method  # aion uses same method name
    if preset == "physics":
        return [
            {
                "label": "Ours (physics) 100k",
                "scores_file": "anomaly_scores_ours_100k.h5",
                "score_key": f"ours/hsc_mean/{method}",
            },
            {
                "label": "AION 100k",
                "scores_file": "anomaly_scores_aion_100k.h5",
                "score_key": f"aion/hsc_mean_pca64/{aion_method}",
            },
            {
                "label": f"Contrastive (physics) {contrastive_size}",
                "scores_file": f"anomaly_scores_contrastive_{contrastive_size}.h5",
                "score_key": f"ours/hsc_mean/{method}",
            },
        ]
    else:  # instrument
        return [
            {
                "label": "Ours (instrument) 100k",
                "scores_file": "anomaly_scores_ins_100k.h5",
                "score_key": f"ours/hsc_flat/{method}",
            },
            {
                "label": "AION 100k",
                "scores_file": "anomaly_scores_aion_100k.h5",
                "score_key": f"aion/hsc_mean_pca64/{aion_method}",
            },
            {
                "label": f"Contrastive (instrument) {contrastive_size}",
                "scores_file": f"anomaly_scores_contrastive_instrument_{contrastive_size}.h5",
                "score_key": f"ours/instrument_emb/{method}",
            },
        ]


def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _load_hsc_images(h5_data, raw_indices):
    sort_order = np.argsort(raw_indices)
    sorted_idx = raw_indices[sort_order]
    imgs = h5_data["images_hsc"][sorted_idx]
    return imgs[np.argsort(sort_order)]


def _top_n_raw_indices(scores_path, score_key, n):
    with h5py.File(scores_path, "r") as f:
        raw_index = f["raw_index"][:]
        # nested key like "ours/hsc_mean/flow"
        parts = score_key.split("/")
        node = f
        for p in parts:
            node = node[p]
        scores = node[:]
    finite = np.isfinite(scores)
    order = np.argsort(scores * finite.astype(float))[::-1]
    return raw_index[order[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["physics", "instrument"], default="physics")
    parser.add_argument("--method", choices=["flow", "iforest", "knn", "mahal"], default="flow")
    parser.add_argument("--contrastive-size", choices=["10k", "100k"], default="10k")
    parser.add_argument("--top-n", type=int, default=12, help="Examples per source (default 12)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    SOURCES = _build_sources(args.preset, args.method, args.contrastive_size)

    top_n = args.top_n
    n_cols_per = 4
    n_rows_per = (top_n + n_cols_per - 1) // n_cols_per  # = 2 for top_n=8

    n_sources = len(SOURCES)
    img_size = 2.0  # inches per image

    fig_w = n_sources * n_cols_per * img_size + (n_sources - 1) * 0.4
    fig_h = n_rows_per * img_size + 0.5  # +0.5 for column title

    fig = plt.figure(figsize=(fig_w, fig_h))

    # outer grid: one column per source, with gaps between
    outer = gridspec.GridSpec(
        1, n_sources, figure=fig,
        left=0.01, right=0.99, top=0.88, bottom=0.01,
        wspace=0.06,
    )

    print(f"Loading top-{top_n} anomaly indices per source...")
    with h5py.File(NEIGHBORS_HDF5, "r") as h5_data:
        for col_idx, src in enumerate(SOURCES):
            scores_path = OUTPUTS_DIR / src["scores_file"]
            print(f"  {src['label']}: {src['score_key']} from {scores_path.name}")
            top_raw = _top_n_raw_indices(scores_path, src["score_key"], top_n)
            hsc_imgs = _load_hsc_images(h5_data, top_raw)

            inner = gridspec.GridSpecFromSubplotSpec(
                n_rows_per, n_cols_per,
                subplot_spec=outer[col_idx],
                hspace=0.05, wspace=0.05,
            )

            for i in range(top_n):
                r, c = divmod(i, n_cols_per)
                ax = fig.add_subplot(inner[r, c])
                ax.imshow(_to_rgb(hsc_imgs[i]))
                ax.axis("off")
                if r == 0 and c == 0:
                    ax.set_title(f"#{i+1}", fontsize=6, pad=1)
                else:
                    ax.set_title(f"#{i+1}", fontsize=6, pad=1)

            # column header
            fig.text(
                (outer[col_idx].get_position(fig).x0 + outer[col_idx].get_position(fig).x1) / 2,
                0.92,
                src["label"],
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
            )

    fig.suptitle(f"Top anomalies — hsc_mean / {args.method} score  [contrastive {args.contrastive_size}]", fontsize=12, y=0.98)

    out_path = Path(args.out) if args.out else OUTPUTS_DIR / "figures_compare" / f"compare_top_anomalies_{args.preset}_{args.method}_contrastive{args.contrastive_size}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
