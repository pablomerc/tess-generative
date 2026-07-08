"""Build the joint 3-column top-12 figure for the HSC downstream experiment.

Replicates figures_for_paper/anomaly_detection_figure/make_figure.py but:
  - Pulls stamps from the HSC downstream memmap (not neighbours_v2.h5)
  - Uses `ours/hsc_flat/<method>` for both ours-physics and ours-instrument
    (paper figure used hsc_mean for physics)
  - Uses `aion/hsc_mean_full/<method>` (no PCA)
  - Renders one figure per method (flow + iforest) for the same 3 columns.

Usage:
  python anomaly_detection/engaging/build_joint_figure_hscdown.py \
    --suffix hscdown_474k \
    --out-dir outputs/figures_hscdown_474k
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ANOMALY_DIR = _HERE.parent
_GALAXY_MODEL = _ANOMALY_DIR.parent
for p in (str(_GALAXY_MODEL), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsc_downstream_dataset import HSCDownstreamDataset
from plot_top12_hscdown import to_rgb, dedup_by_radec, top_candidates

TOP_N = 12
N_ROWS = 4
N_COLS_PER = 3
DEDUP_ARCSEC = 10.0
CANDIDATE_MULT = 500

GROUP_BG_COLORS = ["#E7F3FC", "#DFECE3", "#FCE8EC"]


def sources(suffix):
    out_dir = _ANOMALY_DIR / "outputs"
    return [
        {"label": "Ours (Physics)",
         "scores_file": str(out_dir / f"anomaly_scores_ours_{suffix}.h5"),
         "score_key_template": "ours/hsc_flat/{method}",
         "bg": GROUP_BG_COLORS[0]},
        {"label": "AION (no PCA)",
         "scores_file": str(out_dir / f"anomaly_scores_aion_{suffix}.h5"),
         "score_key_template": "aion/hsc_mean_full/{method}",
         "bg": GROUP_BG_COLORS[1]},
        {"label": "Ours (Instrument)",
         "scores_file": str(out_dir / f"anomaly_scores_ins_{suffix}.h5"),
         "score_key_template": "ours/hsc_flat/{method}",
         "bg": GROUP_BG_COLORS[2]},
    ]


def gather_for_method(srcs, method, ds, top_n, dedup_arcsec, candidate_mult):
    out = []
    for s in srcs:
        key = s["score_key_template"].format(method=method)
        n_cand = top_n * candidate_mult
        top_raw_big, top_pcts_big = top_candidates(s["scores_file"], key, n_cand)
        kept_raw = dedup_by_radec(top_raw_big, ds.ra, ds.dec, top_n, dedup_arcsec)
        order_map = {int(r): i for i, r in enumerate(top_raw_big)}
        kept_pcts = np.array([top_pcts_big[order_map[int(r)]] for r in kept_raw])
        imgs = np.stack([ds._load_flux(int(r)) for r in kept_raw])
        out.append({"label": s["label"], "bg": s["bg"],
                    "kept_raw": kept_raw, "kept_pcts": kept_pcts, "imgs": imgs,
                    "score_key": key})
    return out


def render_joint(group_data, method, out_path):
    n_groups = len(group_data)
    img_size = 1.30
    group_gap = 0.30
    header_pad = 0.42

    fig_w = n_groups * N_COLS_PER * img_size + (n_groups - 1) * group_gap
    fig_h = N_ROWS * img_size + header_pad

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
    outer = fig.add_gridspec(
        1, n_groups,
        left=0.005, right=0.995,
        top=1.0 - (header_pad * 0.75) / fig_h,
        bottom=0.005,
        wspace=group_gap / (N_COLS_PER * img_size),
    )

    # Group separators (vertical dashed lines between columns).
    sep_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    sep_ax.set_xlim(0, 1); sep_ax.set_ylim(0, 1)
    sep_ax.set_xticks([]); sep_ax.set_yticks([])
    sep_ax.patch.set_alpha(0)
    for i in range(1, n_groups):
        x = i / n_groups
        sep_ax.plot([x, x], [0.02, 0.96], color="#888888",
                    linestyle="--", dashes=(6, 4), linewidth=1.0,
                    transform=sep_ax.transAxes)
    sep_ax.set_zorder(-1)

    for g_i, gdata in enumerate(group_data):
        inner = outer[0, g_i].subgridspec(
            N_ROWS, N_COLS_PER, hspace=0.04, wspace=0.04,
        )
        # Background tint (one per group).
        # Done by adding an axes underneath the inner grid.
        n = min(N_ROWS * N_COLS_PER, len(gdata["imgs"]))
        for k in range(n):
            ax = fig.add_subplot(inner[k // N_COLS_PER, k % N_COLS_PER])
            ax.imshow(to_rgb(gdata["imgs"][k]), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(
                0.5, 0.04, f"#{k+1}  ({gdata['kept_pcts'][k]:.2f}%)",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color="black",
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                          alpha=0.75, linewidth=0),
            )

        # Column header.
        # Place at the top of the inner gridspec.
        bbox_first = fig.axes[-N_ROWS * N_COLS_PER].get_position()
        bbox_last_in_first_row = fig.axes[-N_ROWS * N_COLS_PER + N_COLS_PER - 1].get_position()
        x_center = (bbox_first.x0 + bbox_last_in_first_row.x1) / 2
        y_top = bbox_first.y1
        fig.text(x_center, y_top + 0.012, gdata["label"],
                 ha="center", va="bottom", fontsize=13, fontweight="bold")

    fig.suptitle(f"Top-12 anomalies (method = {method}, HSC downstream {len(group_data[0]['kept_raw'])} per col)",
                 fontsize=10, y=0.995)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="hscdown_474k")
    p.add_argument("--out-dir", default=None,
                   help="Default: anomaly_detection/outputs/figures_<suffix>")
    p.add_argument("--methods", nargs="+", default=["flow", "iforest"])
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--dedup-arcsec", type=float, default=DEDUP_ARCSEC)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        _ANOMALY_DIR / "outputs" / f"figures_{args.suffix}"
    )

    print("Loading HSC downstream memmap...")
    ds = HSCDownstreamDataset(view="aion")
    print(f"  N={len(ds)}")

    srcs = sources(args.suffix)
    for method in args.methods:
        print(f"\n=== Building joint figure for method={method} ===")
        group_data = gather_for_method(
            srcs, method, ds, args.top_n, args.dedup_arcsec, CANDIDATE_MULT
        )
        out_path = out_dir / f"joint_top{args.top_n}_{method}_{args.suffix}.png"
        render_joint(group_data, method, out_path)


if __name__ == "__main__":
    main()
