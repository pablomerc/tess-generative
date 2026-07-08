"""Render a 4x3 grid of the top-12 HSC anomaly stamps for one encoder+method.

Reads the scores HDF5, picks top candidates by descending score, dedups by
RA/Dec at 10", and renders RGB stamps from the HSC downstream memmap dataset.

Usage:
  python anomaly_detection/engaging/plot_top12_hscdown.py \
    --scores outputs/anomaly_scores_ours_hscdown_474k.h5 \
    --score-key ours/hsc_flat/flow \
    --label "Ours (Physics) — flow" \
    --out outputs/figures_hscdown_474k/top12_ours_flow.png
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

TOP_N = 12
DEDUP_ARCSEC = 10.0
CANDIDATE_MULT = 500


def to_rgb(img_chw, pct_lo=1, pct_hi=99):
    """Percentile-stretched RGB (g/r/i = first 3 bands). Same as figures_for_paper version."""
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def dedup_by_radec(ranked_raw, ra_all, dec_all, n_keep, min_sep_arcsec):
    threshold_deg2 = (min_sep_arcsec / 3600.0) ** 2
    kept, kept_ra, kept_dec, kept_cosdec = [], [], [], []
    for raw in ranked_raw:
        ra = float(ra_all[raw])
        dec = float(dec_all[raw])
        too_close = False
        for kra, kdec, kcd in zip(kept_ra, kept_dec, kept_cosdec):
            d_ra = (ra - kra) * kcd
            d_dec = dec - kdec
            if d_ra * d_ra + d_dec * d_dec < threshold_deg2:
                too_close = True
                break
        if too_close:
            continue
        kept.append(int(raw))
        kept_ra.append(ra)
        kept_dec.append(dec)
        kept_cosdec.append(np.cos(np.deg2rad(dec)))
        if len(kept) >= n_keep:
            break
    return np.asarray(kept, dtype=np.int64)


def top_candidates(scores_path, score_key, n_candidates):
    with h5py.File(scores_path, "r") as f:
        raw_index = f["raw_index"][:]
        node = f
        for part in score_key.split("/"):
            node = node[part]
        scores = node[:]
    finite = np.isfinite(scores)
    sorted_finite = np.sort(scores[finite])
    order = np.argsort(scores * finite.astype(float))[::-1]
    top_idx = order[:n_candidates]
    top_raw = raw_index[top_idx]
    top_scores = scores[top_idx]
    top_pcts = np.array([
        np.searchsorted(sorted_finite, s, side="left") / len(sorted_finite) * 100
        for s in top_scores
    ])
    return top_raw, top_pcts


def render_grid(hsc_imgs, top_pcts, label, out_path, n_cols=3, n_rows=4):
    img_size = 1.4
    header = 0.42
    fig_w = n_cols * img_size
    fig_h = n_rows * img_size + header
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=0.01, right=0.99,
        top=1.0 - header / fig_h,
        bottom=0.01,
        hspace=0.04, wspace=0.04,
    )
    n = min(n_rows * n_cols, len(hsc_imgs))
    for i in range(n):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.imshow(to_rgb(hsc_imgs[i]), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(
            0.5, 0.04, f"#{i+1}  ({top_pcts[i]:.2f}%)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="black",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                      alpha=0.75, linewidth=0),
        )
    fig.suptitle(label, fontsize=12, y=1.0 - 0.08 / fig_h)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--score-key", required=True,
                   help="e.g. ours/hsc_flat/flow")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--dedup-arcsec", type=float, default=DEDUP_ARCSEC)
    p.add_argument("--candidate-mult", type=int, default=CANDIDATE_MULT)
    args = p.parse_args()

    print(f"Loading dataset (HSC downstream memmap)...")
    ds = HSCDownstreamDataset(view="aion")  # need raw 5-band; we use first 3 for RGB
    ra_all = ds.ra
    dec_all = ds.dec

    n_candidates = args.top_n * args.candidate_mult
    print(f"Loading top-{n_candidates} from {args.scores} :: {args.score_key}")
    top_raw_big, top_pcts_big = top_candidates(args.scores, args.score_key, n_candidates)

    kept_raw = dedup_by_radec(top_raw_big, ra_all, dec_all, args.top_n, args.dedup_arcsec)
    order_map = {int(r): i for i, r in enumerate(top_raw_big)}
    kept_pcts = np.array([top_pcts_big[order_map[int(r)]] for r in kept_raw])
    print(f"  kept {len(kept_raw)} after RA/Dec dedup ({args.dedup_arcsec}\")")

    # Pull stamps from memmap.
    hsc_imgs = np.stack([ds._load_flux(int(r)) for r in kept_raw])  # (N, 5, 160, 160)

    render_grid(hsc_imgs, kept_pcts, args.label, args.out)


if __name__ == "__main__":
    main()
