"""HSC-only re-render of visualize_top_anomalies.py grids.

Reads an existing anomaly_scores_*.h5 + neighbours_v2.h5 and plots, for every
score key, a 5x5 grid of the top-N HSC images (no Legacy row). Used to resend
results without re-running anything.

Run from galaxy_model/:
  python anomaly_detection/top_anomalies_hsc_only.py \
    --scores outputs/anomaly_scores_ours_367k.h5 \
    --suffix ours_367k_hsc_only \
    --out-dir anomaly_detection/outputs/figures_ours_367k_hsc_only
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
NEIGHBORS_HDF5_DEFAULT = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"


def _to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _load_hsc(h5_data, raw_indices):
    sort_order = np.argsort(raw_indices)
    sorted_idx = raw_indices[sort_order]
    imgs = h5_data["images_hsc"][sorted_idx]
    return imgs[np.argsort(sort_order)]


def _dedup_by_radec(ranked_raw, ra_all, dec_all, n_keep, min_sep_arcsec):
    """Keep top-ranked indices that are >min_sep_arcsec from every already-kept one.

    Uses the small-angle approximation:
      sep_arcsec ≈ sqrt((dRA·cos(dec0))² + dDec²) · 3600.
    Brute-force O(n_keep × n_candidates) — fine for n_keep ~ 25.
    """
    threshold_deg = min_sep_arcsec / 3600.0
    threshold_deg2 = threshold_deg ** 2
    kept = []
    kept_ra = []
    kept_dec = []
    kept_cosdec = []
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


def plot_hsc_grid(top_raw, h5_data, score_label, top_n, out_path,
                  ra_all=None, dec_all=None, dedup_arcsec=0.0):
    if dedup_arcsec > 0 and ra_all is not None and dec_all is not None:
        before = len(top_raw)
        top_raw = _dedup_by_radec(top_raw, ra_all, dec_all, top_n, dedup_arcsec)
        print(f"    dedup({dedup_arcsec}\"): kept {len(top_raw)}/{before} candidates")

    n_cols = 5
    n_rows = max(1, top_n // n_cols)
    n_show = min(len(top_raw), n_cols * n_rows)

    hsc_imgs = _load_hsc(h5_data, top_raw[:n_show])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.atleast_2d(axes)
    for i in range(n_show):
        r, c = divmod(i, n_cols)
        axes[r, c].imshow(_to_rgb(hsc_imgs[i]))
        axes[r, c].set_title(f"#{i+1}\nidx={top_raw[i]}", fontsize=7)
        axes[r, c].axis("off")
    # blank any unfilled cells
    for i in range(n_show, n_cols * n_rows):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")
    title = f"Top {n_show} anomalies (HSC only"
    if dedup_arcsec > 0:
        title += f", dedup>{dedup_arcsec}\""
    title += f") — {score_label}"
    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--data", default=NEIGHBORS_HDF5_DEFAULT)
    parser.add_argument("--suffix", default="ours_367k_hsc_only")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--dedup-arcsec", type=float, default=0.0,
        help="If >0, drop near-duplicate sky positions: walk the score-ranked "
             "list and keep an entry only if it's >this many arcsec from every "
             "previously kept entry. 0 disables dedup.",
    )
    parser.add_argument(
        "--candidate-multiplier", type=int, default=200,
        help="When dedup is on, pull this × top-n candidates from the score "
             "ranking before deduping (default 200).",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.is_absolute() and not scores_path.exists():
        scores_path = _here / "outputs" / args.scores

    out_dir = Path(args.out_dir) if args.out_dir else _here / "outputs" / f"figures_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from {scores_path}")
    with h5py.File(scores_path, "r") as f:
        raw_index = f["raw_index"][:]
        score_keys = []
        f.visit(lambda name: score_keys.append(name) if isinstance(f[name], h5py.Dataset) and name != "raw_index" else None)
        all_scores = {k: f[k][:] for k in score_keys}
    print(f"  N={len(raw_index)}, score keys: {score_keys}")

    with h5py.File(args.data, "r") as h5_data:
        ra_all = h5_data["ra"][:] if args.dedup_arcsec > 0 else None
        dec_all = h5_data["dec"][:] if args.dedup_arcsec > 0 else None
        n_candidates = (args.top_n * args.candidate_multiplier) if args.dedup_arcsec > 0 else args.top_n

        for key, scores in all_scores.items():
            if not np.isfinite(scores).any():
                print(f"  Skipping {key} (all NaN)")
                continue
            finite_mask = np.isfinite(scores)
            sorted_idx = np.argsort(scores * finite_mask.astype(float))[::-1]
            top_raw = raw_index[sorted_idx[:n_candidates]]
            label = key.replace("/", "_")
            out_path = out_dir / f"top_anomalies_{label}_{args.suffix}.png"
            print(f"  Plotting top {args.top_n} for {key} -> {out_path}")
            plot_hsc_grid(
                top_raw, h5_data, key, args.top_n, out_path,
                ra_all=ra_all, dec_all=dec_all, dedup_arcsec=args.dedup_arcsec,
            )

    print(f"Done. Saved to {out_dir}")


if __name__ == "__main__":
    main()
