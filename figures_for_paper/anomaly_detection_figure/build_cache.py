"""Rebuild the figure cache from raw anomaly-score and neighbours HDF5 files.

Run from this directory:
    python build_cache.py

Inputs (paths are absolute, edit if your data lives elsewhere):
  - /work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5
      Provides `images_hsc[i]`, `ra[i]`, `dec[i]`.
  - /work1/.../anomaly_detection/outputs/anomaly_scores_{ours,aion,ins}_367k.h5
      Each file has `raw_index[N]` and one or more flat score datasets keyed by
      `<model>/<rep>/<method>`. We pull the `flow` key per source.

For each of the 3 sources (Physics / AION / Instrument), this script:
  1. Loads the chosen `flow` score for all 366,706 entries.
  2. Sorts by descending NLL, takes the top `TOP_N * CANDIDATE_MULT` candidates.
  3. Walks them top-down and keeps a candidate iff it is more than DEDUP_ARCSEC
     from every previously kept candidate (small-angle great-circle on RA/Dec).
     Stops at TOP_N kept entries.
  4. Records the per-model NLL percentile (rank/Nfinite × 100) for each kept entry.
  5. Loads the corresponding HSC stamps from `neighbours_v2.h5`.
  6. Saves an npz to `_cache/`.

The cache files are what `make_figure.py` consumes; that script does not touch
the raw HDF5s.
"""
from pathlib import Path

import h5py
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
SCORES_DIR = Path(
    "/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/"
    "anomaly_detection/outputs"
)

# ─── What to extract ──────────────────────────────────────────────────────────

TOP_N = 12
DEDUP_ARCSEC = 10.0      # angular dedup threshold
CANDIDATE_MULT = 500     # pull TOP_N * this many candidates before deduping

# Each entry produces one .npz in _cache/ named by `cache_stem`.
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


def cache_path(src):
    stem = Path(src["scores_file"]).stem
    key = src["score_key"].replace("/", "_")
    return CACHE_DIR / f"{stem}__{key}__top{TOP_N}.npz"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def top_n_with_percentiles(scores_path, score_key, n):
    """Return (top_raw[n], top_pcts[n]) for the `n` highest-score entries.

    `top_pcts[i]` is the percentile (0-100) of the i-th entry's score among all
    finite scores in that file.
    """
    with h5py.File(scores_path, "r") as f:
        raw_index = f["raw_index"][:]
        node = f
        for part in score_key.split("/"):
            node = node[part]
        scores = node[:]
    finite = np.isfinite(scores)
    sorted_finite = np.sort(scores[finite])
    order = np.argsort(scores * finite.astype(float))[::-1]
    top_idx = order[:n]
    top_raw = raw_index[top_idx]
    top_scores = scores[top_idx]
    top_pcts = np.array([
        np.searchsorted(sorted_finite, s, side="left") / len(sorted_finite) * 100
        for s in top_scores
    ])
    return top_raw, top_pcts


def dedup_by_radec(ranked_raw, ra_all, dec_all, n_keep, min_sep_arcsec):
    """Keep top-ranked indices that are >min_sep_arcsec from every already-kept one.

    Small-angle approximation:
        sep_arcsec ≈ sqrt((Δra·cos(dec))² + Δdec²) · 3600.
    """
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


def load_hsc_images(h5_data, raw_indices):
    """h5py needs strictly increasing fancy indices; sort, slice, restore order."""
    sort_order = np.argsort(raw_indices)
    sorted_idx = raw_indices[sort_order]
    imgs = h5_data["images_hsc"][sorted_idx]
    return imgs[np.argsort(sort_order)]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading RA/Dec from {NEIGHBORS_HDF5}")
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        ra_all = f["ra"][:]
        dec_all = f["dec"][:]
    print(f"  N_total={len(ra_all)}")

    n_candidates = TOP_N * CANDIDATE_MULT
    with h5py.File(NEIGHBORS_HDF5, "r") as h5_data:
        for src in SOURCES:
            scores_path = SCORES_DIR / src["scores_file"]
            print(f"\n[{src['label']}] {scores_path.name} :: {src['score_key']}")

            top_raw_big, top_pcts_big = top_n_with_percentiles(
                scores_path, src["score_key"], n_candidates,
            )
            kept_raw = dedup_by_radec(
                top_raw_big, ra_all, dec_all, TOP_N, DEDUP_ARCSEC,
            )
            order_map = {int(r): i for i, r in enumerate(top_raw_big)}
            kept_pcts = np.array([top_pcts_big[order_map[int(r)]] for r in kept_raw])
            hsc_imgs = load_hsc_images(h5_data, kept_raw)

            cp = cache_path(src)
            np.savez(
                cp,
                top_raw=kept_raw, top_pcts=kept_pcts, hsc_imgs=hsc_imgs,
                label=np.array(src["label"]),
                score_key=np.array(src["score_key"]),
                scores_file=np.array(src["scores_file"]),
                dedup_arcsec=np.array(DEDUP_ARCSEC),
            )
            print(f"  kept={len(kept_raw)}/{len(top_raw_big)}  →  {cp.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
