"""Build the same-instrument neighbour index for the LSST DP2 store.

WHY THIS FILE EXISTS
--------------------
The dual-encoder models feed the *instrument* encoder k images of other
galaxies taken by the same survey as the anchor ("same-instrument
neighbours"). For HSC and Legacy those neighbours come from
data/neighbors_efficient/catalog.parquet (neighbor_idx_hsc / _legacy):
20 nearest galaxies per anchor, as row indices into the 468,197-row parent
catalogue.

Those lists cannot be reused for LSST. All 65,163 LSST rows are parent
*anchor* rows (source_type == 0); the 364,456 rows that exist only to be
neighbours were never crossmatched to DP2, so of an anchor's 20 HSC
neighbours exactly zero have an LSST stamp (measured 2026-09-26). This
script therefore builds the neighbour lists *inside* the LSST store: for
each of the M store rows, the K nearest other store rows in the same field.

Within COSMOS (34,132 objects over ~10 deg^2) the 5th neighbour sits at a
median 58" and the 20th well under 3'; DP2 depth and PSF vary on the
dither scale (degrees), so at these separations a neighbour shares the
anchor's observing conditions, which is what the instrument encoder needs.

WHAT IT WRITES (sidecar files next to the store; nothing existing is touched)
----------------------------------------------------------------------------
  data/lsst_efficient/lsst_neighbors.npz
      idx          int32   (M, K)  LSST *store* rows of the K nearest
                                   neighbours, nearest first; -1 = none
      dist_arcsec  float32 (M, K)  great-circle separation to each
      parent_row   int64   (M,)    parent-catalogue row of each store row
                                   (inverse of lsst_row_map.npy), so a
                                   loader can reach the HSC/Legacy images
                                   of the same galaxy without a search
  data/lsst_efficient/lsst_neighbors.json
      K, per-field counts and distance quantiles, the sha256 of the store's
      object_id_hsc column (so a loader can assert index/store alignment),
      build date and this file's path.
  galaxy_images/galaxy_model/lsst/data_preprocessing/figures/lsst_neighbors_qc.png
      QC figure: distance histograms per field + one anchor with its
      neighbours on the sky.

DESIGN CHOICES (each is a flag or a constant so they are easy to revisit)
-------------------------------------------------------------------------
* K = 20, mirroring the parent catalogue; models use max_neighbors 5 or 15.
* Neighbours are restricted to the anchor's own field. Distance alone
  guarantees this (the fields are ~200 deg apart) but the KD-tree is built
  per field so the guarantee is structural, not accidental.
* Geometry only. No band-availability or stamp-quality cut is baked in:
  the loader applies has_<band> flags and any n_input floor at load time,
  the same way the parent loader masks -1 entries. The index stays valid
  for every band subset and every quality policy.
* Self is excluded (query K+1, drop the first hit).
* Exact great-circle distances: positions are mapped to unit vectors and a
  KD-tree on chord length is queried; chord -> angle is exact.
* Distances are stored in ARCSEC to match the LSST store's own convention
  (lsst_meta.parquet: match_dist_arcsec). The parent stores arcmin.
* Row space is the LSST store (0..M-1), because that is what indexes
  lsst_image.bin. parent_row bridges to the HSC/Legacy binaries.

Runtime: seconds on a login node (scipy KD-tree on 65k points). Run:
    cd tess-generative && conda activate torchenv
    PYTHONPATH=. python -m galaxy_images.galaxy_model.lsst.data_preprocessing.build_lsst_neighbors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ----------------------------------------------------------------- constants
STORE_DIR = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
META = os.path.join(STORE_DIR, "lsst_meta.parquet")
ROW_MAP = os.path.join(STORE_DIR, "lsst_row_map.npy")
OUT_NPZ = os.path.join(STORE_DIR, "lsst_neighbors.npz")
OUT_JSON = os.path.join(STORE_DIR, "lsst_neighbors.json")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

K_DEFAULT = 20           # parent-catalogue parity
ARCSEC_PER_DEG = 3600.0


# ------------------------------------------------------------------ geometry
def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """(N,) RA/Dec in degrees -> (N, 3) unit vectors on the sphere.

    A KD-tree in 3-D Euclidean space then ranks points by chord length, and
    chord length is a monotonic function of angular separation, so the
    k nearest by chord are exactly the k nearest by angle. No small-angle
    approximation, no RA wrap-around issue, no cos(dec) fudge.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])


def chord_to_arcsec(chord: np.ndarray) -> np.ndarray:
    """Chord length between two unit vectors -> angular separation (arcsec).

    For unit vectors, chord = 2 sin(theta / 2), so theta = 2 asin(chord / 2).
    """
    theta_rad = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    return np.degrees(theta_rad) * ARCSEC_PER_DEG


# --------------------------------------------------------------- core build
def neighbours_within_field(ra: np.ndarray, dec: np.ndarray, k: int):
    """K nearest neighbours among the given points, self excluded.

    Returns (idx, dist_arcsec), each (N, k), idx in LOCAL 0..N-1 space
    (the caller maps back to store rows). Rows with fewer than k other
    points (only possible in a tiny field) are padded with -1 / NaN.
    """
    n = len(ra)
    xyz = radec_to_unit(ra, dec)
    tree = cKDTree(xyz)
    # Ask for k+1 because the nearest point to every galaxy is itself at
    # distance 0. cKDTree pads missing neighbours with index == n and
    # distance == inf when fewer than k+1 points exist.
    kk = min(k + 1, n)
    chord, local = tree.query(xyz, k=kk)
    chord = np.atleast_2d(chord)
    local = np.atleast_2d(local)

    # Drop self. Normally it is column 0 (distance exactly 0). If two objects
    # share a position to machine precision, "self" can land in column 1
    # instead, so locate it by index rather than assuming column 0. A k-NN
    # query never returns the same index twice, so each row has at most one
    # self hit; rows where self was pushed out entirely (only possible with
    # several exact duplicates) drop column 0.
    own = np.arange(n)[:, None]
    is_self = local == own
    is_self[~is_self.any(axis=1), 0] = True
    keep = ~is_self                                   # exactly one False per row
    local = local[keep].reshape(n, kk - 1)
    chord = chord[keep].reshape(n, kk - 1)

    idx = np.full((n, k), -1, dtype=np.int32)
    dist = np.full((n, k), np.nan, dtype=np.float32)
    m = min(k, kk - 1)
    valid = local[:, :m] < n                        # cKDTree pads with n
    idx[:, :m] = np.where(valid, local[:, :m], -1)
    dist[:, :m] = np.where(valid, chord_to_arcsec(chord[:, :m]), np.nan)
    return idx, dist


def build(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    meta = pd.read_parquet(META, columns=["object_id_hsc", "field", "ra", "dec"])
    M = len(meta)
    idx_all = np.full((M, k), -1, dtype=np.int32)
    dist_all = np.full((M, k), np.nan, dtype=np.float32)

    # One KD-tree per field: neighbours never cross a field boundary, and a
    # future third field (XMM-LSS, when Rubin coadds it) drops in unchanged.
    for field, rows in meta.groupby("field").indices.items():
        rows = np.asarray(rows)
        idx_local, dist_local = neighbours_within_field(
            meta["ra"].to_numpy()[rows], meta["dec"].to_numpy()[rows], k)
        # local (0..n_field-1) -> store rows; keep -1 as -1
        idx_all[rows] = np.where(idx_local >= 0, rows[np.clip(idx_local, 0, None)], -1)
        dist_all[rows] = dist_local
        print(f"{field:<8} n={len(rows):>6,}  "
              f"d1 median {np.nanmedian(dist_local[:, 0]):5.0f}\"  "
              f"d5 {np.nanmedian(dist_local[:, min(4, k - 1)]):5.0f}\"  "
              f"d{k} {np.nanmedian(dist_local[:, k - 1]):5.0f}\"", flush=True)

    # parent_row: inverse of lsst_row_map (parent row -> store row).
    row_map = np.load(ROW_MAP)
    parent_rows = np.where(row_map >= 0)[0]
    parent_row = np.full(M, -1, dtype=np.int64)
    parent_row[row_map[parent_rows]] = parent_rows
    return idx_all, dist_all, parent_row, meta


# ---------------------------------------------------------------- validation
def validate(idx, dist, parent_row, meta, k):
    M = len(meta)
    assert idx.shape == (M, k) and dist.shape == (M, k)
    assert (parent_row >= 0).all(), "every store row must have a parent row"
    assert len(np.unique(parent_row)) == M, "parent_row must be a bijection"
    own = np.arange(M)[:, None]
    assert not (idx == own).any(), "a row lists itself as a neighbour"
    valid = idx >= 0
    assert valid.all(), "some rows have fewer than K neighbours (unexpected: fields are dense)"
    field = meta["field"].to_numpy()
    assert (field[idx] == field[:, None]).all(), "neighbour crosses a field boundary"
    # nearest-first ordering and finite distances
    assert np.isfinite(dist).all()
    assert (np.diff(dist, axis=1) >= -1e-3).all(), "distances not sorted ascending"
    # Scale sanity. Very close pairs are not an error (deblended pairs exist)
    # but they are worth knowing about, so report rather than assert.
    close = int((dist[:, 0] < 1.0).sum())
    print(f"pairs closer than 1\": {close}  (nearest overall {dist[:, 0].min():.2f}\")")
    assert dist.max() < 30 * 60, f"neighbour beyond 30': {dist.max():.0f}\""
    print("validation OK")


def qc_figure(idx, dist, meta, k, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = list(meta.groupby("field").indices.keys())
    fig, axes = plt.subplots(1, len(fields) + 1, figsize=(5.2 * (len(fields) + 1), 4.2))
    cols = [0, 4, 14, k - 1]
    for ax, f in zip(axes, fields):
        rows = np.asarray(meta.groupby("field").indices[f])
        for c in cols:
            ax.hist(dist[rows, c], bins=80, histtype="step", lw=1.6, label=f"neighbour {c + 1}")
        ax.set_title(f"{f}: separation to k-th neighbour (n={len(rows):,})")
        ax.set_xlabel("arcsec"); ax.set_ylabel("objects"); ax.legend(frameon=False, fontsize=9)
    # one example anchor with its neighbours
    ax = axes[-1]
    rows = np.asarray(meta.groupby("field").indices[fields[0]])
    a = rows[len(rows) // 2]
    nb = idx[a]
    ra0, dec0 = meta["ra"].iloc[a], meta["dec"].iloc[a]
    cosd = np.cos(np.radians(dec0))
    box = 4.0 / 60  # deg half-width
    near = rows[(np.abs(meta["ra"].to_numpy()[rows] - ra0) * cosd < box) &
                (np.abs(meta["dec"].to_numpy()[rows] - dec0) < box)]
    ax.scatter((meta["ra"].to_numpy()[near] - ra0) * cosd * 60, (meta["dec"].to_numpy()[near] - dec0) * 60,
               s=6, c="#b8c4d6", label="other objects")
    ax.scatter((meta["ra"].to_numpy()[nb] - ra0) * cosd * 60, (meta["dec"].to_numpy()[nb] - dec0) * 60,
               s=26, c="#2a78d6", label=f"{k} neighbours")
    ax.scatter([0], [0], s=80, c="#eb6834", marker="*", label="anchor")
    ax.set_aspect("equal"); ax.set_xlabel("ΔRA·cos(dec) [arcmin]"); ax.set_ylabel("ΔDec [arcmin]")
    ax.set_title(f"{fields[0]} anchor {meta['object_id_hsc'].iloc[a]}"); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); fig.savefig(out_png, dpi=140)
    print(f"QC figure -> {out_png}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="neighbours per row (default 20)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing index")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    if os.path.exists(OUT_NPZ) and not args.force:
        raise SystemExit(f"{OUT_NPZ} exists; pass --force to rebuild (no silent overwrite)")

    idx, dist, parent_row, meta = build(args.k)
    validate(idx, dist, parent_row, meta, args.k)

    # fingerprint of the store rows this index is aligned to
    store_fp = hashlib.sha256(
        "\n".join(meta["object_id_hsc"].astype(str).tolist()).encode()).hexdigest()
    with open(os.path.join(STORE_DIR, "meta.json")) as f:
        store_meta = json.load(f)

    q = lambda a, p: float(np.percentile(a, p))
    per_field = {}
    for f, rows in meta.groupby("field").indices.items():
        rows = np.asarray(rows)
        per_field[f] = {
            "n": int(len(rows)),
            "d1_arcsec_median": q(dist[rows, 0], 50),
            "d5_arcsec_median": q(dist[rows, min(4, args.k - 1)], 50),
            "d5_arcsec_p90": q(dist[rows, min(4, args.k - 1)], 90),
            f"d{args.k}_arcsec_median": q(dist[rows, args.k - 1], 50),
            f"d{args.k}_arcsec_p90": q(dist[rows, args.k - 1], 90),
        }
    info = {
        "K": args.k, "M": int(len(meta)),
        "row_space": "lsst_efficient store rows (index into lsst_image.bin)",
        "dist_unit": "arcsec", "self_excluded": True, "same_field_only": True,
        "quality_cuts": "none (loader applies has_<band> and any n_input floor)",
        "per_field": per_field,
        "store_object_id_hsc_sha256": store_fp,
        "store_parent_object_id_hsc_sha256": store_meta.get("parent_object_id_hsc_sha256"),
        "build_date": str(date.today()),
        "built_by": "galaxy_images/galaxy_model/lsst/data_preprocessing/build_lsst_neighbors.py",
    }

    # write to temp names, then atomically replace (never a half-written index)
    tmp_npz, tmp_json = OUT_NPZ + ".tmp.npz", OUT_JSON + ".tmp"
    np.savez(tmp_npz, idx=idx, dist_arcsec=dist, parent_row=parent_row)
    with open(tmp_json, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp_npz, OUT_NPZ); os.replace(tmp_json, OUT_JSON)
    print(f"wrote {OUT_NPZ} ({os.path.getsize(OUT_NPZ) / 1e6:.1f} MB) and {OUT_JSON}")

    if not args.no_fig:
        qc_figure(idx, dist, meta, args.k, os.path.join(FIG_DIR, "lsst_neighbors_qc.png"))


if __name__ == "__main__":
    main()
