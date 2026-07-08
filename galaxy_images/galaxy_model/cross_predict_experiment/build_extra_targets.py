"""
Build a sidecar parquet of extra cross-predict targets that aren't in
neighbours_v2.h5: cross-matches `/orcd/pool/007/pablomer/neighbours_v2.h5`
HSC `object_id_hsc` against `/home/pablomer/orcd/scratch/hsc_downstream/catalog.parquet`
to pull per-band Galactic extinction `a_g/r/i/z/y`.

Output: cross_predict_experiment/extra_targets_per_h5_row.parquet
        — one row per H5 row [0..N_h5-1], columns
          [hsc_a_g, hsc_a_r, hsc_a_i, hsc_a_z, hsc_a_y].
        Non-MMU rows (source_type != 0) stay NaN; CrossPredictDataset
        only consumes the MMU subset anyway.

Run once before re-submitting cross_predict SLURM jobs:
    python build_extra_targets.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

H5_PATH = Path("/orcd/pool/007/pablomer/neighbours_v2.h5")
CATALOG_PATH = Path("/home/pablomer/orcd/scratch/hsc_downstream/catalog.parquet")
OUT_PATH = Path(__file__).resolve().parent / "extra_targets_per_h5_row.parquet"

EXTINCTION_BANDS = ("g", "r", "i", "z", "y")
SIDECAR_COLS = [f"hsc_a_{b}" for b in EXTINCTION_BANDS]


def main():
    print(f"Reading object_id_hsc + source_type from {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        oid_raw = f["object_id_hsc"][:]
        src = f["source_type"][:]
    n_total = oid_raw.size
    mmu_mask = src == 0
    n_mmu = int(mmu_mask.sum())
    print(f"  H5 rows: {n_total},  MMU rows (source_type==0): {n_mmu}")

    # Decode bytes/object → int64. Non-HSC rows (legacy-only) have empty
    # strings; those stay -1 and won't join (NaN extinction is fine since the
    # cross-predict dataset only uses MMU rows anyway).
    oid_int = np.full(n_total, -1, dtype=np.int64)
    n_empty = 0
    for i, s in enumerate(oid_raw):
        if isinstance(s, (bytes, bytearray)):
            s = s.decode()
        else:
            s = str(s)
        if s == "":
            n_empty += 1
            continue
        oid_int[i] = int(s)
    print(f"  decoded object_id_hsc → int64,  empties={n_empty},  "
          f"valid range [{oid_int[oid_int >= 0].min()}, {oid_int.max()}]")

    print(f"Reading catalog from {CATALOG_PATH}")
    cat = pd.read_parquet(CATALOG_PATH,
                          columns=["object_id"] + [f"a_{b}" for b in EXTINCTION_BANDS])
    print(f"  catalog rows: {len(cat)}")

    # Reindex catalog by object_id, then look up our H5 ids in the same order.
    cat_idx = cat.set_index("object_id")
    matched = cat_idx.reindex(oid_int)

    out = pd.DataFrame({
        col: matched[f"a_{b}"].to_numpy(dtype=np.float32)
        for col, b in zip(SIDECAR_COLS, EXTINCTION_BANDS)
    })

    # Verify: every MMU row should have a finite extinction value.
    mmu_finite = np.isfinite(out.iloc[mmu_mask].to_numpy()).all(axis=1).sum()
    print(f"  MMU rows with all 5 bands finite: {mmu_finite} / {n_mmu}  "
          f"({100.0 * mmu_finite / n_mmu:.2f}%)")
    if mmu_finite != n_mmu:
        bad = n_mmu - mmu_finite
        raise RuntimeError(
            f"Cross-match incomplete: {bad} MMU rows have NaN in hsc_a_*. "
            "Aborting before sidecar write — investigate."
        )

    # Per-column ranges as a sanity check.
    print("\nPer-column stats on MMU rows:")
    sub = out.iloc[mmu_mask]
    for c in SIDECAR_COLS:
        v = sub[c].to_numpy()
        print(f"  {c:14s}  min={v.min():.4f}  p50={np.median(v):.4f}  "
              f"max={v.max():.4f}  mean={v.mean():.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
