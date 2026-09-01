#!/usr/bin/env python3
"""Build the LSST DP2 efficient store: data/lsst_efficient/.

Sidecar to data/neighbors_efficient/ adding a Rubin/LSST DP2 view for every
galaxy that was crossmatched against DP2 (cross_matching pipeline). Follows the
efficient-store pattern: raw 160x160 stamps on disk, all preprocessing (clamp,
range compression, FoV zoom, normalization) at load time.

Layout written (see the README.md this script emits):
  lsst_image.bin     float16 (M, 6, 160, 160), bands g,r,i,z,u,y,
                     flux in NANOMAGGIES (DP2 nJy / 3630.78; nJy overflows
                     float16, nanomaggies do not). Missing u/y planes are NaN.
  lsst_row_map.npy   int64 (N_parent,) parent-catalog row -> compact row, -1=none
  lsst_meta.parquet  per compact row: ids, field, ra/dec, per-band psf_fwhm,
                     n_input_images (exposures in the coadd), has_<band> flags
  meta.json          shapes/dtype/unit/band order, parent-catalog fingerprint,
                     provenance, build stats

Band order g,r,i,z,u,y (not ugrizy): the always-present bands form a
contiguous prefix so griz experiments slice [:4], mirroring hsc_image.bin's
grizy convention.

Inclusion rule: an object needs complete g,r,i,z to enter the store; u/y are
optional extras (DEEP2-3 has no u-band coadds at all).

Runtime: one sequential pass over ~70 GB of cutout shards, ~45-90 min.
Output is written to <out>.building/ and renamed only after validation.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sys
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ----------------------------------------------------------------- constants
CM = "/work1/jeroenaudenaert/pablomer/cross_matching"
PARENT_DIR = "/work1/jeroenaudenaert/pablomer/data/neighbors_efficient"
OUT_DIR = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
PAIRED_INDEX = f"{CM}/outputs/paired_index_dp2.parquet"
# run2_full is the primary (complete) extraction; run1 pilot fills any gaps
SHARD_GROUPS = [
    sorted(glob.glob(f"{CM}/outputs/rsp_results/run2_full/dp2_cutouts/shard_*.parquet")),
    sorted(glob.glob(f"{CM}/outputs/rsp_results/run1_2026-08-05/dp2_cutouts/shard_*.parquet")),
]

BANDS = ["g", "r", "i", "z", "u", "y"]          # griz prefix, ragged bands last
CORE = ("g", "r", "i", "z")                      # required for inclusion
NJY_PER_NMGY = 10.0 ** ((31.4 - 22.5) / 2.5)     # 3630.78: AB zp 31.4 -> 22.5
F16_MAX = np.float16(65504).astype(np.float32)
STAMP = 160
N_SPOTCHECK = 20 # self-test


class Builder:
    def __init__(self, out_build):
        self.out = out_build
        self.bin_path = os.path.join(out_build, "lsst_image.bin")
        self.bin_file = open(self.bin_path, "wb")
        self.rows = []               # meta rows, index = compact row
        self.row_of = {}             # object_id_hsc -> compact row
        self.pending = {}            # object_id_hsc -> partial accumulation
        self.n_clipped_px = 0
        self.n_incomplete = 0
        self.spot = {}               # reservoir: id -> float32 (6,160,160)
        self._seen_for_reservoir = 0

    def add_row(self, oid, row):
        """Accumulate one (object, band) shard row into pending."""
        p = self.pending.setdefault(oid, {"bands": {}, "info": None})
        if p["info"] is None:
            p["info"] = (row["field"], float(row["ra"]), float(row["dec"]))
        p["bands"][row["band"]] = (
            np.asarray(row["flux"], dtype=np.float32).reshape(STAMP, STAMP),
            float(row["psf_fwhm"]), float(row["n_input_images"]))

    def finalize(self, oid):
        p = self.pending.pop(oid)
        if not all(b in p["bands"] for b in CORE):
            self.n_incomplete += 1
            return
        cube = np.full((len(BANDS), STAMP, STAMP), np.nan, dtype=np.float32)
        psf = np.full(len(BANDS), np.nan, dtype=np.float32)
        ninp = np.full(len(BANDS), np.nan, dtype=np.float32)
        has = {}
        for bi, b in enumerate(BANDS):
            has[b] = b in p["bands"]
            if has[b]:
                flux, psf[bi], ninp[bi] = p["bands"][b]
                flux = flux / NJY_PER_NMGY
                over = flux > F16_MAX
                if over.any():
                    self.n_clipped_px += int(over.sum())
                    flux = np.minimum(flux, F16_MAX)
                cube[bi] = flux
        # reservoir-sample objects for the post-build spot check
        self._seen_for_reservoir += 1
        if len(self.spot) < N_SPOTCHECK:
            self.spot[oid] = cube.copy()
        elif random.random() < N_SPOTCHECK / self._seen_for_reservoir:
            self.spot.pop(next(iter(self.spot)))
            self.spot[oid] = cube.copy()

        self.row_of[oid] = len(self.rows)
        field, ra, dec = p["info"]
        self.rows.append({
            "object_id_hsc": oid, "field": field, "ra": ra, "dec": dec,
            **{f"has_{b}": has[b] for b in BANDS},
            **{f"psf_fwhm_{b}": psf[i] for i, b in enumerate(BANDS)},
            **{f"n_input_{b}": ninp[i] for i, b in enumerate(BANDS)},
        })
        self.bin_file.write(cube.astype(np.float16).tobytes())

    def flush_untouched(self, touched):
        for oid in [o for o in self.pending if o not in touched]:
            self.finalize(oid)

    def flush_all(self):
        for oid in list(self.pending):
            self.finalize(oid)
        self.bin_file.close()


def main():
    out_build = OUT_DIR + ".building"
    if os.path.exists(OUT_DIR):
        sys.exit(f"{OUT_DIR} already exists — refusing to overwrite. "
                 "Move it aside first.")
    os.makedirs(out_build, exist_ok=True)

    shards = [s for grp in SHARD_GROUPS for s in grp]
    print(f"{len(shards)} shards; building into {out_build}", flush=True)

    b = Builder(out_build)
    cols = ["hsc_object_id", "band", "flux", "psf_fwhm", "n_input_images",
            "field", "ra", "dec"]
    for k, shard in enumerate(shards):
        df = pq.read_table(shard, columns=cols).to_pandas()
        touched = set()
        for _, row in df.iterrows():
            oid = int(row["hsc_object_id"])
            if oid in b.row_of:      # already written (run1 duplicate etc.)
                continue
            b.add_row(oid, row)
            touched.add(oid)
        del df
        # an object's band rows can straddle two consecutive shards (the
        # extraction flushed sequentially), so only finalize objects that
        # received nothing from the current shard
        b.flush_untouched(touched)
        print(f"[{k + 1}/{len(shards)}] {os.path.basename(shard)}: "
              f"written={len(b.rows)} pending={len(b.pending)}", flush=True)
    b.flush_all()

    meta = pd.DataFrame(b.rows)
    # attach DP2 match distance from the paired index (best match per object)
    idx = pd.read_parquet(PAIRED_INDEX)
    if "_dist_arcsec" in idx.columns:
        dist = (idx.sort_values("_dist_arcsec")
                .drop_duplicates("hsc_object_id")
                .set_index("hsc_object_id")["_dist_arcsec"])
        meta["match_dist_arcsec"] = meta["object_id_hsc"].map(dist).astype("float32")
    meta.to_parquet(os.path.join(out_build, "lsst_meta.parquet"), index=False)

    # parent row map + fingerprint
    parent = pd.read_parquet(os.path.join(PARENT_DIR, "catalog.parquet"),
                             columns=["object_id_hsc"])
    fingerprint = hashlib.sha256(
        "\n".join(parent["object_id_hsc"].tolist()).encode()).hexdigest()
    row_map = np.full(len(parent), -1, dtype=np.int64)
    nonempty = parent["object_id_hsc"] != ""
    mapped = parent.loc[nonempty, "object_id_hsc"].astype("int64").map(b.row_of)
    row_map[np.where(nonempty)[0]] = mapped.fillna(-1).astype("int64").to_numpy()
    np.save(os.path.join(out_build, "lsst_row_map.npy"), row_map)

    # ------------------------------------------------------------ validation
    M = len(meta)
    mm = np.memmap(b.bin_path, dtype="float16", mode="r",
                   shape=(M, len(BANDS), STAMP, STAMP))
    assert not np.isinf(mm[:: max(M // 200, 1)]).any(), "inf in sampled rows"
    for oid, cube32 in b.spot.items():
        stored = np.asarray(mm[b.row_of[oid]], dtype=np.float32)
        ref = cube32.astype(np.float16).astype(np.float32)
        same = np.isclose(stored, ref, equal_nan=True).all()
        assert same, f"spot-check mismatch for {oid}"
    print(f"spot-check OK on {len(b.spot)} objects")

    per_field = meta.groupby("field").agg(
        objects=("object_id_hsc", "size"),
        with_u=("has_u", "sum"), with_y=("has_y", "sum"))
    n_mapped = int((row_map >= 0).sum())
    stats = {
        "M": M, "shape": [M, len(BANDS), STAMP, STAMP], "dtype": "float16",
        "flux_unit": "nanomaggy", "njy_per_nanomaggy": NJY_PER_NMGY,
        "bands": BANDS, "core_bands": list(CORE),
        "missing_band_fill": "NaN", "pixel_scale_arcsec": 0.2,
        "parent_catalog": os.path.join(PARENT_DIR, "catalog.parquet"),
        "parent_rows": len(parent),
        "parent_object_id_hsc_sha256": fingerprint,
        "parent_rows_mapped": n_mapped,
        "clipped_pixels": b.n_clipped_px,
        "objects_excluded_incomplete_griz": b.n_incomplete,
        "source_shards": [os.path.relpath(s, CM) for s in shards],
        "build_date": str(date.today()),
        "built_by": "galaxy_images/galaxy_model/lsst/build_lsst_efficient.py",
    }
    with open(os.path.join(out_build, "meta.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_build, "README.md"), "w") as f:
        f.write(__doc__.replace("Build the LSST DP2 efficient store: "
                                "data/lsst_efficient/.", "# lsst_efficient"))

    print("\nper field:")
    print(per_field.to_string())
    print(f"\nM={M} objects  ({b.n_incomplete} excluded, incomplete griz)")
    print(f"parent rows mapped: {n_mapped}/{len(parent)}")
    print(f"clipped pixels (>float16 max): {b.n_clipped_px}")
    print(f"bin size: {os.path.getsize(b.bin_path) / 1e9:.1f} GB")

    os.rename(out_build, OUT_DIR)
    print(f"\nvalidated and renamed -> {OUT_DIR}")


if __name__ == "__main__":
    main()
