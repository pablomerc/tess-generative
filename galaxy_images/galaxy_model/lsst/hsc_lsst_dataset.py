"""HSC x LSST triplet dataset, in RAM, for the dual-encoder models.

WHAT IT PRODUCES
----------------
Exactly the contract of NeighborsEfficientDataset / NeighborsRAM48Dataset,
so `collate_neighbors`, `ConditionalFlowMatchingModule` and the contrastive
baseline consume it unchanged:

    __getitem__(i) -> (target, samegal, sameins, metadata)
        target   (4, 48, 48)  the anchor galaxy in the anchor survey
        samegal  (4, 48, 48)  the SAME galaxy in the other survey
        sameins  (k, 48, 48)  k OTHER galaxies from the anchor's survey
                              (k <= max_neighbors; may be 0)
        metadata {"anchor_survey": "hsc" | "lsst", "idx": <lsst store row>,
                  "parent_row": <parent catalogue row>,
                  "num_same_instrument": k}

Anchor survey alternates by position (even -> HSC, odd -> LSST), as in
NeighborsRAM48Dataset, so a seeded random_split still gives ~50/50.

WHERE EVERYTHING COMES FROM  (all row-aligned stores, nothing is copied twice)
--------------------------------------------------------------------------------
  LSST pixels      data/lsst_efficient/lsst48.bin        (M, 4, 48, 48) f16, store rows
  LSST neighbours  data/lsst_efficient/lsst_neighbors.npz idx (M, 20) store rows,
                                                          parent_row (M,)
  LSST metadata    data/lsst_efficient/lsst_meta.parquet  field, n_input_<b>, psf_fwhm_<b>
  HSC pixels       data/efficient_neighs_48/hsc48.bin     (N_parent, 4, 48, 48) f16,
                                                          PARENT catalogue rows
  HSC neighbours   data/efficient_neighs_48/catalog.parquet neighbor_idx_hsc (N_parent, 20)

An LSST store row r is the galaxy whose HSC image is hsc48[parent_row[r]];
its HSC neighbours are neighbor_idx_hsc[parent_row[r]] (parent rows, into
hsc48); its LSST neighbours are lsst_neighbors.idx[r] (store rows, into
lsst48). Legacy is not loaded: this is the two-view HSC x LSST experiment.

THE SELECTION (DECISIONS.md D1-D3) is applied HERE, as masks, not baked into
any store, so changing field / bands / threshold never triggers a rebuild:
  * field == "COSMOS"                                             (D1)
  * bands g,r,i,z  -- both stores are already griz at 48 px       (D2)
  * n_input_<b> >= min_n_input for every band in g,r,i,z          (D3)
    applied to anchors AND to LSST neighbour slots (a near-empty
    stamp in the instrument encoder is worse than an empty slot).
HSC neighbour slots have no such notion and are taken as the parent
catalogue gives them (drop -1 padding).

MEMORY. By default only the parent rows this selection can ever touch are
kept in RAM (COSMOS: ~165k of 468k rows -> ~3 GB instead of 8.6 GB); a
remap array turns parent rows into compact positions. lsst48 (1.2 GB) is
loaded whole. Forked DataLoader workers share both via copy-on-write.
Login-node caveat: `ulimit -v` there is 32 GB; this fits.

TRAIN/VAL SPLIT is NOT done here (decision 2026-09-27: a single seeded 5 %
validation slice, no downstream-holdout exclusion for now). data_factory's
seeded random_split over dataset positions does it, exactly as for the
other modes. `val_object_ids(positions)` maps split positions back to
object ids if a run needs to freeze or publish its split.

SELF-TEST:  PYTHONPATH=. python -m galaxy_images.galaxy_model.lsst.hsc_lsst_dataset
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LSST_DIR_DEFAULT = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
HSC48_DIR_DEFAULT = "/work1/jeroenaudenaert/pablomer/data/efficient_neighs_48"
BANDS = ("g", "r", "i", "z")
CROP = 48


class HscLsstRAM48Dataset(Dataset):
    def __init__(
        self,
        lsst_dir: str | Path | None = None,     # None -> LSST_DIR_DEFAULT
        hsc48_dir: str | Path | None = None,    # None -> HSC48_DIR_DEFAULT
        field: str | None = "COSMOS",
        min_n_input: int = 5,
        max_neighbors: int = 5,
        random_neighbors: bool = False,
        subset_hsc_rows: bool = True,
        crop_size: int = CROP,          # API-compat with the other datasets; stores are 48
        verbose: bool = True,
    ):
        lsst_dir = Path(lsst_dir or LSST_DIR_DEFAULT)
        hsc48_dir = Path(hsc48_dir or HSC48_DIR_DEFAULT)
        if crop_size != CROP:
            raise ValueError(f"stores are {CROP} px; got crop_size={crop_size}")
        self.max_neighbors = int(max_neighbors)
        self.random_neighbors = bool(random_neighbors)
        self.field, self.min_n_input = field, int(min_n_input)

        # ---------------------------------------------------- LSST side
        with open(lsst_dir / "lsst48.json") as f:
            l48 = json.load(f)
        M = l48["shape"][0]
        assert tuple(l48["bands"]) == BANDS, l48["bands"]
        nb = np.load(lsst_dir / "lsst_neighbors.npz")
        with open(lsst_dir / "lsst_neighbors.json") as f:
            nbj = json.load(f)
        # bin, index and meta must describe the same rows: compare fingerprints
        assert nbj["store_object_id_hsc_sha256"] == l48["store_object_id_hsc_sha256"], \
            "lsst_neighbors.npz and lsst48.bin were built against different stores"
        self.lsst_nb = nb["idx"].astype(np.int64)               # (M, 20) store rows, -1 pad
        self.parent_row = nb["parent_row"].astype(np.int64)     # (M,)  parent rows
        assert self.lsst_nb.shape[0] == M and self.parent_row.shape[0] == M

        meta = pd.read_parquet(lsst_dir / "lsst_meta.parquet")
        assert len(meta) == M
        self.object_id_hsc = meta["object_id_hsc"].to_numpy()
        n_input = np.stack([meta[f"n_input_{b}"].to_numpy() for b in BANDS], axis=1)  # (M, 4)
        ok = np.all(n_input >= self.min_n_input, axis=1)                             # D3
        if field is not None:
            ok &= (meta["field"] == field).to_numpy()                                # D1
        self.ok = ok                                   # rows usable as anchor OR LSST neighbour
        self.anchor_rows = np.where(ok)[0].astype(np.int64)     # dataset position -> store row

        # np.fromfile is a real read into RAM (np.array on a memmap would not be).
        self.lsst = np.fromfile(lsst_dir / "lsst48.bin", dtype="float16").reshape(M, 4, CROP, CROP)

        # ----------------------------------------------------- HSC side
        with open(hsc48_dir / "meta.json") as f:
            h48 = json.load(f)
        N = int(h48["n_records"])
        cat = pd.read_parquet(hsc48_dir / "catalog.parquet", columns=["neighbor_idx_hsc"])
        assert len(cat) == N
        hsc_nb_full = np.stack(cat["neighbor_idx_hsc"].to_numpy()).astype(np.int64)  # (N, 20)
        self.hsc_nb = hsc_nb_full[self.parent_row]              # (M, 20) parent rows, -1 pad
        del hsc_nb_full

        if subset_hsc_rows:
            # Every parent row the selection can touch: anchors' own rows plus
            # all their HSC neighbours. Load only those, remap to 0..P-1.
            touch = np.concatenate([self.parent_row[self.anchor_rows],
                                    self.hsc_nb[self.anchor_rows].ravel()])
            touch = np.unique(touch[touch >= 0])
            self._hsc_remap = np.full(N, -1, dtype=np.int64)
            self._hsc_remap[touch] = np.arange(len(touch))
            self.hsc = np.empty((len(touch), 4, CROP, CROP), dtype="float16")
            # seek+read rather than np.memmap: mapping the 8.6 GB file needs that much
            # virtual address space, and on the login node (ulimit -v 32 GB) the ROCm
            # runtime has already reserved most of it once torch is imported. `touch`
            # is sorted, and the parent store is Hilbert-ordered, so the reads are
            # nearly sequential. Rows are read straight into the preallocated array.
            rowbytes = 4 * CROP * CROP * 2
            with open(hsc48_dir / "hsc48.bin", "rb", buffering=0) as fh:
                for i, r in enumerate(touch):
                    fh.seek(int(r) * rowbytes)
                    fh.readinto(self.hsc[i])
            self._hsc_pool_size = len(touch)
        else:
            self._hsc_remap = None
            self.hsc = np.fromfile(hsc48_dir / "hsc48.bin", dtype="float16").reshape(N, 4, CROP, CROP)
            self._hsc_pool_size = N

        if verbose:
            gb = (self.hsc.nbytes + self.lsst.nbytes) / 1e9
            print(f"[HscLsstRAM48Dataset] field={field} min_n_input={min_n_input}: "
                  f"{len(self.anchor_rows):,} anchors of {M:,} LSST rows | HSC pool "
                  f"{self._hsc_pool_size:,} rows | {gb:.1f} GB in RAM | "
                  f"max_neighbors={max_neighbors} random_neighbors={self.random_neighbors}", flush=True)

    # ------------------------------------------------------------ helpers
    def _hsc_img(self, parent_r: int) -> np.ndarray:
        p = parent_r if self._hsc_remap is None else self._hsc_remap[parent_r]
        assert p >= 0, f"parent row {parent_r} not in the loaded HSC subset"
        return self.hsc[p]

    def _lsst_neighbours(self, row: int) -> np.ndarray:
        if self.random_neighbors:
            return np.random.choice(self.anchor_rows, size=self.max_neighbors, replace=False)
        c = self.lsst_nb[row]
        c = c[(c >= 0)]
        c = c[self.ok[c]]                     # D3 on neighbour slots
        return c[: self.max_neighbors]

    def _hsc_neighbours(self, row: int) -> np.ndarray:
        if self.random_neighbors:
            pool = self.parent_row[self.anchor_rows]      # random rows among the same anchors
            return np.random.choice(pool, size=self.max_neighbors, replace=False)
        c = self.hsc_nb[row]
        return c[c >= 0][: self.max_neighbors]

    def val_object_ids(self, positions) -> np.ndarray:
        """Dataset positions (e.g. a Subset's indices) -> HSC object ids."""
        return self.object_id_hsc[self.anchor_rows[np.asarray(positions)]]

    # -------------------------------------------------------------- API
    def __len__(self) -> int:
        return len(self.anchor_rows)

    def __getitem__(self, idx: int):
        row = int(self.anchor_rows[idx])
        prow = int(self.parent_row[row])
        anchor_is_hsc = (idx % 2 == 0)

        lsst_img = torch.from_numpy(self.lsst[row].astype(np.float32))
        hsc_img = torch.from_numpy(self._hsc_img(prow).astype(np.float32))
        if anchor_is_hsc:
            target, samegal, survey = hsc_img, lsst_img, "hsc"
            nb = [torch.from_numpy(self._hsc_img(int(r)).astype(np.float32)) for r in self._hsc_neighbours(row)]
        else:
            target, samegal, survey = lsst_img, hsc_img, "lsst"
            nb = [torch.from_numpy(self.lsst[int(r)].astype(np.float32)) for r in self._lsst_neighbours(row)]

        sameins = torch.stack(nb, 0) if nb else torch.empty(0, 4, CROP, CROP)
        metadata = {"anchor_survey": survey, "idx": row, "parent_row": prow,
                    "num_same_instrument": len(nb)}
        return target, samegal, sameins, metadata


# ------------------------------------------------------------------ self-test
def _self_test() -> None:
    from galaxy_images.galaxy_model.neighbors import collate_neighbors, preprocess_raw_image

    t0 = time.time()
    ds = HscLsstRAM48Dataset()
    print(f"[test] constructed in {time.time() - t0:.1f} s; len={len(ds):,}; "
          f"5% val would be {int(0.05 * len(ds)):,}")

    # contract + timing
    items = [ds[i] for i in range(64)]
    tgt, sg, si, masks, meta = collate_neighbors(items)
    assert tgt.shape == (64, 4, CROP, CROP) and sg.shape == tgt.shape
    assert si.shape[0] == 64 and si.shape[2:] == (4, CROP, CROP) and masks.shape[:2] == si.shape[:2]
    surveys = [m["anchor_survey"] for m in meta]
    assert surveys[0] == "hsc" and surveys[1] == "lsst"
    assert all(torch.isfinite(t).all() for t in (tgt, sg, si))
    print(f"[test] collate OK: sameins {tuple(si.shape)}, neighbours per item "
          f"min {masks.sum(1).min().item()} / median {masks.sum(1).float().median().item():.0f}")
    t1 = time.time(); [ds[i] for i in range(1000, 1500)]
    print(f"[test] {(time.time() - t1) / 500 * 1e3:.2f} ms per sample")

    # same galaxy? the HSC image must be the parent row of the LSST row
    i = 3; row = int(ds.anchor_rows[i]); prow = int(ds.parent_row[row])
    cat = pd.read_parquet(Path(HSC48_DIR_DEFAULT) / "catalog.parquet", columns=["object_id_hsc"])
    assert str(cat["object_id_hsc"].iloc[prow]) == str(ds.object_id_hsc[row]), "parent_row bridge broken"
    print("[test] parent_row bridge OK (same object id on both sides)")

    # the July hsc48 store vs today's runtime preprocessing (DECISIONS D5 check)
    with open(Path(HSC48_DIR_DEFAULT).parent / "neighbors_efficient" / "meta.json") as f:
        pm = json.load(f)
    shp = tuple(pm["hsc_image"]["shape"]) if "hsc_image" in pm else (468197, 5, 160, 160)
    rowbytes = int(np.prod(shp[1:])) * 2
    rng = np.random.default_rng(0)
    rows = rng.choice(ds.parent_row[ds.anchor_rows], 8, replace=False)
    maxdiff = 0.0
    with open(Path(HSC48_DIR_DEFAULT).parent / "neighbors_efficient" / "hsc_image.bin", "rb") as fh:
        for r in rows:
            fh.seek(int(r) * rowbytes)
            raw = np.frombuffer(fh.read(rowbytes), dtype=np.float16).reshape(shp[1:]).astype(np.float32)
            fresh = preprocess_raw_image(torch.from_numpy(raw), survey="hsc", crop_size=CROP)[:4]
            fresh = fresh.to(torch.float16).float().numpy()
            stored = ds._hsc_img(int(r)).astype(np.float32)
            maxdiff = max(maxdiff, float(np.abs(fresh - stored).max()))
    print(f"[test] hsc48 store vs fresh preprocess_raw_image on 8 rows: max |diff| = {maxdiff:.3g} "
          f"({'identical' if maxdiff == 0 else 'DIFFERS — preprocessing changed since July, rebuild hsc48'})")
    print("SELF_TEST_OK" if maxdiff < 1e-3 else "SELF_TEST_FAILED")


if __name__ == "__main__":
    _self_test()
