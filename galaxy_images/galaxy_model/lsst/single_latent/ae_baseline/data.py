"""Dataset for the LSST AE baseline: grizy stamps from data/lsst_efficient/.

- 64,918 objects with has_y (grizy complete; DEEP2-3 keeps y, lacks only u).
- Deterministic 90/10 split by sha1(object_id_hsc) — id-stable across reruns,
  reusable for later AE-vs-contrastive comparisons (val ids saved by train.py).
- Rows are fetched with seek+read (no memmap: login-node address-space safe),
  one lazily opened file handle per dataloader worker.
- Preprocessing: lsst.preprocessing.preprocess_lsst grizy -> (5, 48, 48).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from galaxy_images.galaxy_model.lsst import preprocessing as pp

STORE = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
BANDS = ("g", "r", "i", "z", "y")
BAND_IDX = [pp.STORE_BANDS.index(b) for b in BANDS]        # [0,1,2,3,5]
FIELDS = ("COSMOS", "DEEP2-3")
FIELD_ARG = {"all": None, "cosmos": "COSMOS", "deep23": "DEEP2-3"}


def _val_mask(ids: np.ndarray, mod: int = 10) -> np.ndarray:
    def bucket(oid):
        h = hashlib.sha1(str(int(oid)).encode()).digest()
        return int.from_bytes(h[:8], "big") % mod
    return np.array([bucket(o) == 0 for o in ids])


class LsstGrizyDataset(Dataset):
    """split: 'train' | 'val' | 'all'; field: None/'all' | 'COSMOS' | 'DEEP2-3'
    (field filter applied before the hash split, so per-field train/val sets
    are exact subsets of the mixed-field ones)."""

    def __init__(self, split: str = "train", store: str = STORE,
                 field: str | None = None):
        self.store = store
        with open(os.path.join(store, "meta.json")) as f:
            self.shape = tuple(json.load(f)["shape"])       # (M, 6, 160, 160)
        self.row_bytes = int(np.prod(self.shape[1:])) * 2

        meta = pd.read_parquet(os.path.join(store, "lsst_meta.parquet"))
        meta = meta[meta.has_y]                              # grizy-complete
        if field and field != "all":
            if field not in FIELDS:
                raise ValueError(f"field must be one of {FIELDS} or 'all'")
            meta = meta[meta.field == field]
        ids = meta.object_id_hsc.to_numpy()
        val = _val_mask(ids)
        if split == "train":
            meta, val = meta[~val], val[~val]
        elif split == "val":
            meta, val = meta[val], val[val]
        elif split != "all":
            raise ValueError(split)

        self.rows = meta.index.to_numpy()                    # store row indices
        self.ids = meta.object_id_hsc.to_numpy()
        self.is_val = val
        self.field = np.array([FIELDS.index(f) for f in meta.field])
        self.ninp_i = meta.n_input_i.to_numpy(dtype=np.float32)
        self.psf_i = meta.psf_fwhm_i.to_numpy(dtype=np.float32)
        self._f = None                                       # per-worker handle

    def __len__(self):
        return len(self.rows)

    def raw_cube(self, i: int) -> np.ndarray:
        """(5, 160, 160) float32 nanomaggies, grizy order."""
        if self._f is None:
            self._f = open(os.path.join(self.store, "lsst_image.bin"), "rb")
        self._f.seek(int(self.rows[i]) * self.row_bytes)
        cube = np.frombuffer(self._f.read(self.row_bytes), np.float16)
        return cube.reshape(self.shape[1:])[BAND_IDX].astype(np.float32)

    def __getitem__(self, i: int):
        x = pp.preprocess_lsst(self.raw_cube(i), bands=BANDS)
        return {
            "x": x,
            "idx": i,
            "field": int(self.field[i]),
            "ninp": float(self.ninp_i[i]),
            "psf": float(self.psf_i[i]),
        }
