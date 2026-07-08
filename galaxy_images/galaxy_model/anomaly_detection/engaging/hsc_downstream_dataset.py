"""Dataset adapter for the HSC downstream dataset (memmap + parquet).

Dataset directory layout (see README.md / meta.json):
    /home/pablomer/orcd/scratch/hsc_downstream/
        meta.json          # shape, dtype, units
        catalog.parquet    # 474,954 rows; row i corresponds to record i in .bin files
        hsc_flux.bin       # float16, (N, 5, 160, 160), nJy
        hsc_ivar.bin       # float32, (N, 5, 160, 160), 1/nJy^2  (UNUSED here)
        hsc_mask.bin       # bool,    (N, 5, 160, 160)           (UNUSED here)

Two views:
  view="ours"  -> (4, 48, 48) float32 after preprocess_raw_image(survey="hsc",
                  crop_size=48) then [:4] (drop y-band).
                  Identical pipeline to NeighborsSimpleDataset.__getitem__ used
                  at training time (neighbors.py:534-537).
  view="aion"  -> (5, 160, 160) float32 raw flux, matching what AION's
                  CodecManager expects via HSCImage(flux=...).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# Allow `from neighbors import ...` from this nested location.
_HERE = Path(__file__).resolve().parent
_GALAXY_MODEL = _HERE.parent.parent
if str(_GALAXY_MODEL) not in sys.path:
    sys.path.insert(0, str(_GALAXY_MODEL))

from neighbors import preprocess_raw_image  # noqa: E402

DEFAULT_ROOT = "/home/pablomer/orcd/scratch/hsc_downstream"


class HSCDownstreamDataset(Dataset):
    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        view: str = "ours",
        crop_size: int = 48,
    ):
        if view not in ("ours", "aion"):
            raise ValueError(f"view must be 'ours' or 'aion', got {view!r}")
        self.root = Path(root)
        self.view = view
        self.crop_size = crop_size

        with open(self.root / "meta.json") as f:
            meta = json.load(f)

        pix = meta["pixels"]
        self.shape = tuple(pix["shape"])  # (N, 5, 160, 160)
        self.n_records = int(self.shape[0])

        flux_dtype = np.dtype(pix["files"]["flux"]["dtype"])  # float16
        self.flux = np.memmap(
            self.root / pix["files"]["flux"]["path"],
            dtype=flux_dtype,
            mode="r",
            shape=self.shape,
        )

        # Catalog: only RA, Dec, object_id needed for downstream dedup/labels.
        cat = pq.read_table(
            self.root / meta["catalog"], columns=["ra", "dec", "object_id"]
        )
        self.ra = cat["ra"].to_numpy()
        self.dec = cat["dec"].to_numpy()
        self.object_id = cat["object_id"].to_numpy()

        if len(self.ra) != self.n_records:
            raise RuntimeError(
                f"catalog rows ({len(self.ra)}) != flux records ({self.n_records})"
            )

    def __len__(self) -> int:
        return self.n_records

    def _load_flux(self, idx: int) -> np.ndarray:
        # Copy out of the memmap and upcast float16 -> float32.
        return np.asarray(self.flux[idx], dtype=np.float32)

    def __getitem__(self, idx: int):
        img = self._load_flux(idx)  # (5, 160, 160) float32

        if self.view == "ours":
            # Same pipeline NeighborsSimpleDataset uses at training time.
            t = preprocess_raw_image(
                img, survey="hsc", crop_size=self.crop_size
            )  # (5, crop, crop) torch.float32
            t = t[:4]  # drop y-band
            return t, int(idx)

        # AION view: hand AION's HSCImage the raw 5-band flux directly.
        t = torch.from_numpy(img)  # (5, 160, 160) float32
        return t, int(idx)


if __name__ == "__main__":
    # Quick smoke test.
    for v in ("ours", "aion"):
        d = HSCDownstreamDataset(view=v)
        x, raw = d[0]
        print(f"view={v:5s}  N={len(d)}  x.shape={tuple(x.shape)}  dtype={x.dtype}  raw_index={raw}")
        # Also pull the last index to verify memmap works at the tail.
        x2, raw2 = d[len(d) - 1]
        print(f"  last:    x.shape={tuple(x2.shape)}  raw_index={raw2}")
        print(f"  ra[0]={d.ra[0]:.6f} dec[0]={d.dec[0]:+.6f} oid[0]={d.object_id[0]}")
