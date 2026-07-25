"""
One-time builder: preprocess ALL rows of the efficient_neighs store into a compact
48x48, 4-band float16 store that fits in RAM. This is the fix for the fact that the
160x160 mmap store is Ceph random-read bound (~0.35 batch/s -> ~78h/run).

Output layout (dir = --out):
  hsc48.bin      float16 (N, 4, 48, 48)   preprocess_raw_image(hsc)[:4]   (y dropped)
  legacy48.bin   float16 (N, 4, 48, 48)   preprocess_raw_image(legacy)   (zoomed+normed)
  catalog.parquet   copied from source (source_type + neighbor_idx_* preserved)
  meta.json         {n_records, shapes}

Reads the source SEQUENTIALLY (shuffle=False) which is fast on Ceph, unlike the
random access training would do on the 160x160 store. Preprocessing is identical to
NeighborsEfficientDataset._preprocess so downstream tensors are bit-for-bit the same.
"""
import argparse, os, sys, json, time, shutil
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
# Derive the repo root from this file so the builder runs on any cluster (it used to
# hardcode the Engaging path, which made it unusable on AMD).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from galaxy_images.galaxy_model.neighbors import preprocess_raw_image

# Engaging defaults, kept so existing no-argument invocations behave exactly as before.
SRC = "/orcd/pool/007/pablomer/efficient_neighs"
OUT = "/orcd/pool/007/pablomer/efficient_neighs_48"
CROP = 48


class _RowPreprocessor(Dataset):
    """Returns (idx, hsc48[4,48,48], legacy48[4,48,48]) for source row idx."""
    def __init__(self, src):
        self.src = src
        with open(os.path.join(src, "meta.json")) as f:
            meta = json.load(f)
        self.hsc_shape = tuple(meta["hsc_image"]["shape"])       # (N,5,160,160)
        self.legacy_shape = tuple(meta["legacy_image"]["shape"]) # (N,4,160,160)
        self.n = self.hsc_shape[0]
        self._hsc = None
        self._leg = None

    def _open(self):
        if self._hsc is None:
            self._hsc = np.memmap(os.path.join(self.src, "hsc_image.bin"),
                                  dtype="float16", mode="r", shape=self.hsc_shape)
            self._leg = np.memmap(os.path.join(self.src, "legacy_image.bin"),
                                  dtype="float16", mode="r", shape=self.legacy_shape)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self._open()
        hsc_raw = torch.from_numpy(self._hsc[i].astype(np.float32))     # (5,160,160)
        leg_raw = torch.from_numpy(self._leg[i].astype(np.float32))     # (4,160,160)
        hsc48 = preprocess_raw_image(hsc_raw, survey="hsc", crop_size=CROP)[:4]  # (4,48,48)
        leg48 = preprocess_raw_image(leg_raw, survey="legacy", crop_size=CROP)   # (4,48,48)
        return i, hsc48.to(torch.float16).numpy(), leg48.to(torch.float16).numpy()


def main():
    global SRC, OUT, CROP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC, help=f"source efficient store (default: {SRC})")
    ap.add_argument("--out", default=OUT, help=f"destination 48x48 store (default: {OUT})")
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    SRC, OUT, CROP = args.src, args.out, args.crop

    os.makedirs(OUT, exist_ok=True)
    ds = _RowPreprocessor(SRC)
    n = len(ds)
    print(f"[build] {n} rows -> {OUT} (crop={CROP}, workers={args.workers})", flush=True)

    hsc_out = np.memmap(os.path.join(OUT, "hsc48.bin"), dtype="float16", mode="w+",
                        shape=(n, 4, CROP, CROP))
    leg_out = np.memmap(os.path.join(OUT, "legacy48.bin"), dtype="float16", mode="w+",
                        shape=(n, 4, CROP, CROP))

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    prefetch_factor=4, collate_fn=lambda b: b)  # list of tuples

    done = 0
    t0 = time.time()
    for batch in dl:
        for i, h, l in batch:
            hsc_out[i] = h
            leg_out[i] = l
        done += len(batch)
        if done % 20480 == 0:
            dt = time.time() - t0
            print(f"[build] {done}/{n}  ({done/dt:.0f} rows/s, ETA {(n-done)/(done/dt)/60:.1f} min)",
                  flush=True)
    hsc_out.flush(); leg_out.flush()
    del hsc_out, leg_out

    # Copy catalog (neighbor indices + source_type) and write meta.
    shutil.copy(os.path.join(SRC, "catalog.parquet"), os.path.join(OUT, "catalog.parquet"))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump({"n_records": n,
                   "hsc48": {"shape": [n, 4, CROP, CROP], "dtype": "float16"},
                   "legacy48": {"shape": [n, 4, CROP, CROP], "dtype": "float16"}}, f)
    print(f"[build] DONE {n} rows in {(time.time()-t0)/60:.1f} min", flush=True)
    print("BUILD_RAM48_OK", flush=True)


if __name__ == "__main__":
    main()
