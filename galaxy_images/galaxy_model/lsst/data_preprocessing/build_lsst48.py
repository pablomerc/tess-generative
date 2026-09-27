"""Build the 48x48 model-ready LSST store: data/lsst_efficient/lsst48.bin.

WHAT THIS IS (and is not)
-------------------------
Step 2 of the HSC x LSST plan (see ../DECISIONS.md, D5). It produces the LSST
counterpart of data/efficient_neighs_48/{hsc48,legacy48}.bin: every LSST
store row preprocessed ONCE to the tensor the model consumes (griz, 48 px,
clamp -> arcsinh -> zoom x0.84 to HSC's 8.0" field -> normalised), stored as
float16 so the whole thing (65,163 x 4 x 48 x 48 x 2 B = 1.2 GB) sits in RAM
during training and __getitem__ is pure indexing.

It does NOT duplicate neighbour images and it is not "the dataset". The
three pieces are:
  1. lsst_neighbors.npz   WHICH rows are an anchor's neighbours  (step 1)
  2. lsst48.bin           the PIXELS of every row, stored once     (this file)
  3. the Dataset class    assembles (target, samegal, sameins) per sample by
                          indexing 1 and 2 plus the HSC/Legacy 48 px store
                          through parent_row                     (step 3)

Row space: identical to lsst_image.bin / lsst_meta.parquet / lsst_neighbors
(row r here is row r there), so no re-indexing anywhere. All 65,163 rows are
built, not only COSMOS: the field restriction (D1) and the n_input >= 5 cut
(D3) are applied by the Dataset class as masks, so this store never has to
be rebuilt when either changes.

Preprocessing is exactly `preprocessing.preprocess_lsst(bands=griz)`, the
fixture-tested function, so a tensor read from lsst48.bin is bit-for-bit
what the runtime path would have produced (the self-check below verifies
that on random rows after writing).

OUTPUT
------
  data/lsst_efficient/lsst48.bin    float16 (M, 4, 48, 48), bands g,r,i,z
  data/lsst_efficient/lsst48.json   shape, bands, crop, norm/clamp/zoom
                                    constants used, store fingerprint,
                                    per-field mean/std of the result
  figures/lsst48_gallery.png        (if lsst_neighbors.npz exists) random
                                    COSMOS anchors passing griz>=5, shown as
                                    HSC 48 | LSST 48 | 5 LSST neighbours —
                                    i.e. exactly what the two encoders see.

RUN (login node is fine: single process, sequential reads, a few minutes)
    cd tess-generative && conda activate torchenv
    PYTHONPATH=. python -m galaxy_images.galaxy_model.lsst.data_preprocessing.build_lsst48
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date

# Same thread pinning as contrastive_baseline/build_ram48.py: torch/BLAS
# thread pools on the login node trip its process limits, and the work here
# is I/O-bound anyway.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from galaxy_images.galaxy_model.lsst import preprocessing as pp

# ----------------------------------------------------------------- constants
STORE_DIR = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
PARENT48_DIR = "/work1/jeroenaudenaert/pablomer/data/efficient_neighs_48"
OUT_BIN = os.path.join(STORE_DIR, "lsst48.bin")
OUT_JSON = os.path.join(STORE_DIR, "lsst48.json")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BANDS = ("g", "r", "i", "z")     # store order is g,r,i,z,u,y -> slice [:4]
CROP = 48
CHUNK = 512                       # rows per read: 512 x 6 x 160 x 160 x 2 B = 157 MB
N_SPOTCHECK = 32
STAMP = 160


def store_fingerprint(meta: pd.DataFrame) -> str:
    """sha256 of the store's object_id_hsc column, same recipe as the
    neighbour index, so a loader can assert that bin, index and meta agree."""
    return hashlib.sha256("\n".join(meta["object_id_hsc"].astype(str).tolist()).encode()).hexdigest()


# --------------------------------------------------------------------- build
def build(force: bool) -> dict:
    with open(os.path.join(STORE_DIR, "meta.json")) as f:
        smeta = json.load(f)
    M, C, H, W = smeta["shape"]
    assert smeta["bands"][:4] == list(BANDS), smeta["bands"]
    assert smeta["flux_unit"] == "nanomaggy", smeta["flux_unit"]
    meta = pd.read_parquet(os.path.join(STORE_DIR, "lsst_meta.parquet"),
                           columns=["object_id_hsc", "field"] + [f"n_input_{b}" for b in BANDS])

    if os.path.exists(OUT_BIN) and not force:
        raise SystemExit(f"{OUT_BIN} exists; pass --force to rebuild (no silent overwrite)")

    src = np.memmap(os.path.join(STORE_DIR, "lsst_image.bin"), dtype="float16", mode="r",
                    shape=(M, C, H, W))
    tmp_bin = OUT_BIN + ".tmp"
    out = np.memmap(tmp_bin, dtype="float16", mode="w+", shape=(M, len(BANDS), CROP, CROP))

    print(f"[build] {M:,} rows x {BANDS} -> {OUT_BIN}", flush=True)
    t0 = time.time()
    for start in range(0, M, CHUNK):
        stop = min(start + CHUNK, M)
        # Sequential slab read of the griz planes only (the u/y planes are
        # never touched). np.array (not asarray) copies out of the read-only
        # memmap so torch gets a writable buffer and stays quiet; the
        # float16 -> float32 conversion inside preprocess_lsst copies again
        # anyway, so this costs nothing that matters.
        cube = np.array(src[start:stop, :4])                  # (n, 4, 160, 160) f16
        x = pp.preprocess_lsst(cube, bands=BANDS, crop_size=CROP,
                               input_unit="nanomaggy", normalize=True)
        out[start:stop] = x.to(torch.float16).numpy()
        if (stop // CHUNK) % 20 == 0 or stop == M:
            dt = time.time() - t0
            print(f"[build] {stop:,}/{M:,}  {stop / dt:,.0f} rows/s  "
                  f"ETA {(M - stop) / max(stop / dt, 1e-9) / 60:.1f} min", flush=True)
    out.flush()
    del out

    # ------------------------------------------------------------ validation
    chk = np.memmap(tmp_bin, dtype="float16", mode="r", shape=(M, len(BANDS), CROP, CROP))
    assert np.isfinite(np.asarray(chk[:: max(M // 500, 1)], dtype=np.float32)).all(), "non-finite values"
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(M, N_SPOTCHECK, replace=False))
    fresh = pp.preprocess_lsst(np.asarray(src[rows, :4]), bands=BANDS, crop_size=CROP,
                               input_unit="nanomaggy", normalize=True).to(torch.float16).numpy()
    assert np.array_equal(np.asarray(chk[rows]), fresh), "stored rows != fresh preprocess_lsst"
    print(f"[check] {N_SPOTCHECK} random rows reproduce preprocess_lsst bit-for-bit", flush=True)

    per_field = {}
    for f, idx in meta.groupby("field").indices.items():
        sample = np.asarray(chk[np.sort(rng.choice(idx, min(2000, len(idx)), replace=False))],
                            dtype=np.float32)
        per_field[f] = {"n": int(len(idx)),
                        "mean": float(sample.mean()), "std": float(sample.std()),
                        "per_band_mean": [float(v) for v in sample.mean(axis=(0, 2, 3))],
                        "per_band_std": [float(v) for v in sample.std(axis=(0, 2, 3))]}
        print(f"[check] {f:<8} mean {per_field[f]['mean']:+.3f} std {per_field[f]['std']:.3f}  "
              f"(per band std {np.round(per_field[f]['per_band_std'], 3)})", flush=True)
    del chk

    info = {
        "shape": [M, len(BANDS), CROP, CROP], "dtype": "float16", "bands": list(BANDS),
        "row_space": "identical to lsst_image.bin / lsst_meta.parquet / lsst_neighbors.npz",
        "preprocessing": "galaxy_images.galaxy_model.lsst.preprocessing.preprocess_lsst",
        "crop": CROP, "zoom_factor": pp.ZOOM_FACTOR, "norm_key": "".join(BANDS),
        "norm_mean_std": list(pp.LSST_NORM["".join(BANDS)]),
        "clamp_nmgy": {b: pp.LSST_CLAMP[b] for b in BANDS},
        "store_object_id_hsc_sha256": store_fingerprint(meta),
        "per_field": per_field,
        "build_date": str(date.today()),
        "built_by": "galaxy_images/galaxy_model/lsst/data_preprocessing/build_lsst48.py",
        "wall_min": round((time.time() - t0) / 60, 2),
    }
    os.replace(tmp_bin, OUT_BIN)
    with open(OUT_JSON + ".tmp", "w") as f:
        json.dump(info, f, indent=2)
    os.replace(OUT_JSON + ".tmp", OUT_JSON)
    print(f"[build] DONE in {info['wall_min']} min -> {OUT_BIN} "
          f"({os.path.getsize(OUT_BIN) / 1e9:.2f} GB)", flush=True)
    return info


# ------------------------------------------------------------------- gallery
def gallery(out_png: str, n_anchors: int = 6, k: int = 5, min_ninput: int = 5) -> None:
    """Random COSMOS anchors passing the griz >= min_ninput cut, each shown as
    HSC 48 | LSST 48 | k nearest LSST neighbours. This is literally what the
    physics encoder (HSC view of the same galaxy) and the instrument encoder
    (same-survey neighbours) will receive for an LSST anchor."""
    nb_path = os.path.join(STORE_DIR, "lsst_neighbors.npz")
    if not os.path.exists(nb_path):
        print("[gallery] lsst_neighbors.npz not found (run build_lsst_neighbors first); skipping")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = np.load(nb_path)
    idx, parent_row = nb["idx"], nb["parent_row"]
    with open(OUT_JSON) as f:
        info = json.load(f)
    M = info["shape"][0]
    meta = pd.read_parquet(os.path.join(STORE_DIR, "lsst_meta.parquet"),
                           columns=["object_id_hsc", "field"] + [f"n_input_{b}" for b in BANDS])
    # pandas may return a read-only view from to_numpy(), so build the mask
    # out-of-place rather than with an in-place &=.
    ok = (meta["field"] == "COSMOS").to_numpy()
    for b in BANDS:
        ok = ok & (meta[f"n_input_{b}"].to_numpy() >= min_ninput)
    lsst48 = np.memmap(OUT_BIN, dtype="float16", mode="r", shape=(M, 4, CROP, CROP))
    with open(os.path.join(PARENT48_DIR, "meta.json")) as f:
        pm = json.load(f)
    hsc48 = np.memmap(os.path.join(PARENT48_DIR, "hsc48.bin"), dtype="float16", mode="r",
                      shape=tuple(pm["hsc48"]["shape"]))

    def rgb(x):
        """(4,48,48) normalised griz -> display RGB from (i, r, g), per-panel
        percentile stretch. Display only; the model sees the raw tensor."""
        x = np.asarray(x, dtype=np.float32)
        img = np.stack([x[2], x[1], x[0]], axis=-1)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    rng = np.random.default_rng(1)
    anchors = rng.choice(np.where(ok)[0], n_anchors, replace=False)
    ncol = 2 + k
    fig, axes = plt.subplots(n_anchors, ncol, figsize=(1.55 * ncol, 1.65 * n_anchors))
    for r, a in enumerate(anchors):
        panels = [("HSC 48", hsc48[parent_row[a]]), ("LSST 48", lsst48[a])]
        # neighbours: nearest first, skipping any that fail the same cut
        good = [j for j in idx[a] if ok[j]][:k]
        panels += [(f"nb {i + 1}", lsst48[j]) for i, j in enumerate(good)]
        for c in range(ncol):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if c < len(panels):
                ax.imshow(rgb(panels[c][1]), origin="lower")
                if r == 0:
                    ax.set_title(panels[c][0], fontsize=9)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(f"{meta['object_id_hsc'].iloc[a]}\nn_i={meta['n_input_i'].iloc[a]}",
                              fontsize=6.5)
    fig.suptitle(f"COSMOS anchors (griz n_input ≥ {min_ninput}): physics-encoder view (HSC), "
                 f"LSST target, and the {k} nearest same-survey neighbours (instrument encoder)",
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[gallery] -> {out_png}")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite an existing lsst48.bin")
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--gallery-only", action="store_true", help="skip the build, just draw the gallery")
    args = ap.parse_args()
    if not args.gallery_only:
        build(args.force)
    if not args.no_gallery:
        gallery(os.path.join(FIG_DIR, "lsst48_gallery.png"))


if __name__ == "__main__":
    main()
