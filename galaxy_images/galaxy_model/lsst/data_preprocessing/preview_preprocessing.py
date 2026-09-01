#!/usr/bin/env python3
"""Visual acceptance test: LSST vs HSC, raw stamps and model inputs.

For a set of objects (brightest + random, per field) renders four panels each:

  1. LSST original 160x160 (gri composite, 32", asinh stretch),
     cyan box = the 8.0" the model sees
  2. LSST normalized 48x48 model input (this package's preprocess_lsst)
  3. HSC original 160x160 (gri, 26.9", same stretch after zp-27 -> nmgy),
     cyan box = 8.06" (48 px)
  4. HSC normalized 48x48 model input (the EXISTING pipeline:
     galaxy_model.neighbors.preprocess_raw_image, survey="hsc")

Same galaxy on every row -> a crossmatch sanity check (morphology must match)
and a preprocessing-parity check (both model inputs on one display stretch).

HSC images come from the parent store via object_id_hsc -> catalog row ->
hsc_image.bin. Output: figures/preprocessing_preview.png.
  PYTHONPATH=. python galaxy_images/galaxy_model/lsst/data_preprocessing/preview_preprocessing.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from galaxy_images.galaxy_model.lsst import preprocessing as pp
from galaxy_images.galaxy_model.neighbors import preprocess_raw_image

LSST_STORE = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
PARENT = "/work1/jeroenaudenaert/pablomer/data/neighbors_efficient"
N_BRIGHT = 3          # per field: brightest of a candidate pool ...
N_RANDOM = 2          # ... plus typical (random) objects
N_CANDIDATES = 150
SEED = 0
HSC_ZPSCALE = 10.0 ** ((27.0 - 22.5) / 2.5)   # counts -> nanomaggies (63.1)
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figures", "preprocessing_preview.png")


class RowReader:
    def __init__(self, path, shape):
        self.shape = shape                       # per-row shape, e.g. (6,160,160)
        self.row_bytes = int(np.prod(shape)) * 2
        self.f = open(path, "rb")

    def row(self, r):
        self.f.seek(int(r) * self.row_bytes)
        return np.frombuffer(self.f.read(self.row_bytes), np.float16).reshape(
            self.shape).astype(np.float32)


def asinh_rgb(cube_gri_nmgy, soft=0.02):
    hi = np.nanpercentile(cube_gri_nmgy[2], 99.5)
    x = np.arcsinh(np.clip(cube_gri_nmgy, 0, None) / soft)
    x = x / max(np.arcsinh(hi / soft), 1e-6)
    return np.clip(np.stack([x[2], x[1], x[0]], -1), 0, 1)


def norm_rgb(inp_griz):
    rgb = np.stack([inp_griz[2], inp_griz[1], inp_griz[0]], -1)
    return np.clip((rgb + 1.0) / 5.0, 0, 1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with open(os.path.join(LSST_STORE, "meta.json")) as f:
        lshape = tuple(json.load(f)["shape"])[1:]
    lsst = RowReader(os.path.join(LSST_STORE, "lsst_image.bin"), lshape)
    hsc = RowReader(os.path.join(PARENT, "hsc_image.bin"), (5, 160, 160))
    meta = pd.read_parquet(os.path.join(LSST_STORE, "lsst_meta.parquet"))
    i_idx = pp.STORE_BANDS.index("i")

    # object_id_hsc -> parent catalog row (holds the HSC image at same index)
    par = pd.read_parquet(os.path.join(PARENT, "catalog.parquet"),
                          columns=["object_id_hsc"])
    ne = par.object_id_hsc != ""
    parent_row = pd.Series(np.where(ne)[0],
                           index=par.object_id_hsc[ne].astype("int64"))

    rng = np.random.default_rng(SEED)
    picks = []
    for field in ("COSMOS", "DEEP2-3"):
        cand = meta[meta.field == field].sample(N_CANDIDATES,
                                                random_state=SEED)
        flux = {r: float(lsst.row(r)[i_idx, 56:104, 56:104].sum())
                for r in cand.index}
        bright = sorted(flux, key=flux.get, reverse=True)[:N_BRIGHT]
        rest = [r for r in cand.index if r not in bright]
        picks += bright + list(rng.choice(rest, N_RANDOM, replace=False))
    print(f"preview rows: {picks}")

    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 3.2 * n))
    for k, r in enumerate(picks):
        m = meta.loc[r]
        lc = lsst.row(r)                                     # (6,160,160) nmgy
        hrow = int(parent_row.loc[int(m.object_id_hsc)])
        hc = hsc.row(hrow)                                   # (5,160,160) counts
        lsst_in = pp.preprocess_lsst(lc[:4], bands=("g", "r", "i", "z")).numpy()
        hsc_in = preprocess_raw_image(
            torch.as_tensor(hc), survey="hsc", crop_size=48)[:4].numpy()

        ax = axes[k, 0]
        ax.imshow(asinh_rgb(lc[:3]), origin="lower")
        ax.add_patch(Rectangle((60, 60), 40, 40, ec="cyan", fc="none", lw=1.2))
        ax.set_ylabel(f"{m.field}\n{m.object_id_hsc}\n"
                      f"psf_i={m.psf_fwhm_i:.2f}\"  N_i={m.n_input_i:.0f}",
                      fontsize=7)
        axes[k, 1].imshow(norm_rgb(lsst_in), origin="lower",
                          interpolation="nearest")
        ax = axes[k, 2]
        ax.imshow(asinh_rgb(hc[:3] / HSC_ZPSCALE), origin="lower")
        ax.add_patch(Rectangle((56, 56), 48, 48, ec="cyan", fc="none", lw=1.2))
        axes[k, 3].imshow(norm_rgb(hsc_in), origin="lower",
                          interpolation="nearest")
        if k == 0:
            for c, t in enumerate(
                    ["LSST original 160×160 (32\")\ncyan = model FoV 8.0\"",
                     "LSST model input 48×48\n(preprocess_lsst)",
                     "HSC original 160×160 (26.9\")\ncyan = model FoV 8.06\"",
                     "HSC model input 48×48\n(existing preprocess_raw_image)"]):
                axes[0, c].set_title(t, fontsize=9)
        for c in range(4):
            axes[k, c].set_xticks([]), axes[k, c].set_yticks([])
    fig.suptitle("Same galaxy, both surveys: raw stamps and normalized model "
                 "inputs (shared display stretch)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
