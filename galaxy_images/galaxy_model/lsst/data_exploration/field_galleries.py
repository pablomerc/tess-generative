#!/usr/bin/env python3
"""Per-field example galleries (COSMOS and DEEP2-3 separately).

For each field, using the same deterministic objects (4 bright + 4 random):

  figures/three_survey_<FIELD>.png
      raw stamps in all three surveys — LSST DP2 / HSC / Legacy — on a common
      24" field of view (the three-way crossmatch, per field)
  figures/preprocessing_preview_<FIELD>.png
      LSST + HSC before/after preprocessing (first 6 of the same objects),
      same layout as data_preprocessing/preview_preprocessing.py

Images come from the local stores only (lsst_efficient + neighbors_efficient);
Legacy is already in nanomaggies, HSC counts are divided by the zp-27 scale,
so one shared asinh stretch applies to all three.

  PYTHONPATH=. python galaxy_images/galaxy_model/lsst/data_exploration/field_galleries.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from galaxy_images.galaxy_model.lsst import preprocessing as pp
from galaxy_images.galaxy_model.lsst.data_preprocessing.preview_preprocessing import (
    HSC_ZPSCALE, RowReader, asinh_rgb, norm_rgb)
from galaxy_images.galaxy_model.neighbors import preprocess_raw_image

LSST_STORE = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
PARENT = "/work1/jeroenaudenaert/pablomer/data/neighbors_efficient"
HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ("COSMOS", "DEEP2-3")
N_BRIGHT, N_RANDOM, N_CANDIDATES, SEED = 4, 4, 150, 1
FOV = 24.0                                    # arcsec, three-survey panels
SCALES = {"lsst": 0.2, "hsc": 0.168, "legacy": 0.262}


def crop_fov(img, scale, fov=FOV):
    """(C,H,W) -> central fov-arcsec crop at the survey's pixel scale."""
    n = min(int(round(fov / scale)), img.shape[-1])
    c = img.shape[-1] // 2
    h = n // 2
    return img[:, c - h:c + h, c - h:c + h]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with open(os.path.join(LSST_STORE, "meta.json")) as f:
        lshape = tuple(json.load(f)["shape"])[1:]
    lsst = RowReader(os.path.join(LSST_STORE, "lsst_image.bin"), lshape)
    hsc = RowReader(os.path.join(PARENT, "hsc_image.bin"), (5, 160, 160))
    legacy = RowReader(os.path.join(PARENT, "legacy_image.bin"), (4, 160, 160))
    meta = pd.read_parquet(os.path.join(LSST_STORE, "lsst_meta.parquet"))
    i_idx = pp.STORE_BANDS.index("i")

    par = pd.read_parquet(os.path.join(PARENT, "catalog.parquet"),
                          columns=["object_id_hsc"])
    ne = par.object_id_hsc != ""
    parent_row = pd.Series(np.where(ne)[0],
                           index=par.object_id_hsc[ne].astype("int64"))

    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    for field in FIELDS:
        cand = meta[meta.field == field].sample(N_CANDIDATES, random_state=SEED)
        flux = {r: float(lsst.row(r)[i_idx, 56:104, 56:104].sum())
                for r in cand.index}
        bright = sorted(flux, key=flux.get, reverse=True)[:N_BRIGHT]
        rest = [r for r in cand.index if r not in bright]
        picks = bright + list(rng.choice(rest, N_RANDOM, replace=False))

        # --------------------------- figure A: three-survey raw stamps ----
        fig, axes = plt.subplots(len(picks), 3,
                                 figsize=(9.6, 3.05 * len(picks)))
        for k, r in enumerate(picks):
            m = meta.loc[r]
            hrow = int(parent_row.loc[int(m.object_id_hsc)])
            panels = [
                ("LSST DP2", crop_fov(lsst.row(r)[:3], SCALES["lsst"])),
                ("HSC", crop_fov(hsc.row(hrow)[:3] / HSC_ZPSCALE,
                                 SCALES["hsc"])),
                ("Legacy", crop_fov(legacy.row(hrow)[:3], SCALES["legacy"])),
            ]
            for c, (title, cube) in enumerate(panels):
                ax = axes[k, c]
                ax.imshow(asinh_rgb(cube), origin="lower")
                if k == 0:
                    ax.set_title(f"{title} (gri, {FOV:.0f}\")", fontsize=10)
                ax.set_xticks([]), ax.set_yticks([])
            axes[k, 0].set_ylabel(
                f"{m.object_id_hsc}\npsf_i={m.psf_fwhm_i:.2f}\" "
                f"N_i={m.n_input_i:.0f}", fontsize=7)
        fig.suptitle(f"{field}: same galaxy in three surveys "
                     f"(top {N_BRIGHT} bright + {N_RANDOM} random)",
                     fontsize=12, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out = os.path.join(HERE, "figures", f"three_survey_{field}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

        # ----------------- figure B: LSST+HSC before/after preprocessing --
        subset = picks[:6]
        fig, axes = plt.subplots(len(subset), 4,
                                 figsize=(13.5, 3.2 * len(subset)))
        for k, r in enumerate(subset):
            m = meta.loc[r]
            lc = lsst.row(r)
            hrow = int(parent_row.loc[int(m.object_id_hsc)])
            hc = hsc.row(hrow)
            lsst_in = pp.preprocess_lsst(lc[:4],
                                         bands=("g", "r", "i", "z")).numpy()
            hsc_in = preprocess_raw_image(
                torch.as_tensor(hc), survey="hsc", crop_size=48)[:4].numpy()

            ax = axes[k, 0]
            ax.imshow(asinh_rgb(lc[:3]), origin="lower")
            ax.add_patch(Rectangle((60, 60), 40, 40, ec="cyan", fc="none",
                                   lw=1.2))
            ax.set_ylabel(f"{m.object_id_hsc}\npsf_i={m.psf_fwhm_i:.2f}\" "
                          f"N_i={m.n_input_i:.0f}", fontsize=7)
            axes[k, 1].imshow(norm_rgb(lsst_in), origin="lower",
                              interpolation="nearest")
            ax = axes[k, 2]
            ax.imshow(asinh_rgb(hc[:3] / HSC_ZPSCALE), origin="lower")
            ax.add_patch(Rectangle((56, 56), 48, 48, ec="cyan", fc="none",
                                   lw=1.2))
            axes[k, 3].imshow(norm_rgb(hsc_in), origin="lower",
                              interpolation="nearest")
            for c in range(4):
                axes[k, c].set_xticks([]), axes[k, c].set_yticks([])
            if k == 0:
                for c, t in enumerate(
                        ["LSST original (32\")\ncyan = model FoV",
                         "LSST model input 48×48",
                         "HSC original (26.9\")\ncyan = model FoV",
                         "HSC model input 48×48"]):
                    axes[0, c].set_title(t, fontsize=9)
        fig.suptitle(f"{field}: raw stamps and normalized model inputs",
                     fontsize=12, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out = os.path.join(HERE, "figures",
                           f"preprocessing_preview_{field}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
