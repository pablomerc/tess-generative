#!/usr/bin/env python3
"""Derive the LSST preprocessing constants still marked TODO in preprocessing.py.

1. u/y CLAMP thresholds by clip-fraction matching: measure what fraction of
   (post-crop, nanomaggy) pixels the DES griz clamps actually clip on LSST
   griz; take the median fraction as the target; set the u/y thresholds at
   the (1 - target) quantile of their own |pixel| distributions, rounded to
   2 significant figures.
2. NORM stats with those clamps applied: scalar (mean, std) of the full
   pipeline output for the band sets griz / grizy / ugrizy, plus per-band
   stats and per-field diagnostics. griz doubles as a consistency check
   against the known (0.0303, 0.0591) from the cross_matching validation.

Prints a paste-ready constants block for preprocessing.py (constants are
committed in source, never loaded at runtime) and saves the full numbers to
<store>/lsst_stats.json for provenance.

Login-node safe: rows are read with seek+read (no memmap — mapping the 19 GB
bin exhausts the login node's address-space ulimit) and every statistic is
accumulated streaming (running sums / bounded pixel buffers). Two passes over
a stratified sample; a few minutes.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from galaxy_images.image_preprocessing import CenterCrop
from galaxy_images.galaxy_model.lsst import preprocessing as pp

STORE = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
N_PER_FIELD = 2500
SEED = 0


class RowReader:
    """Direct-read access to lsst_image.bin without memory-mapping it."""

    def __init__(self, store):
        with open(os.path.join(store, "meta.json")) as f:
            self.meta = json.load(f)
        self.shape = tuple(self.meta["shape"])          # (M, 6, 160, 160)
        self.row_bytes = int(np.prod(self.shape[1:])) * 2
        self.f = open(os.path.join(store, "lsst_image.bin"), "rb")

    def row(self, r: int) -> np.ndarray:
        """(6, 160, 160) float32 nanomaggies (NaN where band missing)."""
        self.f.seek(r * self.row_bytes)
        buf = self.f.read(self.row_bytes)
        return np.frombuffer(buf, dtype=np.float16).reshape(
            self.shape[1:]).astype(np.float32)


class Welford:
    """Streaming sum/sumsq/count -> (mean, std)."""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def add(self, x: np.ndarray):
        self.n += x.size
        self.s += float(x.sum())
        self.s2 += float((x.astype(np.float64) ** 2).sum())

    def stats(self):
        mean = self.s / self.n
        return mean, float(np.sqrt(self.s2 / self.n - mean ** 2))


def round_2sf(x: float) -> float:
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + 1) if x else 0.0


def main():
    reader = RowReader(STORE)
    meta = pd.read_parquet(os.path.join(STORE, "lsst_meta.parquet"))
    band_i = {b: i for i, b in enumerate(pp.STORE_BANDS)}
    crop = CenterCrop(crop_size=48)

    sample = pd.concat([
        meta[meta.field == f].sample(min(N_PER_FIELD, (meta.field == f).sum()),
                                     random_state=SEED)
        for f in ("COSMOS", "DEEP2-3")]).sort_index()   # sorted -> read locality
    print(f"sample: {len(sample)} objects "
          f"({dict(sample.field.value_counts())})", flush=True)

    # ---- pass 1: griz clip counts + u/y pixel buffers (post-crop, nmgy) ----
    clip_counts = {b: 0 for b in "griz"}
    n_px_griz = 0
    uy_pixels = {"u": [], "y": []}
    for r, has_u, has_y in zip(sample.index, sample.has_u, sample.has_y):
        cube = reader.row(r)
        c48 = crop(torch.as_tensor(cube).unsqueeze(0)).numpy()[0]
        for b in "griz":
            clip_counts[b] += int((np.abs(c48[band_i[b]]) >
                                   pp.LSST_CLAMP[b]).sum())
        n_px_griz += c48.shape[-1] * c48.shape[-2]
        if has_u:
            uy_pixels["u"].append(np.abs(c48[band_i["u"]]).ravel())
        if has_y:
            uy_pixels["y"].append(np.abs(c48[band_i["y"]]).ravel())

    fracs = {b: clip_counts[b] / n_px_griz for b in "griz"}
    target = float(np.median(list(fracs.values())))
    print("clip fractions at DES griz clamps:",
          {b: f"{v:.2e}" for b, v in fracs.items()}, f"-> target {target:.2e}")

    clamps_new = {}
    for b in ("u", "y"):
        px = np.concatenate(uy_pixels[b])
        thr = round_2sf(float(np.quantile(px, 1.0 - target)))
        clamps_new[b] = thr
        print(f"clamp {b}: {thr}  (from {px.size / 2304:.0f} objects, "
              f"clips {float((px > thr).mean()):.2e} of pixels)")
        del px
    uy_pixels.clear()
    pp.LSST_CLAMP.update(clamps_new)   # in-process so pass 2 sees them

    # ---- pass 2: pipeline outputs, accumulated per (band, group) ----------
    # groups: 'all' rows, 'has_y', 'uy' (has_u & has_y), and per-field
    acc = {}

    def bump(band, group, x):
        acc.setdefault((band, group), Welford()).add(x)

    for r, field, has_u, has_y in zip(sample.index, sample.field,
                                      sample.has_u, sample.has_y):
        cube = reader.row(r)
        avail = tuple(b for b in pp.STORE_BANDS
                      if not np.isnan(cube[band_i[b]]).any())
        out = pp.preprocess_lsst(
            cube[[band_i[b] for b in avail]], bands=avail,
            normalize=False).numpy()
        for ci, b in enumerate(avail):
            bump(b, "all", out[ci])
            bump(b, field, out[ci])
            if has_y:
                bump(b, "has_y", out[ci])
            if has_u and has_y:
                bump(b, "uy", out[ci])

    def set_stats(bands, group):
        n = sum(acc[(b, group)].n for b in bands)
        s = sum(acc[(b, group)].s for b in bands)
        s2 = sum(acc[(b, group)].s2 for b in bands)
        mean = s / n
        return mean, float(np.sqrt(s2 / n - mean ** 2))

    norms = {"griz": set_stats("griz", "all"),
             "grizy": set_stats("grizy", "has_y"),
             "ugrizy": set_stats("grizuy", "uy")}
    per_band = {b: acc[(b, "all")].stats() for b in pp.STORE_BANDS}
    per_field = {f: set_stats("griz", f) for f in ("COSMOS", "DEEP2-3")}

    print(f"\nnorm griz: ({norms['griz'][0]:.4f}, {norms['griz'][1]:.4f})"
          f"   [consistency ref: {pp.LSST_NORM['griz']}]")
    for f, (m, s) in per_field.items():
        print(f"  griz {f}: ({m:.4f}, {s:.4f})")
    print(f"norm grizy: ({norms['grizy'][0]:.4f}, {norms['grizy'][1]:.4f})")
    print(f"norm ugrizy: ({norms['ugrizy'][0]:.4f}, "
          f"{norms['ugrizy'][1]:.4f})   [COSMOS only by construction]")
    print("per-band:", {b: (f"{m:.4f}", f"{s:.4f}")
                        for b, (m, s) in per_band.items()})

    result = {"sample_per_field": N_PER_FIELD, "seed": SEED,
              "griz_clip_fractions": fracs, "target_clip_fraction": target,
              "clamps_uy": clamps_new, "norms": norms, "per_band": per_band,
              "per_field_griz": per_field}
    with open(os.path.join(STORE, "lsst_stats.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {STORE}/lsst_stats.json")

    print("\n" + "=" * 68)
    print("PASTE INTO preprocessing.py:")
    print(f'  LSST_CLAMP["u"] = {clamps_new["u"]}')
    print(f'  LSST_CLAMP["y"] = {clamps_new["y"]}')
    print(f'  LSST_NORM["grizy"] = ({norms["grizy"][0]:.4f}, '
          f'{norms["grizy"][1]:.4f})')
    print(f'  LSST_NORM["ugrizy"] = ({norms["ugrizy"][0]:.4f}, '
          f'{norms["ugrizy"][1]:.4f})')
    print("  LSST_NORM_PER_BAND = {")
    for b in pp.STORE_BANDS:
        print(f'      "{b}": ({per_band[b][0]:.4f}, {per_band[b][1]:.4f}),')
    print("  }")
    print("=" * 68)


if __name__ == "__main__":
    main()
