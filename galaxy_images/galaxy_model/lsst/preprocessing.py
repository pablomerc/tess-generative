"""LSST DP2 branch of the tess-generative preprocessing.

Turns raw stamps from data/lsst_efficient/ (160x160, nanomaggies, float16,
NaN for missing bands) into model-ready tensors, mirroring the HSC/Legacy
pipeline step for step:

    band select -> [unit] -> CenterCrop(48) -> clamp -> arcsinh range
    compression -> FoV zoom x0.84 (-> 8.0" like HSC) -> normalize

Self-contained: imports CenterCrop/RangeCompress from the shared module
read-only, nothing outside lsst/ is modified. Constants derived by
compute_lsst_stats.py are pasted here (never loaded at runtime).

Self-test (regression, runs anywhere the repo does):
    python -m galaxy_images.galaxy_model.lsst.preprocessing
verifies against the committed fixture in fixtures/preprocess_reference.npz,
which was cross-validated at creation time against the independent
implementation in cross_matching/scripts/07_pixel_parity_gallery.py.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from galaxy_images.image_preprocessing import CenterCrop, RangeCompress

# ---------------------------------------------------------------- constants
STORE_BANDS = ("g", "r", "i", "z", "u", "y")     # lsst_image.bin channel order
NJY_PER_NMGY = 10.0 ** ((31.4 - 22.5) / 2.5)     # 3630.78
ZOOM_FACTOR = 0.168 / 0.200                       # LSST px -> HSC FoV (0.84)

# Clamp thresholds in nanomaggies (symmetric +/-). griz = the DES values used
# for Legacy (identical units).
#
# FINDING (2026-09-01, compute_lsst_stats + spot-check of the parent stores):
# clamping is VESTIGIAL across the whole pipeline — zero pixels exceed the
# thresholds in sampled LSST griz (5000 objs), Legacy@DES, or HSC@HSC (400
# rows each). The arcsinh range compression does all the dynamic-range work.
# We keep the clamp only as a safety rail against rare out-of-sample pixels
# (coadd artifacts, satellite streaks, saturated star spikes) and for parity
# with the HSC/Legacy pipeline. u/y are set ~2x the observed |pixel| maxima
# (0.24 / 6.0; see data/lsst_efficient/lsst_stats.json).
#
# To disable clamping for a band entirely, set its threshold to float("inf")
# (torch.clip with +/-inf is a no-op). None means "not derived yet" and errors.
LSST_CLAMP = {
    "g": 6.0, "r": 15.0, "i": 20.0, "z": 25.0,
    "u": 0.5,
    "y": 10.0,
}

# Scalar (mean, std) applied after the full pipeline, keyed by band string.
# "griz" from 500 stamps (cross_matching script 07); reproduced as
# (0.0295, 0.0588) on 5000 stamps — kept at the original value the committed
# fixture was generated with (difference is sampling noise).
LSST_NORM = {
    "griz": (0.0303, 0.0591),
    "grizy": (0.0322, 0.0696),
    "ugrizy": (0.0291, 0.0640),   # COSMOS-only by construction (u coverage)
}
# Per-band alternative (DEEP2-3 depth-imbalance mitigation), norm_key="per_band".
# From compute_lsst_stats 2026-09-01; u from COSMOS only.
LSST_NORM_PER_BAND = {
    "g": (0.0133, 0.0341),
    "r": (0.0267, 0.0518),
    "i": (0.0361, 0.0642),
    "z": (0.0417, 0.0736),
    "u": (0.0064, 0.0203),
    "y": (0.0433, 0.1011),
}

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "preprocess_reference.npz")


def zoom_fov(x: torch.Tensor, factor: float = ZOOM_FACTOR) -> torch.Tensor:
    """Shrink the field of view: center-crop to round(factor*size), resize
    back with antialiased bilinear (same math as data.zoom_legacy_image)."""
    is_3d = x.ndim == 3
    if is_3d:
        x = x.unsqueeze(0)
    size = x.shape[-1]
    x = CenterCrop(crop_size=round(factor * size))(x)
    x = F.interpolate(x, size=(size, size), mode="bilinear",
                      align_corners=False, antialias=True)
    return x.squeeze(0) if is_3d else x


def preprocess_lsst(
    cube,
    bands: tuple[str, ...] = ("g", "r", "i", "z"),
    crop_size: int = 48,
    input_unit: str = "nanomaggy",
    normalize: bool = True,
    norm_key: str | None = None,
) -> torch.Tensor:
    """Full LSST preprocessing. `cube` is (C, 160, 160) or (B, C, 160, 160)
    with channels == `bands` (pass slices of the store cube plus their
    names). Returns float32, same leading shape, (len(bands), crop, crop).
    """
    x = torch.as_tensor(np.asarray(cube), dtype=torch.float32)
    is_3d = x.ndim == 3
    if is_3d:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != len(bands):
        raise ValueError(f"cube shape {tuple(x.shape)} does not match "
                         f"bands {bands}")
    unknown = [b for b in bands if b not in STORE_BANDS]
    if unknown:
        raise ValueError(f"unknown band(s) {unknown}; store bands are "
                         f"{STORE_BANDS}")

    # missing planes are a caller bug: filter with lsst_meta has_<band> flags
    nan_per_band = torch.isnan(x).flatten(2).any(-1)          # (B, C)
    if nan_per_band.any():
        bad = [bands[c] for c in nan_per_band.any(0).nonzero().flatten()]
        raise ValueError(
            f"NaN plane(s) for band(s) {bad}: these objects lack the band. "
            f"Filter rows with lsst_meta.parquet has_<band> before loading.")

    if input_unit == "njy":
        x = x / NJY_PER_NMGY
    elif input_unit != "nanomaggy":
        raise ValueError(f"input_unit must be 'nanomaggy' or 'njy', "
                         f"got {input_unit!r}")

    x = CenterCrop(crop_size=crop_size)(x)

    for ci, b in enumerate(bands):
        thr = LSST_CLAMP[b]
        if thr is None:
            raise ValueError(f"no clamp for band {b!r} yet — run "
                             "compute_lsst_stats.py and paste LSST_CLAMP")
        x[:, ci] = torch.clip(x[:, ci], -thr, thr)

    x = RangeCompress().forward(x)
    x = zoom_fov(x)

    if normalize:
        key = norm_key or "".join(bands)
        if key == "per_band":
            if LSST_NORM_PER_BAND is None:
                raise ValueError("per-band norm stats not computed yet — run "
                                 "compute_lsst_stats.py")
            for ci, b in enumerate(bands):
                mean, std = LSST_NORM_PER_BAND[b]
                x[:, ci] = (x[:, ci] - mean) / std
        else:
            if LSST_NORM.get(key) is None:
                raise ValueError(f"no norm stats for {key!r} — run "
                                 "compute_lsst_stats.py (or pass norm_key)")
            mean, std = LSST_NORM[key]
            x = (x - mean) / std

    return x.squeeze(0) if is_3d else x


def _self_test() -> None:
    """Regression check against the committed fixture (see module docstring)."""
    ref = np.load(_FIXTURE)
    cubes = ref["input_griz_nmgy"]                       # (K, 4, 160, 160) f16
    got_un = preprocess_lsst(cubes, normalize=False)
    got_no = preprocess_lsst(cubes, normalize=True)
    assert torch.allclose(got_un, torch.as_tensor(ref["expected_unnormalized"]),
                          atol=1e-5), "unnormalized output drifted"
    assert torch.allclose(got_no, torch.as_tensor(ref["expected_normalized"]),
                          atol=1e-4), "normalized output drifted"
    print(f"self-test OK: {len(cubes)} fixture objects reproduce "
          f"(fixture created {ref['created']}, ids {ref['object_ids']})")


if __name__ == "__main__":
    _self_test()
