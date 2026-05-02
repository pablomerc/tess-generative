"""Single source of truth for the 4 physics-latent variants.

Variants (instrument is always model.encode_image(image)["instrument"]["flat"]):

  spatial_flat      physics.spatial_flat                            576-d (bn_36x16)
  global_vec        physics.global_vec                               64-d
  spatial_pooled    mean(physics.spatial_concat, dim=1)              16-d
  global_concat     concat(physics.global_vec, physics.spatial_flat) 640-d
"""

from __future__ import annotations

import torch
from torch import Tensor

VARIANTS = ("spatial_flat", "global_vec", "spatial_pooled", "global_concat")


def physics_dim(variant: str, token_dim: int = 16, global_dim: int = 64,
                num_spatial_tokens: int = 36) -> int:
    if variant == "spatial_flat":
        return num_spatial_tokens * token_dim
    if variant == "global_vec":
        return global_dim
    if variant == "spatial_pooled":
        return token_dim
    if variant == "global_concat":
        return global_dim + num_spatial_tokens * token_dim
    raise ValueError(variant)


@torch.no_grad()
def extract_physics(model, image: Tensor, variant: str) -> Tensor:
    enc = model.encode_image(image)
    p = enc["physics"]
    if variant == "spatial_flat":
        return p["spatial_flat"]
    if variant == "global_vec":
        return p["global_vec"]
    if variant == "spatial_pooled":
        return p["spatial_concat"].mean(dim=1)
    if variant == "global_concat":
        return torch.cat([p["global_vec"], p["spatial_flat"]], dim=1)
    raise ValueError(variant)


@torch.no_grad()
def extract_both(model, image: Tensor, variant: str) -> tuple[Tensor, Tensor]:
    """Single forward pass; returns (physics_latent, instrument_flat)."""
    enc = model.encode_image(image)
    p = enc["physics"]
    if variant == "spatial_flat":
        phys = p["spatial_flat"]
    elif variant == "global_vec":
        phys = p["global_vec"]
    elif variant == "spatial_pooled":
        phys = p["spatial_concat"].mean(dim=1)
    elif variant == "global_concat":
        phys = torch.cat([p["global_vec"], p["spatial_flat"]], dim=1)
    else:
        raise ValueError(variant)
    inst = enc["instrument"]["flat"]
    return phys, inst
