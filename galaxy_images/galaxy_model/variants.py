from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type

import pytorch_lightning as pl

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    model_cls: Type[pl.LightningModule]
    model_overrides: Dict[str, Any] = field(default_factory=dict)


VARIANTS: Dict[str, VariantSpec] = {
    "neighbors_all_attn": VariantSpec(
        name="neighbors_all_attn",
        description="Double-encoder neighbors model with cross-attention in all UNet blocks.",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={"all_attention": True},
    ),
    "neighbors_mixed_attn": VariantSpec(
        name="neighbors_mixed_attn",
        description="Double-encoder neighbors model with mixed attention/non-attention UNet blocks.",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={"all_attention": False},
    ),
}


def get_variant(name: str) -> VariantSpec:
    if name not in VARIANTS:
        available = ", ".join(sorted(VARIANTS.keys()))
        raise ValueError(f"Unknown variant {name!r}. Available variants: {available}")
    return VARIANTS[name]
