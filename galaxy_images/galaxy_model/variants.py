from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, Type

import pytorch_lightning as pl

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.hierarchical_attention.double_train_fm_neighbors_hier_global_ins import (
    HierarchicalGlobalInstrumentFlowMatchingModule,
)


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
    "neighbors_hier_global_ins": VariantSpec(
        name="neighbors_hier_global_ins",
        description="Hierarchical galaxy conditioning with a separate pooled global instrument path.",
        model_cls=HierarchicalGlobalInstrumentFlowMatchingModule,
    ),
    "neighbors_hier_bn36x16_ng": VariantSpec(
        name="neighbors_hier_bn36x16_ng",
        description="bn_36x16 hierarchical variant with the global physics path disabled (spatial tokens + instrument-global only).",
        model_cls=HierarchicalGlobalInstrumentFlowMatchingModule,
        model_overrides={"disable_global_physics": True},
    ),
    "neighbors_base6x6": VariantSpec(
        name="neighbors_base6x6",
        description="Baseline dual-encoder model with layer3+layer4 strides removed -> encoder outputs 6x6 spatial map.",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={
            "all_attention": True,
            "encoder_stride_overrides": {"layer3": 1, "layer4": 1},
        },
    ),
    "neighbors_base3x3": VariantSpec(
        name="neighbors_base3x3",
        description="Baseline dual-encoder model with layer4 stride removed -> encoder outputs 3x3 spatial map.",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={
            "all_attention": True,
            "encoder_stride_overrides": {"layer4": 1},
        },
    ),
    "neighbors_base3x3_phys_only": VariantSpec(
        name="neighbors_base3x3_phys_only",
        description="Physics encoder gets layer4 stride removed (3x3, 9 tokens). Instrument encoder kept at original baseline (2x2, 4 tokens).",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={
            "all_attention": True,
            "encoder_1_stride_overrides": {"layer4": 1},
            "encoder_2_stride_overrides": None,
        },
    ),
}


def get_variant(name: str) -> VariantSpec:
    if name not in VARIANTS:
        available = ", ".join(sorted(VARIANTS.keys()))
        raise ValueError(f"Unknown variant {name!r}. Available variants: {available}")
    return VARIANTS[name]


def filter_supported_model_kwargs(
    model_cls: Type[pl.LightningModule],
    raw_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    signature = inspect.signature(model_cls.__init__)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return dict(raw_kwargs)

    supported_names = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {
        key: value
        for key, value in raw_kwargs.items()
        if key in supported_names
    }
