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
    "neighbors_base_ins4x4": VariantSpec(
        name="neighbors_base_ins4x4",
        description="Same encoder spatial layout as `base` (2x2 -> 4 tokens) but the instrument encoder bottleneck is reduced to 4-d (instrument_zdim=4). An auto-inserted ins_proj (4 -> cross_attention_dim) keeps the UNet cross-attn dim consistent.",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={
            "all_attention": True,
            "instrument_zdim": 4,
        },
    ),
    "neighbors_base_ins_flatten": VariantSpec(
        name="neighbors_base_ins_flatten",
        description="Instrument encoder yields (B, 4, 4); flatten the 4 tokens of 4-d into a single (B, 1, 16) token per neighbor (no ins_proj). 1 instrument token per neighbor, dim 16.",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={
            "all_attention": True,
            "instrument_zdim": 4,
            "instrument_flatten_to_one_token": True,
        },
    ),
    "neighbors_base_ins_globalconv": VariantSpec(
        name="neighbors_base_ins_globalconv",
        description="Instrument encoder uses a 2x2 conv (16 channels) on layer4 instead of the 1x1 per-token proj, producing a single (B, 1, 16) global-features token per neighbor.",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={
            "all_attention": True,
            "instrument_zdim": 16,
            "encoder_2_global_conv": True,
        },
    ),
    "neighbors_base_ins_classcond": VariantSpec(
        name="neighbors_base_ins_classcond",
        description="Same encoder_2 as base_ins_globalconv (1x16 per neighbor via 2x2 conv); per-batch masked-mean across k neighbors gives a single 16-d vector that is fed to the UNet as class conditioning (added to time embedding) instead of cross-attention. encoder_1 / cross-attn unchanged.",
        model_cls=ConditionalFlowMatchingModule,
        model_overrides={
            "all_attention": True,
            "instrument_zdim": 16,
            "encoder_2_global_conv": True,
            "instrument_as_class_conditioning": True,
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
