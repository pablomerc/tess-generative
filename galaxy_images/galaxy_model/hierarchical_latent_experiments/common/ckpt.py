"""Checkpoint loader for HierarchicalGlobalInstrumentFlowMatchingModule."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")


DEFAULT_CKPT = (
    "/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/"
    "hierarchical_attention/outputs/neighbors_hier_global_ins/2026-04-22_2/"
    "checkpoints/latest-step=step=201000.ckpt"
)


def _module_class():
    from galaxy_images.galaxy_model.hierarchical_attention.double_train_fm_neighbors_hier_global_ins import (
        HierarchicalGlobalInstrumentFlowMatchingModule,
    )
    return HierarchicalGlobalInstrumentFlowMatchingModule


def load_trained(ckpt_path: str | Path = DEFAULT_CKPT, device: str | torch.device = "cuda"):
    cls = _module_class()
    model = cls.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


def load_untrained(ckpt_path: str | Path = DEFAULT_CKPT, device: str | torch.device = "cuda"):
    cls = _module_class()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {}) or {}
    valid = set(inspect.signature(cls.__init__).parameters.keys())
    init_kwargs = {k: v for k, v in hparams.items() if k in valid}
    model = cls(**init_kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)
