from pathlib import Path
import sys

import torch

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)
from galaxy_images.galaxy_model.neighbors import NeighborsPrecomputedDataset, simple_collate


TEST_PRECOMPUTED_H5 = Path(
    "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5"
)


def _make_model(encoder_pool: str = "avg") -> DualEncoderContrastiveModule:
    return DualEncoderContrastiveModule(
        in_channels=4,
        embedding_dim=32,
        projection_dim=16,
        projection_hidden_dim=32,
        pretrained_encoder=False,
        encoder_pool=encoder_pool,
        image_size=48,
        temperature_galaxy=0.1,
        temperature_instrument=0.1,
        lambda_galaxy=1.0,
        lambda_instrument=1.0,
        lr=1e-4,
        weight_decay=1e-4,
    )


def _run_synthetic_batch(encoder_pool: str):
    torch.manual_seed(0)
    model = _make_model(encoder_pool=encoder_pool)
    model.train()

    b, c, h, w = 3, 4, 48, 48
    k = 4
    targets = torch.randn(b, c, h, w)
    samegals = torch.randn(b, c, h, w)
    sameins = torch.randn(b, k, c, h, w)
    masks = torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    metadata = [{"anchor_survey": "hsc", "idx": i, "num_same_instrument": int(masks[i].sum())} for i in range(b)]

    loss, metrics = model._compute_losses((targets, samegals, sameins, masks, metadata))
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert torch.isfinite(metrics["loss_galaxy"])
    assert torch.isfinite(metrics["loss_instrument"])
    assert torch.isfinite(metrics["acc_galaxy"])
    assert torch.isfinite(metrics["acc_instrument"])

    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad
    return model


def test_compute_losses_synthetic_batch_backward():
    _run_synthetic_batch("avg")


def test_compute_losses_synthetic_batch_backward_conv1x1():
    # "conv1x1" keeps spatial tokens (mirrors double_train_fm_neighbors.py) via a 1x1
    # conv rather than global pooling. By default token_dim = embedding_dim // (H'*W'),
    # so the flattened output width equals embedding_dim (dim-matched to "avg").
    import torch.nn as nn
    model = _run_synthetic_batch("conv1x1")
    assert isinstance(model.encoder_galaxy.proj, nn.Conv2d)          # not global-pool + Linear
    assert model.encoder_galaxy.out_dim == 32                        # == embedding_dim
    assert model.encoder_galaxy.out_dim == model.head_galaxy.net[0].in_features
    assert model.encoder_galaxy.proj.out_channels < 32              # token_dim < embedding_dim (spatial preserved)


def test_real_dataset_batch_shapes_and_loss():
    assert TEST_PRECOMPUTED_H5.exists(), f"Missing test file: {TEST_PRECOMPUTED_H5}"

    dataset = NeighborsPrecomputedDataset(str(TEST_PRECOMPUTED_H5))
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate,
    )
    batch = next(iter(loader))
    targets, samegals, sameins, masks, metadata = batch

    assert targets.ndim == 4
    assert samegals.shape == targets.shape
    assert sameins.ndim == 5
    assert masks.ndim == 2
    assert sameins.shape[0] == targets.shape[0]
    assert sameins.shape[1] == masks.shape[1]
    assert isinstance(metadata, list) and len(metadata) == targets.shape[0]
    assert masks.any(), "Expected at least one valid same-instrument neighbor"

    model = _make_model()
    model.eval()
    with torch.no_grad():
        loss, metrics = model._compute_losses(batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["loss_galaxy"])
    assert torch.isfinite(metrics["loss_instrument"])


def test_real_batch_single_optimizer_step():
    assert TEST_PRECOMPUTED_H5.exists(), f"Missing test file: {TEST_PRECOMPUTED_H5}"

    dataset = NeighborsPrecomputedDataset(str(TEST_PRECOMPUTED_H5))
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate,
    )
    batch = next(iter(loader))

    model = _make_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    loss, _ = model._compute_losses(batch)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss.detach())
