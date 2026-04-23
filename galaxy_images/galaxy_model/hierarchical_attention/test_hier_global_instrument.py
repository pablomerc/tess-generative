from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from galaxy_images.galaxy_model.config import load_experiment_config
from galaxy_images.galaxy_model.hierarchical_attention.double_train_fm_neighbors_hier_global_ins import (
    HierarchicalGlobalInstrumentFlowMatchingModule,
)
from galaxy_images.galaxy_model.train import _build_model
from galaxy_images.galaxy_model.validation_pairs import reconstruct_hsc_legacy_pairs


def _make_model() -> HierarchicalGlobalInstrumentFlowMatchingModule:
    return HierarchicalGlobalInstrumentFlowMatchingModule(
        experiment_config="bn_36x16",
        in_channels=4,
        cond_channels=4,
        image_size=48,
        model_channels=32,
        channel_mult=(1, 2, 4, 4),
        layers_per_block=1,
        attention_head_dim=8,
        instrument_zdim=16,
        pretrained_encoder=False,
        num_sample_images=2,
        num_mse_images=2,
        num_integration_steps=2,
        lambda_generative=1.0,
        lambda_geometric=0.0,
        num_umap_batches=1,
    )


def test_encode_image_forward_and_sample_shapes():
    torch.manual_seed(0)
    model = _make_model()
    model.eval()

    batch_size = 2
    num_neighbors = 3
    image = torch.randn(batch_size, 4, 48, 48)
    samegal = torch.randn(batch_size, 4, 48, 48)
    sameins = torch.randn(batch_size, num_neighbors, 4, 48, 48)
    masks = torch.tensor(
        [
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.bool,
    )
    metadata = [
        {"anchor_survey": "hsc", "idx": 0, "num_same_instrument": 2},
        {"anchor_survey": "legacy", "idx": 1, "num_same_instrument": 0},
    ]

    encoded = model.encode_image(image)
    physics = encoded["physics"]
    instrument = encoded["instrument"]

    assert len(physics["spatial_levels"]) == 1
    level0 = physics["spatial_levels"][0]
    assert level0["tokens"].shape == (batch_size, 36, 16)
    assert level0["height"] == 6
    assert level0["width"] == 6
    assert level0["rope"] is True
    assert physics["level_flats"][0].shape == (batch_size, 36 * 16)
    assert physics["spatial_concat"].shape == (batch_size, 36, 16)
    assert physics["spatial_flat"].shape == (batch_size, 36 * 16)
    assert physics["global_vec"].shape == (batch_size, 64)
    assert instrument["tokens"].shape == (batch_size, 1, 16)
    assert instrument["flat"].shape == (batch_size, 16)

    x_t = torch.randn_like(image)
    t = torch.rand(batch_size)
    with torch.no_grad():
        predicted = model(x_t, t, samegal, sameins, masks)
    assert predicted.shape == image.shape

    loss = model.compute_loss((image, samegal, sameins, masks, metadata))
    assert torch.isfinite(loss)
    assert model._loss_geom_total.item() == 0.0

    with torch.no_grad():
        samples = model.sample(samegal, sameins, masks=masks, num_steps=2)
    assert samples.shape == image.shape


def test_masked_mean_pooling_behavior():
    model = _make_model()

    tokens = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
            [[9.0, 9.0], [11.0, 11.0], [13.0, 13.0]],
            [[2.0, 4.0], [8.0, 10.0], [14.0, 16.0]],
        ]
    )
    masks = torch.tensor(
        [
            [1, 1, 0],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=torch.bool,
    )

    pooled = model._masked_mean_pool(tokens, masks)
    assert torch.allclose(pooled[0], torch.tensor([3.0, 5.0]))
    assert torch.allclose(pooled[1], torch.zeros(2))
    assert torch.allclose(pooled[2], torch.tensor([8.0, 10.0]))

    changed_padding = tokens.clone()
    changed_padding[0, 2] = torch.tensor([999.0, 999.0])
    pooled_changed_padding = model._masked_mean_pool(changed_padding, masks)
    assert torch.allclose(pooled_changed_padding[0], pooled[0])

    changed_valid = tokens.clone()
    changed_valid[2, 2] = torch.tensor([30.0, 34.0])
    pooled_changed_valid = model._masked_mean_pool(changed_valid, masks)
    assert not torch.allclose(pooled_changed_valid[2], pooled[2])

    sameins = torch.randn(3, 4, 4, 48, 48)
    pooled_instrument = model._pool_instrument_conditioning(sameins, torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=torch.bool,
    ))
    assert pooled_instrument.shape == (3, model.instrument_zdim)


def test_reconstruct_validation_pairs_returns_both_surveys():
    anchor = torch.tensor(
        [
            [[[1.0]]],
            [[[2.0]]],
            [[[3.0]]],
            [[[4.0]]],
        ]
    )
    samegal = torch.tensor(
        [
            [[[10.0]]],
            [[[20.0]]],
            [[[30.0]]],
            [[[40.0]]],
        ]
    )
    metadata = [
        {"anchor_survey": "hsc"},
        {"anchor_survey": "legacy"},
        {"anchor_survey": "hsc"},
        {"anchor_survey": "legacy"},
    ]

    hsc, legacy = reconstruct_hsc_legacy_pairs(anchor, samegal, metadata)

    assert torch.equal(hsc[:, 0, 0, 0], torch.tensor([1.0, 20.0, 3.0, 40.0]))
    assert torch.equal(legacy[:, 0, 0, 0], torch.tensor([10.0, 2.0, 30.0, 4.0]))
    assert hsc.shape[0] == anchor.shape[0]
    assert legacy.shape[0] == anchor.shape[0]
    assert hsc.shape[0] + legacy.shape[0] == 2 * anchor.shape[0]


def test_unified_builder_supports_new_and_existing_variants():
    hier_config = load_experiment_config(
        PROJECT_ROOT / "galaxy_images/galaxy_model/configs/neighbors_hier_global_ins.json",
        [],
    )
    hier_model, hier_variant = _build_model(hier_config)
    assert hier_variant.name == "neighbors_hier_global_ins"
    assert isinstance(hier_model, HierarchicalGlobalInstrumentFlowMatchingModule)
    assert hier_model.instrument_pooling == "masked_mean"

    default_config = load_experiment_config(
        PROJECT_ROOT / "galaxy_images/galaxy_model/configs/neighbors_default.json",
        [],
    )
    default_model, default_variant = _build_model(default_config)
    assert default_variant.name == "neighbors_all_attn"
    assert default_model.__class__.__name__ == "ConditionalFlowMatchingModule"
