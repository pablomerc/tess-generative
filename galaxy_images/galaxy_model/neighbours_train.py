"""
Train the double encoder flow matching model on the Neighbors dataset (48x48 only).

Uses NeighborsDataset from neighbors.py. No lenses, no masking, no multi-size.
"""

import os
import sys

# Print immediately so you see the script started (heavy imports below can take 1–2 min)
print("neighbours_train: loading imports...", flush=True)

import torch
from torch.utils.data import DataLoader, Subset, random_split


_current_path = os.path.abspath(__file__)
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_path)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from galaxy_images.galaxy_model.neighbors import (
    NeighborsDataset,
    NeighborsPrecomputedDataset,
    simple_collate,
    collate_neighbors,
)
from galaxy_images.galaxy_model.double_train_fm_neighbors import (
    ConditionalFlowMatchingModule,
    is_h100_gpu,
)


# Model expects (x_1, cond_image_samegal, cond_image_sameins, metadata) — no masks.

# For NeighborsDataset / NeighborsDatasetRawRAM: variable-length sameins → use collate_neighbors (pads), then drop masks.
def collate_for_model(batch):
    targets, samegals, padded_neighbors, _, metadata = collate_neighbors(batch)
    return (targets, samegals, padded_neighbors, metadata)


# For NeighborsPrecomputedDataset: data is already padded → use simple_collate, then drop masks.
def collate_for_model_precomputed(batch):
    # Don't use the underscore _; grab the masks!
    targets, samegals, sameins, masks, metadata = simple_collate(batch)
    return (targets, samegals, sameins, masks, metadata)


# --- Config ---

# If using neighbors file directly
# NEIGHBORS_H5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
# MAX_NEIGHBORS = 15



PRECOMPUTED_H5 = '/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5'
TRAIN_SHARDS_VDS = "/data/vision/billf/scratch/pablomer/data/neighbors_trainingset_march.vds"
VAL_SHARDS_VDS = "/data/vision/billf/scratch/pablomer/data/neighbors_valset_march.vds"

BATCH_SIZE = 64
NUM_WORKERS = 0
DATALOADER_MODE = "precomputed"

VALIDATION_MODE = "shards"  # "batches" (random split from single VDS) or "shards" (separate train/val VDS files)
VAL_RATIO = 0.05
NUM_STEPS = 300_000 * 5
IMAGE_SIZE = 48
LR = 1e-4
LAMBDA_GENERATIVE = 1.0
# LAMBDA_GEOMETRIC = 7.5e-4
LAMBDA_GEOMETRIC = 0.0
POOLED_CONDITIONING = os.environ.get("POOLED_CONDITIONING", "0") == "1"

WANDB_PROJECT = "galaxy-flow-matching-neighbours"


def main():
    seed = 42
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    is_h100 = is_h100_gpu()
    batch_size = BATCH_SIZE
    precision_setting = "16-mixed"
    num_steps = NUM_STEPS
    if is_h100:
        batch_size = 64
        precision_setting = "bf16-mixed"
        print(f"H100 detected: batch_size={batch_size}, precision={precision_setting}")

    if VALIDATION_MODE == "shards":
        print(f"Using shard-based split: train={TRAIN_SHARDS_VDS}, val={VAL_SHARDS_VDS}")
        train_ds = NeighborsPrecomputedDataset(TRAIN_SHARDS_VDS)
        val_ds = NeighborsPrecomputedDataset(VAL_SHARDS_VDS)
    else:
        dataset = NeighborsPrecomputedDataset(PRECOMPUTED_H5)
        total_size = len(dataset)
        val_size = int(total_size * VAL_RATIO)
        train_size = total_size - val_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    collate_fn = collate_for_model_precomputed if DATALOADER_MODE == "precomputed" else collate_for_model
    conditioning_mode = "pooled" if POOLED_CONDITIONING else "spatial"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )

    model = ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        image_size=IMAGE_SIZE,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=16,
        pretrained_encoder=False,
        concat_conditioning=False,
        pooled_conditioning=POOLED_CONDITIONING,
        lr=LR,
        num_sample_images=10,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=LAMBDA_GENERATIVE,
        lambda_geometric=LAMBDA_GEOMETRIC,
        mask_center=False,
    )

    # Pass config here; Lightning handles DDP so only rank 0 gets real wandb (avoids .config.update() on placeholder)
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=f"neighbours-48x48-zdim16-geom0.0-cond-{conditioning_mode}-val-{VALIDATION_MODE}",
        log_model=False,
        config={
            "batch_size": batch_size,
            "precision": precision_setting,
            "is_h100": is_h100,
            "dataset": "NeighborsDataset",
            "image_size": IMAGE_SIZE,
            "validation": VALIDATION_MODE,
            "pooled_conditioning": POOLED_CONDITIONING,
            "conditioning_mode": conditioning_mode,
        },
    )

    best_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    periodic_checkpoint = ModelCheckpoint(
        every_n_train_steps=1000,
        save_top_k=1,
        filename="latest-step={step}",
        save_last=False,
    )

    n_devices = 4  # set to 4 or more for multi-GPU
    trainer = pl.Trainer(
        max_steps=max(1, int(num_steps / n_devices)),
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        strategy="ddp_find_unused_parameters_true" if n_devices > 1 else "auto",
        log_every_n_steps=10,
        precision=precision_setting,
        val_check_interval=1000, # bring back to 1000
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint],
        num_sanity_val_steps=0,  # skip sanity check to start training faster (first val batch has heavy CUDA init)
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
