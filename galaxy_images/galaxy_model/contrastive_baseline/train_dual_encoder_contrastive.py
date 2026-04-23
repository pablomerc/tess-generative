"""
Train dual-encoder contrastive baseline on precomputed neighbors data.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Avoid matplotlib trying to write to non-writable $HOME/.config on cluster nodes.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# Avoid numba cache writes to read-only environment paths when UMAP is enabled.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import h5py
import numpy as np
import pytorch_lightning as pl
import torch

# ROCm workaround: hipBLASLt is buggy on MI210 for certain matrix shapes.
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

_current_path = Path(__file__).resolve()
_project_root = _current_path.parents[3]  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.neighbors import NeighborsPrecomputedDataset, simple_collate
from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)


PRECOMPUTED_H5 = "/work1/jeroenaudenaert/pablomer/data/neighbor_batches/neighbours_vds.h5"
TRAIN_SHARDS_VDS = "/work1/jeroenaudenaert/pablomer/data/train_neighbors.vds"
VAL_SHARDS_VDS = "/work1/jeroenaudenaert/pablomer/data/val_neighbors.vds"
RANDOM_VAL_DIR = "/work1/jeroenaudenaert/pablomer/data/contrastive_val"

VAL_TYPE = "shards"  # "shards" or "random_batches"
# VAL_TYPE = "random_batches"
VAL_RATIO = 0.08

BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_STEPS = 300_000
LR = 1e-4
EMBEDDING_DIM = 64
PROJECTION_DIM = 32
PROJECTION_HIDDEN_DIM = 64

WANDB_PROJECT = "galaxy-contrastive-neighbours-baseline"
RUN_NAME = "dual-encoder-contrastive-resnet18"
CHECKPOINT_DIR = "/work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline"


def _save_random_val_set(dataset, val_indices, save_dir, seed):
    """Persist the randomly-selected val subset to an H5 file so it is reproducible."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"val_random_seed{seed}.h5")
    if os.path.isfile(out_path):
        print(f"[val] random val set already saved at {out_path}, skipping write.")
        return

    val_indices = np.sort(val_indices)
    print(f"[val] saving {len(val_indices)} val samples to {out_path} ...")

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('targets',        data=dataset.targets[val_indices].numpy(),  compression='lzf')
        f.create_dataset('samegals',       data=dataset.samegals[val_indices].numpy(), compression='lzf')
        f.create_dataset('sameins',        data=dataset.sameins[val_indices].numpy(),  compression='lzf')
        f.create_dataset('neighbor_masks', data=dataset.masks[val_indices].numpy(),    compression='lzf')
        f.create_dataset('meta_idx',       data=dataset.meta_idx[val_indices])

        surveys = [dataset.meta_survey[i].encode('utf-8') for i in val_indices]
        f.create_dataset('meta_survey', data=surveys)

        f.create_dataset('meta_num_same_instrument', data=dataset.meta_num_same[val_indices])
        f.attrs['seed'] = seed
        f.attrs['val_indices_count'] = len(val_indices)

    print(f"[val] saved {out_path}")


class VerboseModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that prints saved checkpoint paths to terminal."""

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            print(f"[checkpoint] saved: {filepath}", flush=True)


def is_h100_gpu() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    try:
        for i in range(torch.cuda.device_count()):
            if "h100" in torch.cuda.get_device_name(i).lower():
                return True
    except Exception:
        return False
    return False


def main():
    seed = 42
    pl.seed_everything(seed, workers=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_run_dir = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_{run_stamp}")
    os.makedirs(checkpoint_run_dir, exist_ok=True)
    print(f"[checkpoint] run directory: {checkpoint_run_dir}", flush=True)

    precision_setting = "bf16-mixed"
    batch_size = BATCH_SIZE

    if VAL_TYPE == "shards":
        train_ds = NeighborsPrecomputedDataset(TRAIN_SHARDS_VDS)
        val_ds = NeighborsPrecomputedDataset(VAL_SHARDS_VDS)

    elif VAL_TYPE == "random_batches":
        dataset = NeighborsPrecomputedDataset(PRECOMPUTED_H5)
        total_size = len(dataset)
        val_size = int(total_size * VAL_RATIO)
        train_size = total_size - val_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        _save_random_val_set(dataset, val_ds.indices, RANDOM_VAL_DIR, seed)

    else:
        raise ValueError(f"Unknown VAL_TYPE: {VAL_TYPE!r}. Use 'shards' or 'random_batches'.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=simple_collate,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=simple_collate,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )

    model = DualEncoderContrastiveModule(
        in_channels=4,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        projection_hidden_dim=PROJECTION_HIDDEN_DIM,
        pretrained_encoder=False,
        temperature_galaxy=0.1,
        temperature_instrument=0.1,
        lambda_galaxy=1.0,
        lambda_instrument=1.0,
        lr=LR,
        weight_decay=1e-4,
        enable_umap_logging=True,
        num_umap_batches=16,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        log_model=False,
        config={
            "batch_size": batch_size,
            "precision": precision_setting,
            "val_type": VAL_TYPE,
            "dataset": PRECOMPUTED_H5 if VAL_TYPE == "random_batches" else TRAIN_SHARDS_VDS,
            "val_ratio": VAL_RATIO if VAL_TYPE == "random_batches" else None,
            "max_steps": NUM_STEPS,
        },
    )

    best_checkpoint = VerboseModelCheckpoint(
        dirpath=checkpoint_run_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    periodic_checkpoint = VerboseModelCheckpoint(
        dirpath=checkpoint_run_dir,
        every_n_train_steps=1000,
        save_top_k=1,
        filename="latest-step={step}",
        save_last=False,
    )

    n_devices = int(os.environ.get("N_DEVICES", 4))
    trainer = pl.Trainer(
        max_steps=NUM_STEPS,
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        strategy="ddp_find_unused_parameters_true" if n_devices > 1 else "auto",
        precision=precision_setting,
        log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
