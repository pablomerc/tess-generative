"""
Variant C: random in-batch same-survey as instrument positives + physics pair as explicit instrument negative.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import pytorch_lightning as pl
import torch

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

_current_path = Path(__file__).resolve()
_project_root = _current_path.parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.neighbors import NeighborsPrecomputedDataset, simple_collate
from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)


TRAIN_SHARDS_VDS = "/work1/jeroenaudenaert/pablomer/data/train_neighbors.vds"
VAL_SHARDS_VDS = "/work1/jeroenaudenaert/pablomer/data/val_neighbors.vds"

BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_STEPS = 100_000
LR = 1e-4
EMBEDDING_DIM = 64
PROJECTION_DIM = 32
PROJECTION_HIDDEN_DIM = 64

WANDB_PROJECT = "galaxy-contrastive-neighbours-baseline"
RUN_NAME = "dual-encoder-contrastive-varC-random-pos-physics-neg"
CHECKPOINT_DIR = "/work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline"


class VerboseModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            print(f"[checkpoint] saved: {filepath}", flush=True)


def main():
    seed = 42
    pl.seed_everything(seed, workers=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_run_dir = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_{run_stamp}")
    os.makedirs(checkpoint_run_dir, exist_ok=True)
    print(f"[checkpoint] run directory: {checkpoint_run_dir}", flush=True)

    train_ds = NeighborsPrecomputedDataset(TRAIN_SHARDS_VDS)
    val_ds = NeighborsPrecomputedDataset(VAL_SHARDS_VDS)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=simple_collate, persistent_workers=NUM_WORKERS > 0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=simple_collate, persistent_workers=NUM_WORKERS > 0, pin_memory=True,
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
        include_physics_pair_as_instrument_negative=True,
        use_random_instrument_positives=True,
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        log_model=False,
        config={"batch_size": BATCH_SIZE, "max_steps": NUM_STEPS,
                "include_physics_pair_as_instrument_negative": True,
                "use_random_instrument_positives": True},
    )

    best_checkpoint = VerboseModelCheckpoint(
        dirpath=checkpoint_run_dir, monitor="val/loss", mode="min", save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}", auto_insert_metric_name=False,
    )
    periodic_checkpoint = VerboseModelCheckpoint(
        dirpath=checkpoint_run_dir, every_n_train_steps=1000, save_top_k=1,
        filename="latest-step={step}", save_last=False,
    )

    n_devices = int(os.environ.get("N_DEVICES", 4))
    trainer = pl.Trainer(
        max_steps=NUM_STEPS,
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        strategy="ddp_find_unused_parameters_true" if n_devices > 1 else "auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
