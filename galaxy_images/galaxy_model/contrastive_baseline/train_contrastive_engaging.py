"""
Train the dual-encoder contrastive baseline ON ENGAGING, reading the local
memory-mapped `efficient_neighs` store instead of the AMD-only precomputed VDS
shards.

This is the Engaging counterpart of `train_dual_encoder_contrastive.py`. The two
differ ONLY in the data path (per the repo convention of keeping cluster-specific
entry points separate):

  * AMD baseline : NeighborsPrecomputedDataset(train_neighbors.vds) + simple_collate
  * Engaging     : NeighborsEfficientDataset(efficient_neighs)       + collate_neighbors

Both feed the model the identical 5-tuple (targets, samegals, sameins, masks,
metadata). `collate_neighbors` is what builds the neighbor mask that the efficient
dataset does not carry per-item.

Experiment axes (both selectable from the CLI):
  --encoder-pool {avg,conv1x1}   pooling variant added in dual_encoder_contrastive.py
  --random-neighbors {0,1}       dataset-level random conditioning set
                                 (NeighborsEfficientDataset.random_neighbors) — the
                                 faithful parallel to the flow-matching "random
                                 conditioning" ablation. The model's own spatial
                                 sameins path is used unchanged; only the CONTENTS
                                 of the neighbor set change.

Resume: writes `<ckpt-dir>/last.ckpt` continually (save_last) and resumes from it
if present, so the run survives being killed at the mit_normal_gpu wall limit and
continues under the self-chaining SLURM launcher until it reaches --max-steps.
"""

import argparse
import os
import sys
from pathlib import Path

# Avoid matplotlib / numba trying to write to non-writable $HOME on cluster nodes.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
# Single-threaded numpy/BLAS per process so DataLoader workers don't oversubscribe
# cores (N workers x N BLAS threads was a big part of the earlier stall).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pytorch_lightning as pl
import torch
torch.set_num_threads(1)

# hipBLASLt workaround — guarded so it is a harmless no-op on Engaging (NVIDIA).
try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        if os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "0":
            torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

_current_path = Path(__file__).resolve()
_project_root = _current_path.parents[3]  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.neighbors import collate_neighbors
from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)

# Baseline hyperparameters (match train_dual_encoder_contrastive.py so runs are
# comparable to the existing contrastive baseline).
# Default to the compact in-RAM 48x48 store (built by build_ram48.py): the 160x160
# mmap store is Ceph random-read bound (~0.35 batch/s -> ~78h/run); the RAM store
# trains at memory speed.
DEFAULT_DATA_DIR = "/orcd/pool/007/pablomer/efficient_neighs_48"
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 64
PROJECTION_DIM = 32
PROJECTION_HIDDEN_DIM = 64
IMAGE_SIZE = 48
MAX_NEIGHBORS = 5           # match the AMD precomputed shards (K=5)
TEMPERATURE = 0.1
VAL_FRACTION = 0.05
SEED = 42


class VerboseModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            print(f"[checkpoint] saved: {filepath}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder-pool", choices=["avg", "conv1x1"], default="avg")
    p.add_argument("--random-neighbors", type=int, choices=[0, 1], default=0)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--loader", choices=["ram48", "efficient"], default="ram48",
                   help="ram48 = compact in-RAM 48x48 store (fast); efficient = "
                        "original 160x160 mmap store (Ceph-bound, slow).")
    p.add_argument("--ckpt-dir", required=True,
                   help="Run checkpoint directory (should live on pool, not scratch).")
    p.add_argument("--run-name", required=True, help="WandB run name / tag.")
    p.add_argument("--wandb-project", default="galaxy-contrastive-engaging")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p.add_argument("--val-check-interval", type=int, default=1000)
    p.add_argument("--enable-umap", type=int, choices=[0, 1], default=1)
    # Smoke-test knobs.
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(SEED, workers=True)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] run directory: {ckpt_dir}", flush=True)
    print(f"[cfg] encoder_pool={args.encoder_pool} random_neighbors={bool(args.random_neighbors)} "
          f"max_steps={args.max_steps} batch_size={args.batch_size}", flush=True)

    # ---- data ----------------------------------------------------------------
    if args.loader == "ram48":
        from galaxy_images.galaxy_model.contrastive_baseline.neighbors_ram48 import (
            NeighborsRAM48Dataset,
        )
        dataset = NeighborsRAM48Dataset(
            data_dir=args.data_dir,
            crop_size=IMAGE_SIZE,
            max_neighbors=args.max_neighbors,
            random_neighbors=bool(args.random_neighbors),
            load_into_ram=True,
        )
    else:
        from galaxy_images.galaxy_model.neighbors_efficient import NeighborsEfficientDataset
        dataset = NeighborsEfficientDataset(
            data_dir=args.data_dir,
            crop_size=IMAGE_SIZE,
            max_neighbors=args.max_neighbors,
            random_neighbors=bool(args.random_neighbors),
        )
    total = len(dataset)
    val_size = max(1, int(total * args.val_fraction))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"[data] total={total:,}  train={train_size:,}  val={val_size:,}", flush=True)

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_neighbors,
        persistent_workers=persistent, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_neighbors,
        persistent_workers=persistent, pin_memory=True,
    )

    # ---- model ---------------------------------------------------------------
    model = DualEncoderContrastiveModule(
        in_channels=4,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        projection_hidden_dim=PROJECTION_HIDDEN_DIM,
        pretrained_encoder=False,
        encoder_pool=args.encoder_pool,
        image_size=IMAGE_SIZE,
        temperature_galaxy=TEMPERATURE,
        temperature_instrument=TEMPERATURE,
        lambda_galaxy=1.0,
        lambda_instrument=1.0,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        enable_umap_logging=bool(args.enable_umap),
        num_umap_batches=8,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        # random_neighbors is handled at the DATASET level, so the model keeps its
        # normal spatial sameins path for all four runs.
        use_random_instrument_positives=False,
    )

    # ---- logging -------------------------------------------------------------
    logger = False
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            log_model=False,
            config={
                "encoder_pool": args.encoder_pool,
                "random_neighbors": bool(args.random_neighbors),
                "batch_size": args.batch_size,
                "max_neighbors": args.max_neighbors,
                "embedding_dim": EMBEDDING_DIM,
                "projection_dim": PROJECTION_DIM,
                "lr": LR,
                "max_steps": args.max_steps,
                "data_dir": args.data_dir,
                "precision": "bf16-mixed",
                "cluster": "engaging",
            },
        )

    best_ckpt = VerboseModelCheckpoint(
        dirpath=str(ckpt_dir), monitor="val/loss", mode="min", save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}", auto_insert_metric_name=False,
    )
    # save_last keeps a rolling `last.ckpt` for resume across the 6h wall limit.
    periodic_ckpt = VerboseModelCheckpoint(
        dirpath=str(ckpt_dir), every_n_train_steps=500, save_top_k=1,
        filename="latest-step={step}", save_last=True, auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        callbacks=[best_ckpt, periodic_ckpt],
        num_sanity_val_steps=0,
        default_root_dir=str(ckpt_dir),
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
    )

    # Resume from the rolling checkpoint if a previous (killed) attempt left one.
    last_ckpt = ckpt_dir / "last.ckpt"
    resume_path = str(last_ckpt) if last_ckpt.exists() else None
    if resume_path:
        print(f"[resume] resuming from {resume_path}", flush=True)

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    # Mark completion so the self-chaining SLURM launcher stops resubmitting.
    if trainer.global_step >= args.max_steps:
        (ckpt_dir / "DONE").write_text(f"global_step={trainer.global_step}\n")
        print(f"[done] reached {trainer.global_step} steps; wrote DONE marker.", flush=True)
    else:
        print(f"[incomplete] stopped at global_step={trainer.global_step} "
              f"(< {args.max_steps}); a successor job should resume.", flush=True)


if __name__ == "__main__":
    main()
