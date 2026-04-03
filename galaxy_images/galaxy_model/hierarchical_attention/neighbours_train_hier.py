"""
Standalone training script for the hierarchical attention neighbors model.

Usage:
    python neighbours_train_hier.py
    python neighbours_train_hier.py --experiment bn_36x64 --max_steps 500000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
# Force regular hipBLAS instead of hipBLASLt — buggy on MI210
torch.backends.cuda.preferred_blas_library("hipblas")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from galaxy_images.galaxy_model.neighbors import (
    NeighborsPrecomputedDataset,
    simple_collate,
)
from galaxy_images.galaxy_model.hierarchical_attention.train_experiments import (
    EXPERIMENTS,
    BACKBONE_RES,
)
from galaxy_images.galaxy_model.hierarchical_attention.double_train_fm_neighbors_hier import (
    HierarchicalFlowMatchingModule,
)


def is_h100_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    for i in range(torch.cuda.device_count()):
        if "h100" in torch.cuda.get_device_name(i).lower():
            return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train hierarchical attention neighbors model.",
    )
    parser.add_argument(
        "--experiment", type=str, default="bn_36x16",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment config name (default: bn_36x16)",
    )
    parser.add_argument("--max_steps", type=int, default=1_500_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--precomputed_h5", type=str,
        default="/work1/jeroenaudenaert/pablomer/data/train_neighbors.vds",
    )
    parser.add_argument(
        "--wandb_project", type=str,
        default="galaxy-flow-matching-neighbours",
    )
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str,
        default="galaxy_images/galaxy_model/hierarchical_attention/outputs",
    )
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument(
        "--scale_steps_by_devices", action="store_true", default=True,
    )
    parser.add_argument(
        "--no_scale_steps", dest="scale_steps_by_devices", action="store_false",
    )

    args = parser.parse_args(argv)

    pl.seed_everything(args.seed, workers=True)

    cfg = EXPERIMENTS[args.experiment]

    # --- Print experiment info ---
    total_spatial = sum(
        (BACKBONE_RES[idx]
         // (2 if cfg.get("reductions", {}).get(idx) else 1)) ** 2
        * cfg["token_dim"]
        for idx in cfg["spatial_indices"]
    )
    total_cond = total_spatial + cfg["global_dim"]
    input_vals = 4 * 48 * 48
    ratio = input_vals / total_cond
    tag = (f"{ratio:.1f}x compression" if ratio > 1
           else f"{1/ratio:.1f}x expansion")

    print(f"\n{'='*60}")
    print(f"Hierarchical Neighbors — Experiment: {args.experiment}")
    print(f"  {total_cond} conditioning values ({tag})")
    print(f"{'='*60}\n")

    # --- Data ---
    print(f"Loading data from {args.precomputed_h5}...")
    dataset = NeighborsPrecomputedDataset(args.precomputed_h5)

    total_size = len(dataset)
    val_size = int(total_size * args.val_ratio)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator,
    )

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=simple_collate,
        persistent_workers=persistent,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=simple_collate,
        persistent_workers=persistent,
        pin_memory=True,
    )

    # --- Adjust for hardware ---
    h100 = is_h100_gpu()
    runtime_batch_size = args.batch_size
    runtime_precision = args.precision

    # --- Model ---
    model = HierarchicalFlowMatchingModule(
        experiment_config=cfg,
        lr=args.lr,
        num_sample_images=10,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=1.0,
        lambda_geometric=0.0,
        num_umap_batches=8,
        mask_center=False,
    )

    enc1_p = sum(p.numel() for p in model.encoder_1.parameters())
    enc2_p = sum(p.numel() for p in model.encoder_2.parameters())
    unet_p = sum(p.numel() for p in model.conditioned_unet.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Parameter count:")
    print(f"  Encoder 1 (galaxy):      {enc1_p / 1e6:.1f}M")
    print(f"  Encoder 2 (neighbors):   {enc2_p / 1e6:.1f}M")
    print(f"  UNet:                    {unet_p / 1e6:.1f}M")
    print(f"  Total:                   {total_p / 1e6:.1f}M")

    # --- Max steps (optionally scale by device count) ---
    max_steps = args.max_steps
    n_devices = args.devices
    if n_devices > 1 and args.scale_steps_by_devices:
        max_steps = max(1, int(max_steps / n_devices))

    strategy = "ddp_find_unused_parameters_true" if n_devices > 1 else "auto"

    # --- Output dir + checkpoints ---
    run_dir = Path(args.output_dir) / args.experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    periodic_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_train_steps=1000,
        save_top_k=1,
        filename="latest-step={step}",
        save_last=False,
    )

    # --- W&B ---
    wandb_name = args.wandb_name or f"hier-{args.experiment}"
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=wandb_name,
        log_model=False,
        config={
            "experiment": args.experiment,
            **{k: str(v) for k, v in cfg.items()},
            "total_conditioning_values": total_cond,
            "compression": tag,
            "encoder1_params_M": enc1_p / 1e6,
            "encoder2_params_M": enc2_p / 1e6,
            "unet_params_M": unet_p / 1e6,
            "total_params_M": total_p / 1e6,
            "batch_size": runtime_batch_size,
            "max_steps": max_steps,
            "h100_detected": h100,
            "precision": runtime_precision,
        },
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_steps=max_steps,
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        strategy=strategy,
        log_every_n_steps=10,
        precision=runtime_precision,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[best_ckpt, periodic_ckpt],
        num_sanity_val_steps=0,
    )

    print(f"Experiment: {args.experiment}")
    print(f"Data: {args.precomputed_h5}")
    print(f"Batch size: {runtime_batch_size}")
    print(f"Precision: {runtime_precision}")
    print(f"Devices: {n_devices} | Strategy: {strategy}")
    print(f"Max steps: {max_steps}")
    print(f"H100 detected: {h100}")
    print(f"Run dir: {run_dir}")

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
