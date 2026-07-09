"""
Stage 2 — train the embedding-conditioned flow-matching decoder.

Reconstructs galaxy images from the frozen contrastive embeddings precomputed in Stage 1.
Logs to a SEPARATE W&B project (default: galaxy-embedding-fm-decoder) with the same metric
family as the neighbors FM model: train/val loss (+ per-survey), val/mse{,_32,_hsc,_legacy},
val/recon_grid{,_row_scaled}, val/embedding_umap.

Resumable: a periodic checkpoint with save_last=True writes <run-dir>/checkpoints/last.ckpt
(model + optimizer + scheduler + global_step). Resume with --resume-from that path; the W&B
run continues (resume="allow", stable id). The 6 h SLURM wrapper auto-resubmits until done.

Example:
    python -m galaxy_images.galaxy_model.contrastive_baseline.embedding_fm_decoder.train_embedding_fm \
        --data-h5 /orcd/pool/007/pablomer/outputs/embedding_fm/anchor_embeddings_100k.h5 \
        --injection adagn --cond-mode concat128 --max-steps 100000 --devices 1
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parents[4]  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.contrastive_baseline.embedding_fm_decoder.embedding_dataset import (
    EmbeddingImageDataset,
    embedding_collate,
    make_train_val_split,
)
from galaxy_images.galaxy_model.contrastive_baseline.embedding_fm_decoder.embedding_fm_module import (
    EmbeddingConditionedFlowMatching,
)

DEFAULT_OUTPUT_DIR = "/orcd/pool/007/pablomer/outputs/embedding_fm/runs"


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train embedding-conditioned FM decoder.")
    ap.add_argument("--data-h5", required=True)
    ap.add_argument("--injection", default="adagn", choices=["adagn", "hybrid", "xattn", "concat"])
    ap.add_argument("--cond-mode", default="concat128", choices=["concat128", "galaxy64", "instrument64"])
    # run / logging
    ap.add_argument("--wandb-project", default="galaxy-embedding-fm-decoder")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--run-dir", default=None, help="Fixed run dir (required for auto-resubmit resume).")
    ap.add_argument("--resume-from", default=None)
    # optimization
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--num-integration-steps", type=int, default=250)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--devices", type=int, default=int(os.environ.get("N_DEVICES", 1)))
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--val-check-interval", type=int, default=1000)
    ap.add_argument("--ckpt-every-n-steps", type=int, default=500)
    ap.add_argument("--n-cond-tokens", type=int, default=2)
    return ap.parse_args(argv)


def _build_run_dir(output_dir: str, run_name: str) -> Path:
    base = Path(output_dir) / run_name
    today = date.today().strftime("%Y-%m-%d")
    candidate = base / today
    n = 2
    while candidate.exists():
        candidate = base / f"{today}_{n}"
        n += 1
    (candidate / "checkpoints").mkdir(parents=True, exist_ok=True)
    return candidate


def main(argv=None):
    args = _parse_args(argv)
    pl.seed_everything(args.seed, workers=True)

    run_name = args.wandb_name or f"emb-fm-{args.injection}-{args.cond_mode}"

    # Resume reuses the existing run dir; otherwise use --run-dir (stable, for resubmit) or a fresh one.
    if args.resume_from is not None:
        run_dir = Path(args.resume_from).parent.parent
    elif args.run_dir is not None:
        run_dir = Path(args.run_dir)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _build_run_dir(args.output_dir, run_name)
    print(f"[train] run_dir={run_dir}")

    # ---- data
    full = EmbeddingImageDataset(args.data_h5, cond_mode=args.cond_mode)
    train_ds, val_ds = make_train_val_split(full, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[train] train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=embedding_collate, drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=embedding_collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ---- model
    model = EmbeddingConditionedFlowMatching(
        cond_dim=full.cond_dim,
        in_channels=4,
        image_size=48,
        injection=args.injection,
        n_cond_tokens=args.n_cond_tokens,
        lr=args.lr,
        num_integration_steps=args.num_integration_steps,
    )

    # ---- logger (separate project; stable id so auto-resubmit continues the same run)
    if args.no_wandb:
        logger = None
    else:
        logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            version=run_name,  # stable run id -> resume="allow" continues it
            resume="allow" if args.resume_from is not None else "never",
            log_model=False,
            config=vars(args),
        )

    # ---- checkpoints: best (val/loss) + periodic with save_last (resume anchor)
    best_ckpt = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", monitor="val/loss", mode="min", save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}", auto_insert_metric_name=False,
    )
    periodic_ckpt = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", every_n_train_steps=args.ckpt_every_n_steps,
        save_top_k=1, filename="latest-step={step}", save_last=True,
    )

    devices = args.devices
    strategy = "ddp_find_unused_parameters_true" if devices > 1 else "auto"
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        logger=logger,
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        callbacks=[best_ckpt, periodic_ckpt],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)

    # Signal completion for the auto-resubmit wrapper (only when we actually reached the target).
    if trainer.is_global_zero and args.max_steps > 0 and trainer.global_step >= args.max_steps:
        (run_dir / "DONE").touch()
        print(f"[train] reached {trainer.global_step} steps — wrote {run_dir/'DONE'}")


if __name__ == "__main__":
    main()
