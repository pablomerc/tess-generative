"""
Stage 2 — train the counterfactual embedding-conditioned FM decoder.

Reconstructs galaxy images from frozen contrastive embeddings of samegal (physics)
and sameins neighbors (instrument). Never encodes the target.

Defaults target cond-style=tokens + injection=xattn (plan B). For the mean-pool
AdaGN control (plan A): --cond-style meanpool --injection adagn.

Example:
    python -m galaxy_images.galaxy_model.contrastive_baseline.counterfactual_fm_decoder.train_cf_fm \
        --data-h5 /orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5 \
        --cond-style tokens --injection xattn --max-steps 100000 --devices 1
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

try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parents[4]  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.contrastive_baseline.counterfactual_fm_decoder.cf_embedding_dataset import (
    CFEmbeddingDataset,
    cf_collate,
    make_train_val_split,
)
from galaxy_images.galaxy_model.contrastive_baseline.counterfactual_fm_decoder.cf_fm_module import (
    CFEmbeddingFlowMatching,
)

DEFAULT_OUTPUT_DIR = "/orcd/pool/007/pablomer/outputs/counterfactual_fm/runs"


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train counterfactual embedding FM decoder.")
    ap.add_argument("--data-h5", required=True)
    ap.add_argument("--cond-style", default="tokens", choices=["tokens", "meanpool"])
    ap.add_argument("--injection", default=None,
                    help="Default: xattn for tokens, adagn for meanpool.")
    # run / logging
    ap.add_argument("--wandb-project", default="galaxy-counterfactual-fm-decoder")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume-from", default=None)
    # optimization
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.05,
                    help="Match data_factory / paper / fm-control (0.05).")
    ap.add_argument("--monitor", default="val/mse",
                    help="Checkpoint selection metric. Default val/mse matches fm-control's "
                         "monitor_metric so 'best' means the same thing on both sides of the "
                         "recon-MSE comparison; val/loss selects on velocity MSE instead.")
    ap.add_argument("--allow-partial-h5", action="store_true",
                    help="Permit training on an H5 that does not cover every anchor of its "
                         "source store. Such a run's val split does NOT match "
                         "data_factory/paper/fm-control, so it cannot be compared to them.")
    ap.add_argument("--attn-mask-pads", action="store_true",
                    help="Mask pad neighbor tokens out of cross-attention. OFF by default for "
                         "parity with the paper model, which only zeroes them.")
    ap.add_argument("--num-integration-steps", type=int, default=250)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--devices", type=int, default=int(os.environ.get("N_DEVICES", 1)))
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--val-check-interval", type=int, default=1000)
    ap.add_argument("--ckpt-every-n-steps", type=int, default=500)
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

    if args.injection is None:
        args.injection = "xattn" if args.cond_style == "tokens" else "adagn"

    run_name = args.wandb_name or f"cf-fm-{args.cond_style}-{args.injection}"

    if args.resume_from is not None:
        run_dir = Path(args.resume_from).parent.parent
    elif args.run_dir is not None:
        run_dir = Path(args.run_dir)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _build_run_dir(args.output_dir, run_name)
    print(f"[train-cf] run_dir={run_dir}")
    print(f"[train-cf] cond_style={args.cond_style} injection={args.injection}")

    full = CFEmbeddingDataset(args.data_h5, cond_style=args.cond_style)
    if not full.is_full_store and not args.allow_partial_h5:
        raise SystemExit(
            f"[train-cf] {args.data_h5} holds {len(full):,} of "
            f"{full.n_anchors_total or '?'} source anchors. The seed-{args.seed} "
            f"{args.val_ratio:.0%} split only reproduces data_factory's (and therefore only "
            f"stays disjoint from the eval val set) on a FULL corpus — training on a subset "
            f"would leak eval galaxies into training. Regenerate the H5, or pass "
            f"--allow-partial-h5 for a throwaway smoke run."
        )
    train_ds, val_ds = make_train_val_split(full, val_ratio=args.val_ratio, seed=args.seed)
    print(
        f"[train-cf] train={len(train_ds):,}  val={len(val_ds):,}  "
        f"(store={full.source_loader}, full={full.is_full_store})"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=cf_collate, drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=cf_collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = CFEmbeddingFlowMatching(
        cond_style=args.cond_style,
        cond_dim=full.emb_dim if args.cond_style == "tokens" else full.cond_dim,
        vector_cond_dim=full.vector_cond_dim,
        in_channels=4,
        image_size=48,
        injection=args.injection,
        cross_attention_dim=full.emb_dim,
        lr=args.lr,
        num_integration_steps=args.num_integration_steps,
        attn_mask_pads=args.attn_mask_pads,
    )

    if args.no_wandb:
        logger = None
    else:
        logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            version=run_name,
            resume="allow",
            log_model=False,
            config=vars(args),
        )

    print(f"[train-cf] checkpoint monitor={args.monitor} (fm-control selects on val/mse)")
    best_ckpt = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", monitor=args.monitor, mode="min", save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}", auto_insert_metric_name=False,
    )
    periodic_ckpt = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", every_n_train_steps=args.ckpt_every_n_steps,
        save_top_k=1, filename="latest-step={step}", save_last=True,
        auto_insert_metric_name=False,
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

    if trainer.is_global_zero and args.max_steps > 0 and trainer.global_step >= args.max_steps:
        (run_dir / "DONE").touch()
        print(f"[train-cf] reached {trainer.global_step} steps — wrote {run_dir / 'DONE'}")


if __name__ == "__main__":
    main()
