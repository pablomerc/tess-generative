"""Train the LSST flow-matching AE, then run the anomaly stage.

  PYTHONPATH=. python -m galaxy_images.galaxy_model.lsst.single_latent.fm_ae.train \
      [--field cosmos|deep23|all] [--steps 30000] [--smoke]

Anomaly rankings are latent-based (NSF + IForest on the flattened 128-d
token latent); reconstructions appear only in the top-K galleries via
250-step Euler sampling. WandB: tess-ml/lsst-disentanglement.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")   # MI210 workaround

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

try:                                                        # AMD only; no-op guard
    torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

from ..ae_baseline.anomaly import run_anomaly_stage
from ..ae_baseline.data import FIELD_ARG, LsstGrizyDataset
from ..ae_baseline.notify import discord_post
from ..ae_baseline.train import DiscordCallback
from .model import LsstFmAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64,
                    help="the all-CrossAttn UNet OOMs on MI210 above ~64 "
                         "(attention over 48x48 positions); the neighbors "
                         "baseline also trains at 64")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--token-dim", type=int, default=32)    # 4 tokens -> 128
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--field", choices=list(FIELD_ARG), default="all")
    ap.add_argument("--patience", type=int, default=5,
                    help="early-stop patience in validations on val/fm_loss "
                         "(0 = off); FM val loss is noisier than AE MSE")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    field = FIELD_ARG[args.field]
    run_name = f"lsst-fmae-{args.field}-tok4x{args.token_dim}-{stamp}"
    run_dir = os.path.join("/work1/jeroenaudenaert/pablomer/outputs/"
                           "lsst_fm_ae", run_name)
    os.makedirs(run_dir, exist_ok=False)                    # no-overwrite
    pl.seed_everything(0, workers=True)

    train_ds = LsstGrizyDataset("train", field=field)
    val_ds = LsstGrizyDataset("val", field=field)
    print(f"train {len(train_ds)}, val {len(val_ds)}")
    with open(os.path.join(run_dir, "val_object_ids.txt"), "w") as f:
        f.writelines(f"{i}\n" for i in val_ds.ids)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    model = LsstFmAE(token_dim=args.token_dim, lr=args.lr)
    callbacks = [ModelCheckpoint(dirpath=os.path.join(run_dir, "checkpoints"),
                                 monitor="val/fm_loss", mode="min",
                                 save_last=True, save_top_k=1,
                                 filename="best-{step}")]
    if args.patience and not args.smoke:
        callbacks.append(EarlyStopping(monitor="val/fm_loss", mode="min",
                                       patience=args.patience))
    logger = False
    if not args.smoke:
        logger = WandbLogger(entity="tess-ml", project="lsst-disentanglement",
                             name=run_name, save_dir=run_dir)
        callbacks.append(DiscordCallback(run_name))

    trainer = pl.Trainer(
        max_steps=200 if args.smoke else args.steps,
        val_check_interval=50 if args.smoke else args.val_every,
        check_val_every_n_epoch=None,
        limit_val_batches=2 if args.smoke else 1.0,
        accelerator="auto", devices=1, precision="bf16-mixed",
        logger=logger, callbacks=callbacks, log_every_n_steps=50,
        default_root_dir=run_dir,
    )
    trainer.fit(model, train_dl, val_dl)

    best = callbacks[0].best_model_path or callbacks[0].last_model_path
    print(f"best checkpoint: {best}")
    if best:
        model = LsstFmAE.load_from_checkpoint(best)
    if not args.smoke:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            run_anomaly_stage(
                model, run_dir, device, train_field=field,
                gallery_recon_fn=lambda m, x: m.reconstruct(x))
        except Exception as e:
            discord_post(f"❌ **{run_name}**: anomaly stage failed: {e}")
            raise
        discord_post(f"✅ **{run_name}** complete. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
