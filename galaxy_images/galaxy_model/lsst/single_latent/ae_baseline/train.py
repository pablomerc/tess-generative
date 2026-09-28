"""Train the LSST AE baseline, then run the anomaly stage — one command.

  PYTHONPATH=. python -m galaxy_images.galaxy_model.lsst.single_latent.ae_baseline.train \
      [--steps 40000] [--batch 256] [--smoke]

WandB: entity tess-ml, project lsst-disentanglement (login via ~/.netrc).
Discord updates at start, every 5th validation, and the final anomaly report.
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

from .anomaly import run_anomaly_stage
from .data import FIELD_ARG, LsstGrizyDataset
from .model import LsstAE
from .notify import discord_post


class DiscordCallback(pl.Callback):
    def __init__(self, run_name, every=5):
        self.run_name, self.every, self.n = run_name, every, 0

    def on_fit_start(self, trainer, module):
        dim = next((module.hparams[k] for k in ("latent_dim", "token_dim",
                                                "dim") if k in module.hparams),
                   "?")
        discord_post(f"🚀 **{self.run_name}** started "
                     f"({type(module).__name__}, dim {dim}), "
                     f"{trainer.max_steps} steps on {trainer.num_devices} GPU")

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        self.n += 1
        if self.n % self.every == 0:
            m = {k: f"{v:.4f}" for k, v in trainer.callback_metrics.items()
                 if k.startswith("val/")}
            discord_post(f"📈 **{self.run_name}** step {trainer.global_step}: {m}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--field", choices=list(FIELD_ARG), default="all",
                    help="train on one field only (COSMOS / DEEP2-3)")
    ap.add_argument("--patience", type=int, default=3,
                    help="early-stop after N validations without val/mse "
                         "improvement (0 = disabled); first run overfit "
                         "past step 14k")
    ap.add_argument("--smoke", action="store_true",
                    help="200 steps, no wandb/discord, tiny val")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    field = FIELD_ARG[args.field]
    run_name = f"lsst-ae-{args.field}-z{args.latent}-{stamp}"
    run_dir = os.path.join("/work1/jeroenaudenaert/pablomer/outputs/"
                           "lsst_ae_baseline", run_name)
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

    model = LsstAE(latent_dim=args.latent, lr=args.lr)
    callbacks = [ModelCheckpoint(dirpath=os.path.join(run_dir, "checkpoints"),
                                 monitor="val/mse", mode="min",
                                 save_last=True, save_top_k=1,
                                 filename="best-{step}")]
    if args.patience and not args.smoke:
        callbacks.append(EarlyStopping(monitor="val/mse", mode="min",
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
        model = LsstAE.load_from_checkpoint(best)
    if not args.smoke:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            run_anomaly_stage(model, run_dir, device, train_field=field)
        except Exception as e:
            discord_post(f"❌ **{run_name}**: anomaly stage failed: {e}")
            raise
        discord_post(f"✅ **{run_name}** complete. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
