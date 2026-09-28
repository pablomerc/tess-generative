"""LSST autoencoder baseline: contrastive-matched ResNet18 encoder + residual
upsample decoder (no skip connections — the 128-d bottleneck carries all
information, so latents and recon errors are meaningful for anomaly work).
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    ResNetEncoder)

from .data import FIELDS


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ResUpDecoder(nn.Module):
    """latent (B,128) -> (B,5,48,48): Linear to (512,3,3), then 4x
    [nearest-upsample x2 + 2 ResBlocks], 512->256->128->64->32, final conv."""

    def __init__(self, latent_dim: int = 128, out_channels: int = 5):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 3 * 3)
        chans = [(512, 256), (256, 128), (128, 64), (64, 32)]
        stages = []
        for cin, cout in chans:
            stages += [nn.Upsample(scale_factor=2, mode="nearest"),
                       ResBlock(cin, cout), ResBlock(cout, cout)]
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(),
                                  nn.Conv2d(32, out_channels, 3, padding=1))

    def forward(self, z):
        h = self.fc(z).view(-1, 512, 3, 3)
        return self.head(self.stages(h))


class LsstAE(pl.LightningModule):
    def __init__(self, latent_dim: int = 128, in_channels: int = 5,
                 lr: float = 3e-4, weight_decay: float = 1e-4,
                 umap_points: int = 3000):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = ResNetEncoder(in_channels=in_channels,
                                     embedding_dim=latent_dim, pool="avg")
        self.decoder = ResUpDecoder(latent_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def training_step(self, batch, _):
        xhat, _ = self(batch["x"])
        loss = F.mse_loss(xhat, batch["x"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    # ---------------------------------------------------------- validation
    def on_validation_epoch_start(self):
        self._val = {"z": [], "field": [], "ninp": [], "psf": [],
                     "mse": [], "examples": None}

    def validation_step(self, batch, batch_idx):
        xhat, z = self(batch["x"])
        per_obj = F.mse_loss(xhat, batch["x"], reduction="none").mean((1, 2, 3))
        v = self._val
        v["mse"].append(torch.stack([per_obj, batch["field"].float()], 1).cpu())
        if sum(t.shape[0] for t in v["z"]) < self.hparams.umap_points:
            v["z"].append(z.float().cpu())
            v["field"].append(batch["field"].cpu())
            v["ninp"].append(batch["ninp"].cpu())
            v["psf"].append(batch["psf"].cpu())
        if batch_idx == 0:
            v["examples"] = (batch["x"][:8].float().cpu(),
                             xhat[:8].float().cpu())

    def on_validation_epoch_end(self):
        v = self._val
        if not v["mse"]:
            return
        m = torch.cat(v["mse"])
        self.log("val/mse", m[:, 0].mean())
        for fi, fname in enumerate(FIELDS):
            sel = m[:, 1] == fi
            if sel.any():
                key = fname.lower().replace("-", "")
                self.log(f"val/mse_{key}", m[sel, 0].mean())
        if self.trainer.sanity_checking:
            return
        try:
            self._log_figures(v)
        except Exception as e:  # figures must never kill training
            print(f"[val figures] skipped: {e}")

    # ------------------------------------------------------------- figures
    @staticmethod
    def _rgb(x5):  # normalized (5,48,48) -> display RGB from i,r,g
        rgb = torch.stack([x5[2], x5[1], x5[0]], -1).numpy()
        return np.clip((rgb + 1.0) / 5.0, 0, 1)

    def _log_figures(self, v):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        logs = {}
        if v["examples"] is not None:
            x, xh = v["examples"]
            n = x.shape[0]
            fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5))
            for k in range(n):
                axes[0, k].imshow(self._rgb(x[k]), origin="lower")
                axes[1, k].imshow(self._rgb(xh[k]), origin="lower")
                res = (x[k] - xh[k]).abs().sum(0).numpy()
                axes[2, k].imshow(res, origin="lower", cmap="inferno")
                for r_ in range(3):
                    axes[r_, k].set_xticks([]), axes[r_, k].set_yticks([])
            for r_, lab in enumerate(["input", "recon", "|residual|"]):
                axes[r_, 0].set_ylabel(lab)
            fig.tight_layout()
            logs["figures/recon_grid"] = wandb.Image(fig)
            plt.close(fig)

        if v["z"]:
            import umap
            z = torch.cat(v["z"]).numpy()
            emb = umap.UMAP(n_neighbors=30, min_dist=0.1,
                            random_state=0).fit_transform(z)
            colors = [("field", torch.cat(v["field"]).numpy(), "coolwarm"),
                      ("n_input_i", torch.cat(v["ninp"]).numpy(), "viridis"),
                      ("psf_fwhm_i", torch.cat(v["psf"]).numpy(), "magma")]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
            for ax, (name, c, cmap) in zip(axes, colors):
                s = ax.scatter(emb[:, 0], emb[:, 1], c=c, s=2, cmap=cmap, lw=0)
                ax.set_title(name), ax.set_xticks([]), ax.set_yticks([])
                fig.colorbar(s, ax=ax, fraction=0.046)
            fig.suptitle(f"latent UMAP (step {self.global_step})")
            fig.tight_layout()
            logs["figures/latent_umap"] = wandb.Image(fig)
            plt.close(fig)

        if logs and self.logger is not None:
            self.logger.experiment.log(logs, commit=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)
