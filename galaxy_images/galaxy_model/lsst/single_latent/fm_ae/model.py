"""Joint flow-matching autoencoder for LSST grizy stamps.

Architecture follows the main neighbors flow-matching baseline
(double_train_fm_neighbors.ConditionalFlowMatchingModule, variant
neighbors_all_attn) — NOT the embedding_fm_decoder AdaGN probe:
  - its ResNetEncoder producing spatial tokens: layer4 (B,512,2,2) ->
    Conv1x1 -> (B, 4, token_dim) cross-attention tokens, trained jointly;
  - UNet2DConditionModel velocity model, ALL CrossAttn blocks,
    model_channels 128, mult (1,2,4,4), layers_per_block 2, head_dim 8,
    timesteps = t * 1000, rectified-flow loss, Euler sampler (250 steps).

Bottleneck matched to the AE baseline: token_dim=32 x 4 tokens = 128
numbers; encode() flattens tokens to a 128-d latent for anomaly/UMAP work
(dim-matched to the AE's z for NSF / probe comparisons).
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from galaxy_images.galaxy_model.double_train_fm_neighbors import ResNetEncoder

from ..ae_baseline.data import FIELDS
from ..ae_baseline.model import LsstAE


class LsstFmAE(pl.LightningModule):
    def __init__(self, token_dim: int = 32, in_channels: int = 5,
                 lr: float = 1e-4, weight_decay: float = 1e-4,
                 model_channels: int = 128, sample_steps: int = 250,
                 val_sample_steps: int = 50, umap_points: int = 3000):
        super().__init__()
        self.save_hyperparameters()
        from diffusers import UNet2DConditionModel
        self.encoder = ResNetEncoder(in_channels=in_channels,
                                     cross_attention_dim=token_dim)
        c = model_channels
        self.unet = UNet2DConditionModel(
            sample_size=48, in_channels=in_channels, out_channels=in_channels,
            block_out_channels=(c, 2 * c, 4 * c, 4 * c), layers_per_block=2,
            down_block_types=("CrossAttnDownBlock2D",) * 4,
            up_block_types=("CrossAttnUpBlock2D",) * 4,
            attention_head_dim=8, cross_attention_dim=token_dim,
        )

    # --------------------------------------------------------------- core
    def tokens(self, x):
        return self.encoder(x)                       # (B, 4, token_dim)

    def encode(self, x):
        return self.tokens(x).flatten(1)             # (B, 4*token_dim) = 128

    def velocity(self, x_t, t, tok):
        return self.unet(x_t, t * 1000.0,
                         encoder_hidden_states=tok).sample

    def _fm_loss(self, x1, tok):
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        tb = t.view(-1, 1, 1, 1)
        x_t = (1 - tb) * x0 + tb * x1
        v = self.velocity(x_t, t, tok)
        return F.mse_loss(v, x1 - x0, reduction="none").mean((1, 2, 3))

    @torch.no_grad()
    def sample(self, tok, steps=None, x=None):
        steps = steps or self.hparams.sample_steps
        if x is None:
            x = torch.randn(tok.shape[0], self.hparams.in_channels, 48, 48,
                            device=tok.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((tok.shape[0],), i * dt, device=tok.device)
            x = x + self.velocity(x, t, tok) * dt
        return x

    @torch.no_grad()
    def reconstruct(self, x, steps=None):
        return self.sample(self.tokens(x), steps=steps)

    # ----------------------------------------------------------- training
    def training_step(self, batch, _):
        loss = self._fm_loss(batch["x"], self.tokens(batch["x"])).mean()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val = {"z": [], "field": [], "ninp": [], "psf": [],
                     "loss": [], "examples": None}

    def validation_step(self, batch, batch_idx):
        tok = self.tokens(batch["x"])
        per_obj = self._fm_loss(batch["x"], tok)
        v = self._val
        v["loss"].append(torch.stack([per_obj,
                                      batch["field"].float()], 1).cpu())
        if sum(t.shape[0] for t in v["z"]) < self.hparams.umap_points:
            v["z"].append(tok.flatten(1).float().cpu())
            v["field"].append(batch["field"].cpu())
            v["ninp"].append(batch["ninp"].cpu())
            v["psf"].append(batch["psf"].cpu())
        if batch_idx == 0:
            v["examples"] = (batch["x"][:8], tok[:8])

    def on_validation_epoch_end(self):
        v = self._val
        if not v["loss"]:
            return
        m = torch.cat(v["loss"])
        self.log("val/fm_loss", m[:, 0].mean())
        for fi, fname in enumerate(FIELDS):
            sel = m[:, 1] == fi
            if sel.any():
                key = fname.lower().replace("-", "")
                self.log(f"val/fm_loss_{key}", m[sel, 0].mean())
        if self.trainer.sanity_checking:
            return
        try:
            self._log_figures(v)
        except Exception as e:  # figures must never kill training
            print(f"[val figures] skipped: {e}")

    # ------------------------------------------------------------- figures
    def _log_figures(self, v):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        logs = {}
        if v["examples"] is not None:
            x, tok = v["examples"]
            xh = self.sample(tok, steps=self.hparams.val_sample_steps).float()
            self.log("val/sample_mse", F.mse_loss(xh, x))
            x, xh = x.float().cpu(), xh.cpu()
            n = x.shape[0]
            fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5))
            for k in range(n):
                axes[0, k].imshow(LsstAE._rgb(x[k]), origin="lower")
                axes[1, k].imshow(LsstAE._rgb(xh[k]), origin="lower")
                res = (x[k] - xh[k]).abs().sum(0).numpy()
                axes[2, k].imshow(res, origin="lower", cmap="inferno")
                for r_ in range(3):
                    axes[r_, k].set_xticks([]), axes[r_, k].set_yticks([])
            for r_, lab in enumerate(["input", "FM sample", "|residual|"]):
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
                s = ax.scatter(emb[:, 0], emb[:, 1], c=c, s=2, cmap=cmap,
                               lw=0)
                ax.set_title(name), ax.set_xticks([]), ax.set_yticks([])
                fig.colorbar(s, ax=ax, fraction=0.046)
            fig.suptitle(f"FM-AE latent UMAP (step {self.global_step})")
            fig.tight_layout()
            logs["figures/latent_umap"] = wandb.Image(fig)
            plt.close(fig)

        if logs and self.logger is not None:
            self.logger.experiment.log(logs, commit=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)
