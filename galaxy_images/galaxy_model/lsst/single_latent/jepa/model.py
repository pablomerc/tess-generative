"""I-JEPA for LSST grizy stamps (48x48x5, patch 4 -> 12x12 = 144 tokens).

Self-contained small ViT (no timm dependency): context encoder + EMA target
encoder + narrow predictor. Per-step block masking on the patch grid
(shared across the batch): 4 target blocks of 15-20% area, aspect 0.75-1.5;
context = an 85-100% block minus all target patches. Loss = MSE between
predictor outputs and target-encoder features at the masked positions.

encode(x) = target-encoder forward, mean over ALL tokens (no token cap,
per the AION convention) -> (B, embed_dim) latent for anomaly/UMAP work.
No decoder: anomaly rankings are latent-only (NSF + IForest).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ae_baseline.data import FIELDS

GRID = 12                                    # 48 / patch_size(4)


def sincos_pos_embed(dim, grid=GRID):
    """(grid*grid, dim) fixed 2D sin-cos positional embedding."""
    def enc1d(pos, d):
        omega = 1.0 / (10000 ** (np.arange(d // 2) / (d / 2.0)))
        out = np.einsum("p,f->pf", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], 1)
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    emb = np.concatenate([enc1d(ys.ravel(), dim // 2),
                          enc1d(xs.ravel(), dim // 2)], 1)
    return torch.tensor(emb, dtype=torch.float32)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    def __init__(self, dim=384, depth=8, heads=6, in_channels=5, patch=4):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, dim, patch, stride=patch)
        self.register_buffer("pos", sincos_pos_embed(dim), persistent=False)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, keep=None):
        """keep: optional LongTensor of patch indices (shared across batch)."""
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        if keep is not None:
            t = t[:, keep]
        for b in self.blocks:
            t = b(t)
        return self.norm(t)                          # (B, n_keep, dim)


class Predictor(nn.Module):
    def __init__(self, dim=384, pdim=192, depth=4, heads=6):
        super().__init__()
        self.inp = nn.Linear(dim, pdim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pdim))
        self.register_buffer("pos", sincos_pos_embed(pdim), persistent=False)
        self.blocks = nn.ModuleList(Block(pdim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(pdim)
        self.out = nn.Linear(pdim, dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, ctx, ctx_idx, tgt_idx):
        B = ctx.shape[0]
        h = self.inp(ctx) + self.pos[ctx_idx]
        m = self.mask_token.expand(B, len(tgt_idx), -1) + self.pos[tgt_idx]
        h = torch.cat([h, m], 1)
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h[:, -len(tgt_idx):]))


def _rand_block(gen, lo, hi, aspect=(0.75, 1.5)):
    """Random rectangle on the GRID x GRID grid covering lo-hi area frac."""
    frac = lo + (hi - lo) * torch.rand((), generator=gen).item()
    a = aspect[0] + (aspect[1] - aspect[0]) * torch.rand((), generator=gen).item()
    n = frac * GRID * GRID
    h = max(1, min(GRID, round(math.sqrt(n * a))))
    w = max(1, min(GRID, round(math.sqrt(n / a))))
    y = torch.randint(0, GRID - h + 1, (), generator=gen).item()
    x = torch.randint(0, GRID - w + 1, (), generator=gen).item()
    ys, xs = np.meshgrid(range(y, y + h), range(x, x + w), indexing="ij")
    return set((ys * GRID + xs).ravel().tolist())


def sample_masks(gen):
    """-> (context_idx, target_idx) LongTensors, target = union of 4 blocks."""
    tgt = set()
    for _ in range(4):
        tgt |= _rand_block(gen, 0.15, 0.2)
    ctx = _rand_block(gen, 0.85, 1.0, aspect=(1.0, 1.0)) - tgt
    if not ctx:                                  # degenerate; keep something
        ctx = set(range(GRID * GRID)) - tgt
    return (torch.tensor(sorted(ctx), dtype=torch.long),
            torch.tensor(sorted(tgt), dtype=torch.long))


class LsstJepa(pl.LightningModule):
    def __init__(self, dim: int = 384, depth: int = 8, heads: int = 6,
                 pred_dim: int = 192, pred_depth: int = 4,
                 in_channels: int = 5, lr: float = 5e-4,
                 weight_decay: float = 0.05, warmup_steps: int = 1000,
                 ema_start: float = 0.996, umap_points: int = 3000):
        super().__init__()
        self.save_hyperparameters()
        self.enc = ViT(dim, depth, heads, in_channels)
        self.target_enc = copy.deepcopy(self.enc)
        for p in self.target_enc.parameters():
            p.requires_grad = False
        self.pred = Predictor(dim, pred_dim, pred_depth, heads)
        self._gen = None

    # --------------------------------------------------------------- core
    @torch.no_grad()
    def encode(self, x):
        return self.target_enc(x).mean(1)            # (B, dim)

    def _step_loss(self, x, gen):
        ctx_idx, tgt_idx = sample_masks(gen)
        ctx_idx, tgt_idx = ctx_idx.to(x.device), tgt_idx.to(x.device)
        with torch.no_grad():
            target = self.target_enc(x)[:, tgt_idx]
        ctx = self.enc(x, keep=ctx_idx)
        pred = self.pred(ctx, ctx_idx, tgt_idx)
        return F.mse_loss(pred, target, reduction="none").mean((1, 2))

    # ----------------------------------------------------------- training
    def training_step(self, batch, _):
        if self._gen is None:
            self._gen = torch.Generator().manual_seed(
                int(self.global_step) + 12345)
        loss = self._step_loss(batch["x"], self._gen).mean()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, *a):
        frac = min(1.0, self.global_step / max(1, self.trainer.max_steps))
        m = self.hparams.ema_start + (1.0 - self.hparams.ema_start) * frac
        with torch.no_grad():
            for pt, pc in zip(self.target_enc.parameters(),
                              self.enc.parameters()):
                pt.mul_(m).add_(pc.detach(), alpha=1 - m)
        self.log("train/ema_momentum", m)

    # --------------------------------------------------------- validation
    def on_validation_epoch_start(self):
        self._val = {"z": [], "field": [], "ninp": [], "psf": [], "loss": []}
        self._val_gen = torch.Generator().manual_seed(0)   # fixed val masks

    def validation_step(self, batch, batch_idx):
        per_obj = self._step_loss(batch["x"], self._val_gen)
        v = self._val
        v["loss"].append(torch.stack([per_obj,
                                      batch["field"].float()], 1).cpu())
        if sum(t.shape[0] for t in v["z"]) < self.hparams.umap_points:
            v["z"].append(self.encode(batch["x"]).float().cpu())
            v["field"].append(batch["field"].cpu())
            v["ninp"].append(batch["ninp"].cpu())
            v["psf"].append(batch["psf"].cpu())

    def on_validation_epoch_end(self):
        v = self._val
        if not v["loss"]:
            return
        m = torch.cat(v["loss"])
        self.log("val/jepa_loss", m[:, 0].mean())
        for fi, fname in enumerate(FIELDS):
            sel = m[:, 1] == fi
            if sel.any():
                key = fname.lower().replace("-", "")
                self.log(f"val/jepa_loss_{key}", m[sel, 0].mean())
        if self.trainer.sanity_checking:
            return
        try:
            self._log_umap(v)
        except Exception as e:  # figures must never kill training
            print(f"[val figures] skipped: {e}")

    def _log_umap(self, v):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
        import wandb

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
        fig.suptitle(f"JEPA latent UMAP (step {self.global_step})")
        fig.tight_layout()
        if self.logger is not None:
            self.logger.experiment.log(
                {"figures/latent_umap": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        def sched(step):
            w = self.hparams.warmup_steps
            if step < w:
                return step / max(1, w)
            frac = (step - w) / max(1, self.trainer.max_steps - w)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
        return {"optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, sched),
                    "interval": "step"}}
