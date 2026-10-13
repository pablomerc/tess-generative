"""
jepa-hsc-vit: true I-JEPA with latent multi-block masking (see PLAN.md).

A compact ViT for 48x48 stamps (patch 4 -> 12x12 = 144 tokens, dim 192, depth 6)
plus a narrow ViT predictor (dim 96, depth 4) that fills mask tokens at the
target positions. Targets are the EMA encoder's token outputs (post final LN)
at those positions; loss is plain MSE averaged over target tokens (I-JEPA).

Simplification vs the official I-JEPA collator: one mask configuration is
sampled PER BATCH (shared across samples), resampled every step. This keeps all
tensors rectangular; at 144 tokens the loss of per-sample mask diversity is
minor and amortizes over 100k steps.

Eval embedding = mean-pooled EMA target tokens (B, 192), exposed through the
same `encoder_galaxy`/`encoder_instrument` duck-typed properties as the ResNet
arms (width differs from their 64-d — flagged in PLAN.md).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

_galaxy_model = Path(__file__).resolve().parents[1]
_project_root = _galaxy_model.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.jepa_ablation.jepa_base import JEPABase  # noqa: E402


class PatchViT(nn.Module):
    """Minimal ViT encoder over patch tokens; supports encoding a token subset."""

    def __init__(
        self,
        img_size: int = 48,
        patch_size: int = 4,
        in_chans: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, keep_idx: torch.Tensor | None = None) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, D); keep_idx (n_keep,) selects a shared token subset.
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        if keep_idx is not None:
            t = t[:, keep_idx]
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)


class ViTPredictor(nn.Module):
    """Narrow ViT that predicts target-position tokens from context tokens."""

    def __init__(
        self,
        embed_dim: int = 192,
        pred_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        num_patches: int = 144,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        self.blocks = nn.ModuleList(
            [Block(pred_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.proj = nn.Linear(pred_dim, embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, ctx_tokens, ctx_idx, tgt_idx):
        B, n_tgt = ctx_tokens.size(0), tgt_idx.numel()
        x_ctx = self.embed(ctx_tokens) + self.pos_embed[:, ctx_idx]
        x_tgt = self.mask_token.expand(B, n_tgt, -1) + self.pos_embed[:, tgt_idx]
        x = torch.cat([x_ctx, x_tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm(x[:, -n_tgt:]))


class _MeanPoolViT(nn.Module):
    """Stateless wrapper so `encoder_galaxy(imgs)` returns a flat (B, D) vector."""

    def __init__(self, vit: PatchViT):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x).mean(dim=1)


class ViTMeanEncoder(nn.Module):
    """PatchViT + mean-pool as a drop-in ResNetEncoder replacement: forward gives
    a flat (B, D) vector and `out_dim` is exposed so ProjectionHead sizing works.
    Used by the cf-vit arm (JEPACounterfactualModule with encoder_arch='vit')."""

    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 48,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
    ):
        super().__init__()
        self.vit = PatchViT(
            img_size=image_size, patch_size=patch_size, in_chans=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        )
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x).mean(dim=1)


class IJEPAViTModule(JEPABase):
    """I-JEPA on single-survey (HSC) images. Batch: (B, C, H, W) clean images."""

    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 48,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        pred_dim: int = 96,
        pred_depth: int = 4,
        pred_num_heads: int = 3,
        n_target_blocks: int = 4,
        target_scale_min: float = 0.15,
        target_scale_max: float = 0.2,
        target_aspect_min: float = 0.75,
        target_aspect_max: float = 1.5,
        context_scale_min: float = 0.85,
        context_scale_max: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 2000,
        ema_base_momentum: float = 0.996,
        enable_umap_logging: bool = True,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.context_encoder = PatchViT(
            img_size=image_size, patch_size=patch_size, in_chans=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        )
        self.predictor = ViTPredictor(
            embed_dim=embed_dim, pred_dim=pred_dim, depth=pred_depth,
            num_heads=pred_num_heads, num_patches=self.context_encoder.num_patches,
        )
        self.target_encoder = self._clone_frozen(self.context_encoder)

    @property
    def encoder_galaxy(self):
        return _MeanPoolViT(self.target_encoder)

    @property
    def encoder_instrument(self):
        return _MeanPoolViT(self.target_encoder)

    def _ema_pairs(self):
        return [(self.context_encoder, self.target_encoder)]

    # ------------------------------------------------------------- masking --
    def _rand_block(self, scale_lo, scale_hi, aspect_lo, aspect_hi, generator):
        g = self.context_encoder.grid
        scale = scale_lo + (scale_hi - scale_lo) * float(torch.rand(1, generator=generator))
        aspect = aspect_lo + (aspect_hi - aspect_lo) * float(torch.rand(1, generator=generator))
        s = scale * g * g
        h = max(1, min(g, round(math.sqrt(s / aspect))))
        w = max(1, min(g, round(math.sqrt(s * aspect))))
        top = int(torch.randint(0, g - h + 1, (1,), generator=generator))
        left = int(torch.randint(0, g - w + 1, (1,), generator=generator))
        block = torch.zeros(g, g, dtype=torch.bool)
        block[top : top + h, left : left + w] = True
        return block.flatten()

    def _sample_masks(self, generator=None):
        """One shared (ctx_idx, tgt_idx) pair per batch (I-JEPA multi-block)."""
        hp = self.hparams
        tgt = torch.zeros(self.context_encoder.num_patches, dtype=torch.bool)
        for _ in range(hp.n_target_blocks):
            tgt |= self._rand_block(
                hp.target_scale_min, hp.target_scale_max,
                hp.target_aspect_min, hp.target_aspect_max, generator,
            )
        ctx = self._rand_block(
            hp.context_scale_min, hp.context_scale_max, 1.0, 1.0, generator,
        ) & ~tgt
        if not ctx.any():  # degenerate draw: fall back to everything-but-targets
            ctx = ~tgt
        return ctx.nonzero(as_tuple=False).squeeze(1), tgt.nonzero(as_tuple=False).squeeze(1)

    # ---------------------------------------------------------------- steps --
    def _step(self, batch, stage: str, batch_idx: int):
        imgs = batch
        gen = None
        if stage == "val":
            gen = torch.Generator().manual_seed(20_000 + batch_idx)
        ctx_idx, tgt_idx = self._sample_masks(gen)
        ctx_idx, tgt_idx = ctx_idx.to(imgs.device), tgt_idx.to(imgs.device)

        z_ctx = self.context_encoder(imgs, keep_idx=ctx_idx)
        pred = self.predictor(z_ctx, ctx_idx, tgt_idx)          # (B, n_tgt, D)
        with torch.no_grad():
            t_full = self.target_encoder(imgs)                  # (B, N, D)
            t_tgt = t_full[:, tgt_idx]

        loss = F.mse_loss(pred, t_tgt)
        align = F.cosine_similarity(
            pred.detach().flatten(0, 1), t_tgt.flatten(0, 1), dim=-1
        ).mean()

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align", align, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_online", self._embedding_std(z_ctx.mean(dim=1)), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target", self._embedding_std(t_full.mean(dim=1)), on_step=on_step, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", batch_idx)
