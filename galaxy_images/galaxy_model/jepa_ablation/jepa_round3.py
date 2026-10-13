"""
Round-3 arms (see PLAN.md §Round 3 and REPORT.md §4b for the motivating results):

  * CFTokenJEPAModule (`--arm cf-token`) — the counterfactual dual-tower JEPA with
    TOKEN-level targets. Identical structure to cf-vit except pooled->token: the
    controlled test of "pooling killed cf-vit". No masking (the views are the
    corruption — the token targets protect spatial info instead).
      physics tower:    TokenTranslator(partner tokens) -> EMA physics tokens of anchor
      instrument tower: CrossAttnPredictor(neighbour token set) -> EMA instrument
                        tokens of anchor (no positional correspondence assumed)

  * XJEPAModule (`--arm xjepa`) — shared ViT trunk with physics/instrument token
    SUBSPACES (per-token linear heads), three losses:
      L_mask (within-image I-JEPA on trunk tokens, low-weight insurance)
      L_phys (cross-survey token translation into the physics subspace)
      L_inst (neighbour-set cross-attention into the instrument subspace)
    Eval: encoder_galaxy = mean-pooled physics subspace (128-d),
          encoder_instrument = mean-pooled instrument subspace (64-d).

  * `--arm multi-vit` needs NO new module — it is IJEPAViTModule trained on the
    both-surveys stamp pool (AllStamps48Dataset in datasets.py).

Design rule (eval-integration lesson from rounds 1-2): survey/direction
conditioning lives only in predictors, never in encoders — the target survey is
always the complement of the input's, so it is inferable and the duck-typed
eval stack (`encoder_*(imgs)`) keeps working unchanged.

Class-sniff markers (encode_latents.py / regenerate_umap_jepa.py):
  XJEPAModule -> `phys_dim`; CFTokenJEPAModule -> `translator_depth`.
Both also carry lambda_galaxy, so these markers MUST be checked before the
JEPACounterfactualModule branch.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_galaxy_model = Path(__file__).resolve().parents[1]
_project_root = _galaxy_model.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.jepa_ablation.jepa_base import JEPABase  # noqa: E402
from galaxy_images.galaxy_model.jepa_ablation.jepa_vit import (  # noqa: E402
    PatchViT,
    ViTPredictor,
    _MeanPoolViT,
)
from timm.models.vision_transformer import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

class TokenTranslator(nn.Module):
    """Position-aligned token->token predictor (partner tokens -> anchor tokens).

    Well-posed because the ram48 Legacy stamps were zoomed to HSC's pixel scale
    at build time, so token (i,j) covers the same sky patch in both surveys.
    Direction is inferable (target survey = complement of the input's), so no
    explicit conditioning is needed.
    """

    def __init__(self, in_dim: int, out_dim: int, num_patches: int,
                 pred_dim: int = 96, depth: int = 4, num_heads: int = 3,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, pred_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        self.blocks = nn.ModuleList(
            [Block(pred_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.proj = nn.Linear(pred_dim, out_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm(x))


class CrossAttnPredictor(nn.Module):
    """Set->tokens predictor for the instrument pathway.

    Neighbours are NOT pixel-aligned with the anchor (different galaxies), so
    positional correspondence is never assumed: queries are mask tokens carrying
    the anchor's positions; keys/values are the full neighbour token set. The
    predictor learns to extract observing-condition statistics (PSF, depth,
    background) and render them at the anchor's positions.
    """

    def __init__(self, in_dim: int, out_dim: int, num_patches: int,
                 pred_dim: int = 96, depth: int = 2, num_heads: int = 3,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.num_patches = num_patches
        self.kv_embed = nn.Linear(in_dim, pred_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(pred_dim, num_heads, batch_first=True),
                nn.LayerNorm(pred_dim),
                nn.Sequential(
                    nn.Linear(pred_dim, int(pred_dim * mlp_ratio)), nn.GELU(),
                    nn.Linear(int(pred_dim * mlp_ratio), pred_dim),
                ),
                nn.LayerNorm(pred_dim),
            ]))
        self.norm = nn.LayerNorm(pred_dim)
        self.proj = nn.Linear(pred_dim, out_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, neigh_tokens: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
        # neigh_tokens: (B, K*N, in_dim); valid_token_mask: (B, K*N) True = real.
        B, M, _ = neigh_tokens.shape
        k = M // self.num_patches
        kv = self.kv_embed(neigh_tokens) + self.pos_embed.repeat(1, k, 1)
        q = self.mask_token.expand(B, self.num_patches, -1) + self.pos_embed
        pad = ~valid_token_mask  # MultiheadAttention: True = EXCLUDE
        for attn, ln1, mlp, ln2 in self.layers:
            a, _ = attn(q, kv, kv, key_padding_mask=pad, need_weights=False)
            q = ln1(q + a)
            q = ln2(q + mlp(q))
        return self.proj(self.norm(q))


class _MeanPoolHead(nn.Module):
    """trunk -> per-token head -> mean pool, as a flat (B, D) eval encoder."""

    def __init__(self, trunk: nn.Module, head: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).mean(dim=1)


def _neighbour_token_batch(encoder, head, sameins, masks):
    """Encode (B,K,C,H,W) neighbours -> ((B, K*N, D) tokens, (B, K*N) validity)."""
    B, K, C, H, W = sameins.shape
    tok = encoder(sameins.view(B * K, C, H, W))
    if head is not None:
        tok = head(tok)
    N, D = tok.shape[1], tok.shape[2]
    tok = tok.view(B, K * N, D)
    valid = masks.bool().repeat_interleave(N, dim=1)  # (B, K*N)
    return tok, valid


# ---------------------------------------------------------------------------
# cf-token
# ---------------------------------------------------------------------------

class CFTokenJEPAModule(JEPABase):
    """Counterfactual dual-tower JEPA with token-level targets (no masking).

    Batch contract identical to cf/cf-vit: (targets, samegals, sameins, masks, meta).
    """

    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 48,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        translator_depth: int = 4,      # sniff marker — keep unique to this class
        xattn_depth: int = 2,
        predictor_dim: int = 96,
        lambda_galaxy: float = 1.0,
        lambda_instrument: float = 1.0,
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

        def vit():
            return PatchViT(img_size=image_size, patch_size=patch_size,
                            in_chans=in_channels, embed_dim=embed_dim,
                            depth=depth, num_heads=num_heads)

        self.context_encoder_galaxy = vit()
        self.context_encoder_instrument = vit()
        n = self.context_encoder_galaxy.num_patches
        self.predictor_galaxy = TokenTranslator(
            embed_dim, embed_dim, n, pred_dim=predictor_dim, depth=translator_depth,
            num_heads=num_heads,
        )
        self.predictor_instrument = CrossAttnPredictor(
            embed_dim, embed_dim, n, pred_dim=predictor_dim, depth=xattn_depth,
            num_heads=num_heads,
        )
        self.target_encoder_galaxy = self._clone_frozen(self.context_encoder_galaxy)
        self.target_encoder_instrument = self._clone_frozen(self.context_encoder_instrument)

    @property
    def encoder_galaxy(self):
        return _MeanPoolViT(self.target_encoder_galaxy)

    @property
    def encoder_instrument(self):
        return _MeanPoolViT(self.target_encoder_instrument)

    def _ema_pairs(self):
        return [
            (self.context_encoder_galaxy, self.target_encoder_galaxy),
            (self.context_encoder_instrument, self.target_encoder_instrument),
        ]

    def _umap_panels(self):
        return [
            ("Physics Encoder (EMA) UMAP", self.encoder_galaxy),
            ("Instrument Encoder (EMA) UMAP", self.encoder_instrument),
        ]

    def _step(self, batch, stage: str):
        targets, samegals, sameins, masks, _meta = batch

        # ---- physics: partner tokens -> anchor tokens (position-aligned) ----
        pred_g = self.predictor_galaxy(self.context_encoder_galaxy(samegals))
        with torch.no_grad():
            t_g = self.target_encoder_galaxy(targets)
        loss_galaxy = F.mse_loss(pred_g, t_g)
        align_galaxy = F.cosine_similarity(
            pred_g.detach().flatten(0, 1), t_g.flatten(0, 1), dim=-1).mean()

        # ---- instrument: neighbour token set -> anchor tokens ---------------
        valid_rows = masks.bool().any(dim=1)
        if valid_rows.any():
            tok, valid = _neighbour_token_batch(
                self.context_encoder_instrument, None,
                sameins[valid_rows], masks[valid_rows])
            pred_i = self.predictor_instrument(tok, valid)
            with torch.no_grad():
                t_i = self.target_encoder_instrument(targets[valid_rows])
            loss_instrument = F.mse_loss(pred_i, t_i)
            align_instrument = F.cosine_similarity(
                pred_i.detach().flatten(0, 1), t_i.flatten(0, 1), dim=-1).mean()
        else:
            loss_instrument = targets.new_tensor(0.0)
            align_instrument = targets.new_tensor(0.0)
            t_i = t_g

        loss = (self.hparams.lambda_galaxy * loss_galaxy
                + self.hparams.lambda_instrument * loss_instrument)

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_galaxy", loss_galaxy.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_instrument", loss_instrument.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align_galaxy", align_galaxy, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align_instrument", align_instrument, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target", self._embedding_std(t_g.mean(dim=1)), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target_ins", self._embedding_std(t_i.mean(dim=1)), on_step=on_step, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")


# ---------------------------------------------------------------------------
# X-JEPA
# ---------------------------------------------------------------------------

class XJEPAModule(JEPABase):
    """Shared trunk + physics/instrument token subspaces (A2 + B1 in PLAN.md).

    Batch contract identical to cf: (targets, samegals, sameins, masks, meta).
    """

    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 48,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        phys_dim: int = 128,            # sniff marker — keep unique to this class
        inst_dim: int = 64,
        predictor_dim: int = 96,
        pred_depth_mask: int = 4,
        pred_depth_translator: int = 4,
        pred_depth_xattn: int = 2,
        n_target_blocks: int = 4,
        target_scale_min: float = 0.15,
        target_scale_max: float = 0.2,
        target_aspect_min: float = 0.75,
        target_aspect_max: float = 1.5,
        context_scale_min: float = 0.85,
        context_scale_max: float = 1.0,
        lambda_mask: float = 0.5,
        lambda_galaxy: float = 1.0,
        lambda_instrument: float = 1.0,
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

        self.trunk = PatchViT(img_size=image_size, patch_size=patch_size,
                              in_chans=in_channels, embed_dim=embed_dim,
                              depth=depth, num_heads=num_heads)
        n = self.trunk.num_patches
        # Per-token subspace heads; LayerNorm so EMA targets are normalized.
        self.phys_head = nn.Sequential(nn.Linear(embed_dim, phys_dim), nn.LayerNorm(phys_dim))
        self.inst_head = nn.Sequential(nn.Linear(embed_dim, inst_dim), nn.LayerNorm(inst_dim))

        self.predictor_mask = ViTPredictor(
            embed_dim=embed_dim, pred_dim=predictor_dim, depth=pred_depth_mask,
            num_heads=num_heads, num_patches=n)
        self.predictor_galaxy = TokenTranslator(
            phys_dim, phys_dim, n, pred_dim=predictor_dim,
            depth=pred_depth_translator, num_heads=num_heads)
        self.predictor_instrument = CrossAttnPredictor(
            inst_dim, inst_dim, n, pred_dim=predictor_dim,
            depth=pred_depth_xattn, num_heads=num_heads)

        self.target_trunk = self._clone_frozen(self.trunk)
        self.target_phys_head = self._clone_frozen(self.phys_head)
        self.target_inst_head = self._clone_frozen(self.inst_head)

    @property
    def encoder_galaxy(self):
        return _MeanPoolHead(self.target_trunk, self.target_phys_head)

    @property
    def encoder_instrument(self):
        return _MeanPoolHead(self.target_trunk, self.target_inst_head)

    def _ema_pairs(self):
        return [
            (self.trunk, self.target_trunk),
            (self.phys_head, self.target_phys_head),
            (self.inst_head, self.target_inst_head),
        ]

    def _umap_panels(self):
        return [
            ("Physics Subspace (EMA) UMAP", self.encoder_galaxy),
            ("Instrument Subspace (EMA) UMAP", self.encoder_instrument),
        ]

    # I-JEPA multi-block masking on the 12x12 trunk grid (same recipe as hsc-vit).
    def _rand_block(self, scale_lo, scale_hi, aspect_lo, aspect_hi, generator):
        g = self.trunk.grid
        scale = scale_lo + (scale_hi - scale_lo) * float(torch.rand(1, generator=generator))
        aspect = aspect_lo + (aspect_hi - aspect_lo) * float(torch.rand(1, generator=generator))
        s = scale * g * g
        h = max(1, min(g, round(math.sqrt(s / aspect))))
        w = max(1, min(g, round(math.sqrt(s * aspect))))
        top = int(torch.randint(0, g - h + 1, (1,), generator=generator))
        left = int(torch.randint(0, g - w + 1, (1,), generator=generator))
        block = torch.zeros(g, g, dtype=torch.bool)
        block[top: top + h, left: left + w] = True
        return block.flatten()

    def _sample_masks(self, generator=None):
        hp = self.hparams
        tgt = torch.zeros(self.trunk.num_patches, dtype=torch.bool)
        for _ in range(hp.n_target_blocks):
            tgt |= self._rand_block(hp.target_scale_min, hp.target_scale_max,
                                    hp.target_aspect_min, hp.target_aspect_max, generator)
        ctx = self._rand_block(hp.context_scale_min, hp.context_scale_max,
                               1.0, 1.0, generator) & ~tgt
        if not ctx.any():
            ctx = ~tgt
        return ctx.nonzero(as_tuple=False).squeeze(1), tgt.nonzero(as_tuple=False).squeeze(1)

    def _step(self, batch, stage: str, batch_idx: int):
        targets, samegals, sameins, masks, _meta = batch
        hp = self.hparams

        with torch.no_grad():
            t_trunk = self.target_trunk(targets)          # (B, N, D)
            t_phys = self.target_phys_head(t_trunk)       # (B, N, phys_dim)
            t_inst = self.target_inst_head(t_trunk)       # (B, N, inst_dim)

        # ---- 1) within-image I-JEPA on trunk tokens (insurance) -------------
        gen = torch.Generator().manual_seed(30_000 + batch_idx) if stage == "val" else None
        ctx_idx, tgt_idx = self._sample_masks(gen)
        ctx_idx, tgt_idx = ctx_idx.to(targets.device), tgt_idx.to(targets.device)
        z_ctx = self.trunk(targets, keep_idx=ctx_idx)
        pred_m = self.predictor_mask(z_ctx, ctx_idx, tgt_idx)
        loss_mask = F.mse_loss(pred_m, t_trunk[:, tgt_idx])

        # ---- 2) cross-survey token translation into the physics subspace ----
        partner_phys = self.phys_head(self.trunk(samegals))
        pred_g = self.predictor_galaxy(partner_phys)
        loss_galaxy = F.mse_loss(pred_g, t_phys)
        align_galaxy = F.cosine_similarity(
            pred_g.detach().flatten(0, 1), t_phys.flatten(0, 1), dim=-1).mean()

        # ---- 3) neighbour set -> instrument subspace tokens ------------------
        valid_rows = masks.bool().any(dim=1)
        if valid_rows.any():
            tok, valid = _neighbour_token_batch(
                self.trunk, self.inst_head, sameins[valid_rows], masks[valid_rows])
            pred_i = self.predictor_instrument(tok, valid)
            loss_instrument = F.mse_loss(pred_i, t_inst[valid_rows])
            align_instrument = F.cosine_similarity(
                pred_i.detach().flatten(0, 1), t_inst[valid_rows].flatten(0, 1), dim=-1).mean()
        else:
            loss_instrument = targets.new_tensor(0.0)
            align_instrument = targets.new_tensor(0.0)

        loss = (hp.lambda_mask * loss_mask
                + hp.lambda_galaxy * loss_galaxy
                + hp.lambda_instrument * loss_instrument)

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_mask", loss_mask.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_galaxy", loss_galaxy.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_instrument", loss_instrument.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align_galaxy", align_galaxy, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align_instrument", align_instrument, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target", self._embedding_std(t_phys.mean(dim=1)), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target_ins", self._embedding_std(t_inst.mean(dim=1)), on_step=on_step, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", batch_idx)
