"""
The three ResNet-encoder JEPA arms (see PLAN.md):

  * JEPAHSCModule            — jepa-hsc-resnet: image-level JEPA on HSC stamps.
                               Corruption = random block masking of the input;
                               predict the EMA embedding of the CLEAN image.
  * JEPACounterfactualModule — jepa-cf: dual towers, no masking. Physics tower
                               predicts the anchor from its cross-survey partner;
                               instrument tower predicts the anchor from the mean
                               of its K sky-neighbor embeddings.
  * JEPACrossSurveyModule    — jepa-x: single encoder, two direction predictors
                               (hsc->legacy, legacy->hsc), symmetric loss.

Encoders/projection heads are the contrastive baseline's own classes, so every
arm is dim-matched (64-d encoder output) to the published contrastive 2x2 arms.

Duck-typing contract: `encoder_galaxy` / `encoder_instrument` properties return
the EMA target encoders (single-encoder arms expose the same module under both
names). Online context encoders live under `context_encoder*` names.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_galaxy_model = Path(__file__).resolve().parents[1]
_project_root = _galaxy_model.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (  # noqa: E402
    ProjectionHead,
    ResNetEncoder,
)
from galaxy_images.galaxy_model.jepa_ablation.jepa_base import JEPABase  # noqa: E402


def _make_encoder(hp):
    if getattr(hp, "encoder_arch", "resnet") == "vit":
        # cf-vit arm: mean-pooled PatchViT (192-d out_dim), a drop-in
        # ResNetEncoder replacement. Lazy import avoids a circular dependency.
        from galaxy_images.galaxy_model.jepa_ablation.jepa_vit import ViTMeanEncoder
        return ViTMeanEncoder(in_channels=hp.in_channels, image_size=hp.image_size)
    return ResNetEncoder(
        in_channels=hp.in_channels,
        embedding_dim=hp.embedding_dim,
        pretrained=False,
        pool=hp.encoder_pool,
        token_dim=hp.encoder_token_dim,
        image_size=hp.image_size,
    )


# ---------------------------------------------------------------------------
# jepa-hsc-resnet
# ---------------------------------------------------------------------------

class JEPAHSCModule(JEPABase):
    """Image-level masked JEPA on single-survey (HSC) images.

    Batch: (B, C, H, W) clean images. The block mask is generated on the fly in
    the step (data is normalized, so masked pixels = 0 = the per-survey mean).
    """

    def __init__(
        self,
        in_channels: int = 4,
        embedding_dim: int = 64,
        projection_dim: int = 32,
        projection_hidden_dim: int = 64,
        predictor_hidden_dim: int = 64,
        encoder_pool: str = "conv1x1",
        encoder_token_dim: int = None,
        image_size: int = 48,
        mask_blocks_max: int = 4,
        mask_side_min: int = 12,
        mask_side_max: int = 24,
        mask_frac_min: float = 0.3,
        mask_frac_max: float = 0.6,
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

        self.context_encoder = _make_encoder(self.hparams)
        self.context_head = ProjectionHead(
            self.context_encoder.out_dim, projection_hidden_dim, projection_dim
        )
        self.predictor = ProjectionHead(projection_dim, predictor_hidden_dim, projection_dim)

        self.target_encoder = self._clone_frozen(self.context_encoder)
        self.target_head = self._clone_frozen(self.context_head)

    # EMA target under the duck-typed names (same encoder for both branches).
    @property
    def encoder_galaxy(self):
        return self.target_encoder

    @property
    def encoder_instrument(self):
        return self.target_encoder

    def _ema_pairs(self):
        return [
            (self.context_encoder, self.target_encoder),
            (self.context_head, self.target_head),
        ]

    def _block_mask(self, batch_size: int, height: int, width: int, generator=None):
        """(B, 1, H, W) float mask, 1 = keep, 0 = masked. Built on CPU (cheap)."""
        hp = self.hparams

        def _rand(lo, hi):
            return int(torch.randint(lo, hi + 1, (1,), generator=generator).item())

        mask = torch.ones(batch_size, 1, height, width)
        for b in range(batch_size):
            frac_target = hp.mask_frac_min + (hp.mask_frac_max - hp.mask_frac_min) * float(
                torch.rand(1, generator=generator).item()
            )
            masked = torch.zeros(height, width, dtype=torch.bool)
            for _ in range(2 * hp.mask_blocks_max):
                if masked.float().mean().item() >= frac_target:
                    break
                bh = _rand(hp.mask_side_min, hp.mask_side_max)
                bw = _rand(hp.mask_side_min, hp.mask_side_max)
                top = _rand(0, height - bh)
                left = _rand(0, width - bw)
                masked[top : top + bh, left : left + bw] = True
            mask[b, 0][masked] = 0.0
        return mask

    def _step(self, batch, stage: str, batch_idx: int):
        imgs = batch  # (B, C, H, W)
        # Deterministic masks in validation so val/loss is comparable across epochs.
        gen = None
        if stage == "val":
            gen = torch.Generator().manual_seed(10_000 + batch_idx)
        mask = self._block_mask(imgs.size(0), imgs.size(2), imgs.size(3), gen).to(imgs.device)

        h_ctx = self.context_head(self.context_encoder(imgs * mask))
        pred = self.predictor(h_ctx)
        with torch.no_grad():
            h_tgt = self.target_head(self.target_encoder(imgs))

        loss, align = self._cosine_loss(pred, h_tgt)
        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align", align, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_online", self._embedding_std(h_ctx), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target", self._embedding_std(h_tgt), on_step=on_step, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", batch_idx)


# ---------------------------------------------------------------------------
# jepa-cf
# ---------------------------------------------------------------------------

class JEPACounterfactualModule(JEPABase):
    """Counterfactual/triplet JEPA with physics + instrument towers, no masking.

    Batch contract identical to the contrastive baseline (collate_neighbors):
      (targets, samegals, sameins, masks, metadata)

    Physics tower:    online(samegal)          -> predict EMA_physics(anchor)
    Instrument tower: mean_k online(neighbor_k) -> predict EMA_instrument(anchor)
    Anchor survey alternates by dataset index parity, exactly as in the baseline.
    """

    def __init__(
        self,
        in_channels: int = 4,
        embedding_dim: int = 64,
        projection_dim: int = 32,
        projection_hidden_dim: int = 64,
        predictor_hidden_dim: int = 64,
        encoder_pool: str = "conv1x1",
        encoder_token_dim: int = None,
        encoder_arch: str = "resnet",   # "resnet" (64-d) or "vit" (mean-pooled PatchViT, 192-d)
        image_size: int = 48,
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

        self.context_encoder_galaxy = _make_encoder(self.hparams)
        self.context_head_galaxy = ProjectionHead(
            self.context_encoder_galaxy.out_dim, projection_hidden_dim, projection_dim
        )
        self.predictor_galaxy = ProjectionHead(
            projection_dim, predictor_hidden_dim, projection_dim
        )

        self.context_encoder_instrument = _make_encoder(self.hparams)
        self.context_head_instrument = ProjectionHead(
            self.context_encoder_instrument.out_dim, projection_hidden_dim, projection_dim
        )
        self.predictor_instrument = ProjectionHead(
            projection_dim, predictor_hidden_dim, projection_dim
        )

        self.target_encoder_galaxy = self._clone_frozen(self.context_encoder_galaxy)
        self.target_head_galaxy = self._clone_frozen(self.context_head_galaxy)
        self.target_encoder_instrument = self._clone_frozen(self.context_encoder_instrument)
        self.target_head_instrument = self._clone_frozen(self.context_head_instrument)

    @property
    def encoder_galaxy(self):
        return self.target_encoder_galaxy

    @property
    def encoder_instrument(self):
        return self.target_encoder_instrument

    def _ema_pairs(self):
        return [
            (self.context_encoder_galaxy, self.target_encoder_galaxy),
            (self.context_head_galaxy, self.target_head_galaxy),
            (self.context_encoder_instrument, self.target_encoder_instrument),
            (self.context_head_instrument, self.target_head_instrument),
        ]

    def _umap_panels(self):
        return [
            ("Physics Encoder (EMA) UMAP", self.target_encoder_galaxy),
            ("Instrument Encoder (EMA) UMAP", self.target_encoder_instrument),
        ]

    def _step(self, batch, stage: str):
        targets, samegals, sameins, masks, _metadata = batch

        # ---- physics: cross-survey partner -> anchor -------------------------
        h_g = self.context_head_galaxy(self.context_encoder_galaxy(samegals))
        pred_g = self.predictor_galaxy(h_g)
        with torch.no_grad():
            t_g = self.target_head_galaxy(self.target_encoder_galaxy(targets))
        loss_galaxy, align_galaxy = self._cosine_loss(pred_g, t_g)

        # ---- instrument: mean of sky neighbors -> anchor ---------------------
        B, K, C, H, W = sameins.shape
        z = self.context_head_instrument(
            self.context_encoder_instrument(sameins.view(B * K, C, H, W))
        ).view(B, K, -1)
        m = masks.float().unsqueeze(-1)  # (B, K, 1)
        counts = m.sum(dim=1)            # (B, 1)
        valid = counts.squeeze(1) > 0
        if valid.any():
            mean_z = (z * m).sum(dim=1) / counts.clamp(min=1.0)
            pred_i = self.predictor_instrument(mean_z[valid])
            with torch.no_grad():
                t_i = self.target_head_instrument(self.target_encoder_instrument(targets[valid]))
            loss_instrument, align_instrument = self._cosine_loss(pred_i, t_i)
        else:
            loss_instrument = targets.new_tensor(0.0)
            align_instrument = targets.new_tensor(0.0)
            t_i = t_g  # placeholder for diagnostics only

        loss = (
            self.hparams.lambda_galaxy * loss_galaxy
            + self.hparams.lambda_instrument * loss_instrument
        )

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_galaxy", loss_galaxy.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_instrument", loss_instrument.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align_galaxy", align_galaxy, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align_instrument", align_instrument, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_online", self._embedding_std(h_g), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target", self._embedding_std(t_g), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target_ins", self._embedding_std(t_i), on_step=on_step, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")


# ---------------------------------------------------------------------------
# jepa-x
# ---------------------------------------------------------------------------

class JEPACrossSurveyModule(JEPABase):
    """Cross-survey JEPA: one encoder, two direction predictors, symmetric loss.

    Batch: (hsc, legacy) — the same galaxy in both surveys (AnchorPairs48Dataset).
    """

    def __init__(
        self,
        in_channels: int = 4,
        embedding_dim: int = 64,
        projection_dim: int = 32,
        projection_hidden_dim: int = 64,
        predictor_hidden_dim: int = 64,
        encoder_pool: str = "conv1x1",
        encoder_token_dim: int = None,
        image_size: int = 48,
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

        self.context_encoder = _make_encoder(self.hparams)
        self.context_head = ProjectionHead(
            self.context_encoder.out_dim, projection_hidden_dim, projection_dim
        )
        self.predictor_h2l = ProjectionHead(projection_dim, predictor_hidden_dim, projection_dim)
        self.predictor_l2h = ProjectionHead(projection_dim, predictor_hidden_dim, projection_dim)

        self.target_encoder = self._clone_frozen(self.context_encoder)
        self.target_head = self._clone_frozen(self.context_head)

    @property
    def encoder_galaxy(self):
        return self.target_encoder

    @property
    def encoder_instrument(self):
        return self.target_encoder

    def _ema_pairs(self):
        return [
            (self.context_encoder, self.target_encoder),
            (self.context_head, self.target_head),
        ]

    def _step(self, batch, stage: str):
        hsc, legacy = batch

        h_h = self.context_head(self.context_encoder(hsc))
        h_l = self.context_head(self.context_encoder(legacy))
        with torch.no_grad():
            t_h = self.target_head(self.target_encoder(hsc))
            t_l = self.target_head(self.target_encoder(legacy))

        loss_h2l, align_h2l = self._cosine_loss(self.predictor_h2l(h_h), t_l)
        loss_l2h, align_l2h = self._cosine_loss(self.predictor_l2h(h_l), t_h)
        loss = 0.5 * (loss_h2l + loss_l2h)

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_h2l", loss_h2l.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss_l2h", loss_l2h.detach(), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align_h2l", align_h2l, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/align_l2h", align_l2h, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_online", self._embedding_std(h_h), on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_target", self._embedding_std(t_l), on_step=on_step, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
