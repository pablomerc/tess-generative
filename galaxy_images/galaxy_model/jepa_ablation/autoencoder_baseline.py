"""
ResNet autoencoder baseline (`--arm ae-hsc`): same encoder, same data, same budget
as jepa-hsc-resnet, but a plain reconstruction objective instead of JEPA.

Purpose: isolate what the predict-in-embedding-space objective contributes beyond
"any self-supervised encoder trained on the 367k HSC pool". Encoder is the
contrastive baseline's ResNetEncoder (64-d, conv1x1); decoder is a small
ConvTranspose stack 3->6->12->24->48. Loss = pixel MSE on the normalized stamps.

No EMA, no predictor, no masking. Downstream eval uses the encoder output under
the same duck-typed `encoder_galaxy`/`encoder_instrument` names as the JEPA arms
(single space exposed under both, so e1 == e2 in the embeddings H5).

Class-sniff marker for encode_latents.py / regenerate_umap_jepa.py:
`decoder_channels` in hparams.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_galaxy_model = Path(__file__).resolve().parents[1]
_project_root = _galaxy_model.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (  # noqa: E402
    ResNetEncoder,
)
from galaxy_images.galaxy_model.jepa_ablation.jepa_base import JEPABase  # noqa: E402


class AEHSCModule(JEPABase):
    """ResNet18 autoencoder on single-survey (HSC) images. Batch: (B, C, H, W)."""

    def __init__(
        self,
        in_channels: int = 4,
        embedding_dim: int = 64,
        encoder_pool: str = "conv1x1",
        encoder_token_dim: int = None,
        image_size: int = 48,
        decoder_channels: tuple = (256, 128, 64, 32),
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 2000,
        enable_umap_logging: bool = True,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            pretrained=False,
            pool=encoder_pool,
            token_dim=encoder_token_dim,
            image_size=image_size,
        )
        c = list(decoder_channels)
        # 48 = 3 * 2^4: seed a 3x3 map, then 4 stride-2 deconvs.
        layers = [nn.Linear(self.encoder.out_dim, c[0] * 3 * 3), nn.GELU(),
                  nn.Unflatten(1, (c[0], 3, 3))]
        for c_in, c_out in zip(c, c[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), nn.GELU()]
        layers += [nn.ConvTranspose2d(c[-1], in_channels, 4, stride=2, padding=1)]
        self.decoder = nn.Sequential(*layers)

    # Single space exposed under both duck-typed names (no EMA — the trained
    # encoder itself is what gets evaluated).
    @property
    def encoder_galaxy(self):
        return self.encoder

    @property
    def encoder_instrument(self):
        return self.encoder

    # --- JEPABase overrides: an AE has no EMA target ------------------------
    def _ema_pairs(self):
        return []

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def _umap_panels(self):
        return [("AE Encoder UMAP", self.encoder)]

    # ------------------------------------------------------------------ steps
    def _step(self, batch, stage: str):
        imgs = batch
        z = self.encoder(imgs)
        recon = self.decoder(z)
        loss = F.mse_loss(recon, imgs)

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/std_online", self._embedding_std(z), on_step=on_step, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
