"""
Flow-matching decoder conditioned on a SINGLE global embedding vector.

Reconstructs a (4, 48, 48) galaxy image from a fixed 128-d embedding = concat(galaxy, instrument)
produced by the frozen contrastive encoder. Rectified-flow (optimal-transport path) objective,
Euler ODE sampler, and W&B logging are mirrored verbatim from
``galaxy_images/galaxy_model/double_train_fm_neighbors.py``; only the conditioning pathway and
the validation visualization differ.

Conditioning injection (``injection``), decided by a design workflow:
  - "adagn"  (PRIMARY): class-embedding projection as TRUE AdaGN
      (class_embed_type="projection" + resnet_time_scale_shift="scale_shift"), no cross-attn.
      The right conduit for a single GLOBAL vector (DiT/ADM/StyleGAN precedent).
  - "hybrid": AdaGN + a few cross-attention tokens (max-bandwidth reference for this probe).
  - "xattn" : cross-attention tokens only.
  - "concat": broadcast the vector over HxW and concat to the input channels.

All four share the SAME loss / sampler / metric code — only ``_unet_kwargs_for`` and
``_build_cond`` change, so ablations touch nothing else.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def is_h100_gpu() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        for i in range(torch.cuda.device_count()):
            if "h100" in torch.cuda.get_device_name(i).lower():
                return True
    except Exception:
        return False
    return False


def _unet_kwargs_for(
    injection: str,
    image_size: int,
    in_channels: int,
    cond_dim: int,
    cross_attention_dim: int,
    model_channels: int,
    channel_mult: tuple,
    layers_per_block: int,
    attention_head_dim: int,
) -> dict:
    """Return UNet2DConditionModel kwargs for the chosen injection design.

    Verified valid against diffusers 0.38.0 (see fm_cond_sanity.py). Key gotchas:
      - AdaGN uses class_embed_type="projection" + resnet_time_scale_shift="scale_shift"
        (the multiplicative FiLM form; the plain "default" is additive and weaker).
      - With non-attention down/up blocks and encoder_hidden_states=None, the mid block MUST be
        overridden to "UNetMidBlock2D" (default is a CrossAttn mid that crashes on ehs=None).
    """
    block_out_channels = tuple(model_channels * m for m in channel_mult)
    n = len(channel_mult)
    common = dict(
        sample_size=image_size,
        out_channels=in_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
    )
    if injection == "adagn":
        return dict(
            **common,
            in_channels=in_channels,
            down_block_types=("DownBlock2D",) * n,
            up_block_types=("UpBlock2D",) * n,
            mid_block_type="UNetMidBlock2D",
            class_embed_type="projection",
            projection_class_embeddings_input_dim=cond_dim,
            resnet_time_scale_shift="scale_shift",
        )
    if injection == "concat":
        return dict(
            **common,
            in_channels=in_channels + cond_dim,
            down_block_types=("DownBlock2D",) * n,
            up_block_types=("UpBlock2D",) * n,
            mid_block_type="UNetMidBlock2D",
        )
    if injection == "xattn":
        return dict(
            **common,
            in_channels=in_channels,
            down_block_types=("CrossAttnDownBlock2D",) * n,
            up_block_types=("CrossAttnUpBlock2D",) * n,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )
    if injection == "hybrid":
        return dict(
            **common,
            in_channels=in_channels,
            down_block_types=("CrossAttnDownBlock2D",) * n,
            up_block_types=("CrossAttnUpBlock2D",) * n,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            class_embed_type="projection",
            projection_class_embeddings_input_dim=cond_dim,
            resnet_time_scale_shift="scale_shift",
        )
    raise ValueError(f"unknown injection {injection!r}; use adagn|hybrid|xattn|concat")


class EmbeddingConditionedFlowMatching(pl.LightningModule):
    def __init__(
        self,
        cond_dim: int = 128,
        in_channels: int = 4,
        image_size: int = 48,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        cross_attention_dim: int = 256,
        injection: str = "adagn",
        n_cond_tokens: int = 2,
        lr: float = 1e-4,
        num_integration_steps: int = 250,
        num_sample_images: int = 8,
        num_mse_images: int = 64,
        num_samples_per_cond: int = 5,
        num_umap_points: int = 2000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.injection = injection
        self.cond_dim = cond_dim
        self.in_channels = in_channels
        self.image_size = image_size
        self.cross_attention_dim = cross_attention_dim
        self.n_cond_tokens = n_cond_tokens
        self.lr = lr
        self.num_integration_steps = num_integration_steps
        self.num_sample_images = num_sample_images
        self.num_mse_images = num_mse_images
        self.num_samples_per_cond = num_samples_per_cond
        self.num_umap_points = num_umap_points
        self.is_h100 = is_h100_gpu()

        self.velocity_model = UNet2DConditionModel(**_unet_kwargs_for(
            injection, image_size, in_channels, cond_dim, cross_attention_dim,
            model_channels, channel_mult, layers_per_block, attention_head_dim,
        ))

        # Token projection only for the cross-attention pathways.
        if injection in ("xattn", "hybrid"):
            self.token_proj = nn.Linear(cond_dim, n_cond_tokens * cross_attention_dim)
        else:
            self.token_proj = None

    # ------------------------------------------------------------------ conditioning
    def _build_cond(self, cond_vec: torch.Tensor):
        """Map (B, cond_dim) -> (extra_input_or_None, unet_forward_kwargs)."""
        B = cond_vec.shape[0]
        kwargs = {}
        extra = None
        if self.injection == "adagn":
            kwargs["encoder_hidden_states"] = None
            kwargs["class_labels"] = cond_vec
        elif self.injection == "hybrid":
            tokens = self.token_proj(cond_vec).view(B, self.n_cond_tokens, self.cross_attention_dim)
            kwargs["encoder_hidden_states"] = tokens
            kwargs["class_labels"] = cond_vec
        elif self.injection == "xattn":
            tokens = self.token_proj(cond_vec).view(B, self.n_cond_tokens, self.cross_attention_dim)
            kwargs["encoder_hidden_states"] = tokens
        elif self.injection == "concat":
            kwargs["encoder_hidden_states"] = None
            extra = cond_vec
        return extra, kwargs

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        timesteps = t * 1000
        extra, cond_kwargs = self._build_cond(cond_vec)
        if extra is not None:  # concat mode: broadcast vector to spatial channels
            B, _, H, W = x_t.shape
            broadcast = extra[:, :, None, None].expand(B, self.cond_dim, H, W)
            x_in = torch.cat([x_t, broadcast], dim=1)
        else:
            x_in = x_t
        return self.velocity_model(x_in, timesteps, **cond_kwargs).sample

    # ------------------------------------------------------------------ loss
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        cond_vec, x_1, metadata = batch
        B = x_1.shape[0]

        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=x_1.device)
        t_exp = t[:, None, None, None]
        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        target_velocity = x_1 - x_0

        predicted_velocity = self(x_t, t, cond_vec)
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity, reduction="none")
        per_example = loss.mean(dim=(1, 2, 3))

        surveys = [m["anchor_survey"] for m in metadata]
        is_hsc = torch.tensor([s == "hsc" for s in surveys], device=per_example.device)
        is_legacy = torch.tensor([s == "legacy" for s in surveys], device=per_example.device)
        nan = torch.tensor(float("nan"), device=per_example.device)
        self._loss_hsc = (per_example[is_hsc].mean() if is_hsc.any() else nan).detach()
        self._loss_legacy = (per_example[is_legacy].mean() if is_legacy.any() else nan).detach()
        return per_example.mean()

    # ------------------------------------------------------------------ train / val
    def on_train_start(self):
        self._train_start_time = time.time()
        print(f"\n{'='*60}\nTraining started — target {self.trainer.max_steps} steps | "
              f"injection={self.injection} | H100={self.is_h100}\n{'='*60}\n", flush=True)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_hsc", self._loss_hsc, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_legacy", self._loss_legacy, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self._umap_conds = []
        self._umap_surveys = []

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_hsc", self._loss_hsc, on_epoch=True, sync_dist=True)
        self.log("val/loss_legacy", self._loss_legacy, on_epoch=True, sync_dist=True)

        cond_vec, target, metadata = batch
        if batch_idx == 0:
            self._val_cond = cond_vec[: self.num_sample_images].clone()
            self._val_target = target[: self.num_sample_images].clone()
            b = target.shape[0]
            n_mse = min(self.num_mse_images, b)
            self._val_mse_cond = cond_vec[:n_mse].clone()
            self._val_mse_target = target[:n_mse].clone()
            self._val_mse_meta = metadata[:n_mse]

        # collect conditioning vectors for the embedding UMAP (bounded)
        collected = sum(c.shape[0] for c in self._umap_conds)
        if collected < self.num_umap_points:
            self._umap_conds.append(cond_vec.detach().cpu())
            self._umap_surveys.extend([m["anchor_survey"] for m in metadata])
        return loss

    # ------------------------------------------------------------------ sampling / mse
    @torch.no_grad()
    def sample(self, cond_vec: torch.Tensor, num_steps: Optional[int] = None,
               x_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_steps = num_steps or self.num_integration_steps
        B = cond_vec.shape[0]
        device = cond_vec.device
        if x_noise is None:
            x = torch.randn(B, self.in_channels, self.image_size, self.image_size, device=device)
        else:
            x = x_noise.to(device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            x = x + self(x, t, cond_vec) * dt
        return x

    @torch.no_grad()
    def compute_mse(self, target_image, cond_vec, metadata=None, mask_sizes=(48, 32)):
        samples = self.sample(cond_vec)
        diff = target_image - samples
        _, _, height, width = diff.shape
        device = diff.device

        mse_by_size = {}
        for ms in mask_sizes:
            sx = (width - ms) // 2
            sy = (height - ms) // 2
            mse_by_size[ms] = torch.mean(diff[:, :, sy:sy + ms, sx:sx + ms] ** 2)

        primary = mask_sizes[0]
        sx = (width - primary) // 2
        sy = (height - primary) // 2
        diff_primary = diff[:, :, sy:sy + primary, sx:sx + primary]

        mse_hsc = mse_legacy = None
        if metadata is not None:
            surveys = [m["anchor_survey"] for m in metadata]
            hsc_mask = torch.tensor([s == "hsc" for s in surveys], device=device)
            legacy_mask = torch.tensor([s == "legacy" for s in surveys], device=device)
            nan = torch.tensor(float("nan"), device=device)
            mse_hsc = torch.mean(diff_primary[hsc_mask] ** 2) if hsc_mask.any() else nan
            mse_legacy = torch.mean(diff_primary[legacy_mask] ** 2) if legacy_mask.any() else nan
        return mse_by_size, mse_hsc, mse_legacy

    def _normalize_for_vis(self, img: torch.Tensor) -> torch.Tensor:
        img = img.clone()
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return img

    # ------------------------------------------------------------------ validation viz + metrics
    def on_validation_epoch_end(self) -> None:
        if not self.logger or not hasattr(self, "_val_cond"):
            return

        import matplotlib.pyplot as plt
        import wandb

        n_rows = min(6, len(self._val_cond))
        n_samp = self.num_samples_per_cond
        n_cols = 1 + n_samp + 1  # Target | Sample1..N | Mean
        col_titles = ["Target"] + [f"Sample {j+1}" for j in range(n_samp)] + ["Mean"]

        def _row_scale_rgb(x_chw, vmin, vmax):
            x = x_chw[:3]
            vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
            vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
            return ((x - vmin_t) / (vmax_t - vmin_t + 1e-8)).clamp(0, 1).permute(1, 2, 0)

        fig_orig, axes_orig = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), squeeze=False)
        fig_row, axes_row = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), squeeze=False)
        for j, title in enumerate(col_titles):
            axes_orig[0, j].set_title(title, fontsize=10)
            axes_row[0, j].set_title(title, fontsize=10)

        for i in range(n_rows):
            cond = self._val_cond[i:i + 1].to(self.device)
            target = self._val_target[i:i + 1].to(self.device)
            cond_rep = cond.repeat(n_samp, 1)
            samples = self.sample(cond_rep)
            mean_sample = samples.mean(dim=0, keepdim=True)

            # (A) per-image min-max normalized
            axes_orig[i, 0].imshow(self._normalize_for_vis(target[0, :3]).cpu().permute(1, 2, 0).numpy())
            axes_orig[i, 0].axis("off")
            for j in range(n_samp):
                axes_orig[i, 1 + j].imshow(self._normalize_for_vis(samples[j, :3]).cpu().permute(1, 2, 0).numpy())
                axes_orig[i, 1 + j].axis("off")
            axes_orig[i, -1].imshow(self._normalize_for_vis(mean_sample[0, :3]).cpu().permute(1, 2, 0).numpy())
            axes_orig[i, -1].axis("off")

            # (B) row-scaled to the target's per-channel range (compare fluxes fairly)
            tchw = target[0, :3]
            vmin, vmax = tchw.amin(dim=(1, 2)), tchw.amax(dim=(1, 2))
            axes_row[i, 0].imshow(_row_scale_rgb(target[0, :3], vmin, vmax).detach().cpu().numpy())
            axes_row[i, 0].axis("off")
            for j in range(n_samp):
                axes_row[i, 1 + j].imshow(_row_scale_rgb(samples[j, :3], vmin, vmax).detach().cpu().numpy())
                axes_row[i, 1 + j].axis("off")
            axes_row[i, -1].imshow(_row_scale_rgb(mean_sample[0, :3], vmin, vmax).detach().cpu().numpy())
            axes_row[i, -1].axis("off")

        plt.figure(fig_orig.number); plt.tight_layout()
        plt.figure(fig_row.number); plt.tight_layout()
        self.logger.experiment.log({
            "val/recon_grid": wandb.Image(fig_orig),
            "val/recon_grid_row_scaled": wandb.Image(fig_row),
            "global_step": self.global_step,
        })
        plt.close(fig_orig)
        plt.close(fig_row)

        # Reconstruction MSE (48 + 32 center crop, split by survey)
        if hasattr(self, "_val_mse_target"):
            mse_by_size, mse_hsc, mse_legacy = self.compute_mse(
                self._val_mse_target.to(self.device),
                self._val_mse_cond.to(self.device),
                self._val_mse_meta,
                mask_sizes=(48, 32),
            )
            self.log("val/mse", mse_by_size[48], sync_dist=True)
            self.log("val/mse_32", mse_by_size[32], sync_dist=True)
            if mse_hsc is not None:
                self.log("val/mse_hsc", mse_hsc, sync_dist=True)
            if mse_legacy is not None:
                self.log("val/mse_legacy", mse_legacy, sync_dist=True)

        # Embedding UMAP colored by survey (analog of the FM model's latent UMAP)
        self._log_embedding_umap()

    @torch.no_grad()
    def _log_embedding_umap(self):
        if not self._umap_conds:
            return
        try:
            import matplotlib.pyplot as plt
            import umap
            import wandb
        except Exception as e:
            print(f"[UMAP] skipping ({e})")
            return
        try:
            conds = torch.cat(self._umap_conds, dim=0)[: self.num_umap_points].numpy()
            surveys = np.array(self._umap_surveys[: conds.shape[0]])
            emb2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                              metric="euclidean", random_state=42, n_jobs=1).fit_transform(conds)
            fig, ax = plt.subplots(figsize=(6, 5))
            for name in ("hsc", "legacy"):
                m = surveys == name
                if m.any():
                    ax.scatter(emb2d[m, 0], emb2d[m, 1], s=6, alpha=0.6, label=name)
            ax.set_title("Conditioning embedding UMAP")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            self.logger.experiment.log({"val/embedding_umap": wandb.Image(fig),
                                        "global_step": self.global_step})
            plt.close(fig)
        except Exception as e:
            print(f"[UMAP] error: {e}")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        # Step-based cosine so it is robust under max_steps training (and resumes).
        t_max = max(1, int(self.trainer.max_steps)) if self.trainer.max_steps and self.trainer.max_steps > 0 else 100_000
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
