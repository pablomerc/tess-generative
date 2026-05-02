"""
Single-encoder cross-survey flow matching — ablation of the dual-encoder model.

Architecture difference vs ConditionalFlowMatchingModule:
- One ResNetEncoder instead of two.
- No same-instrument neighbor path; the UNet is conditioned only on the
  same-galaxy image from the other survey.
- Everything else (flow matching objective, Euler sampling, logging) is
  identical so comparisons are apples-to-apples.

Data interface: accepts the same (target, samegal, sameins, [masks,] metadata)
batch tuple produced by NeighborsEfficientDataset / collate_neighbors.
sameins and masks are present in the batch but deliberately ignored.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import umap
import wandb
from diffusers import UNet2DConditionModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from galaxy_images.galaxy_model.validation_pairs import reconstruct_hsc_legacy_pairs


class ResNetEncoder(nn.Module):
    """ResNet18 spatial encoder for cross-attention conditioning tokens."""

    def __init__(
        self,
        in_channels: int = 4,
        cross_attention_dim: int = 16,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
        self.proj = nn.Conv2d(512, cross_attention_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, seq_len, cross_attention_dim) conditioning tokens."""
        features = self.backbone(x)
        feat = features[-1]          # (B, 512, H', W')
        feat = self.proj(feat)       # (B, cross_attention_dim, H', W')
        B, D, H, W = feat.shape
        return feat.view(B, D, H * W).permute(0, 2, 1)  # (B, H'*W', D)


class SingleEncoderFlowMatchingModule(pl.LightningModule):
    """
    Ablation: single shared encoder for cross-survey image prediction.

    One ResNetEncoder encodes the same-galaxy conditioning image from the
    other survey. The UNet is conditioned only on those tokens — no
    same-instrument neighbor path exists.
    """

    def __init__(
        self,
        in_channels: int = 4,
        cond_channels: int = 4,
        image_size: int = 48,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        cross_attention_dim: int = 16,
        pretrained_encoder: bool = False,
        all_attention: bool = True,
        lr: float = 1e-4,
        num_sample_images: int = 10,
        num_mse_images: int = 32,
        num_integration_steps: int = 250,
        lambda_generative: float = 1.0,
        num_umap_batches: int = 8,
        mask_center: bool = False,
        figures_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_sample_images = num_sample_images
        self.num_mse_images = num_mse_images
        self.num_integration_steps = num_integration_steps
        self.in_channels = in_channels
        self.image_size = image_size
        self.lambda_generative = lambda_generative
        self.num_umap_batches = num_umap_batches
        self.mask_center = mask_center
        self.figures_dir = Path(figures_dir) if figures_dir else (Path(__file__).resolve().parent / "figures")
        self.is_h100 = _is_h100_gpu()

        block_out_channels = tuple(model_channels * m for m in channel_mult)

        self.encoder = ResNetEncoder(
            in_channels=cond_channels,
            cross_attention_dim=cross_attention_dim,
            pretrained=pretrained_encoder,
        )

        if all_attention:
            down_block_types = (
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            )
            up_block_types = (
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            )
        else:
            down_block_types = (
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            )
            up_block_types = (
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            )

        self.velocity_model = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )

    # ------------------------------------------------------------------
    # Core forward / loss
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_image: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity v(x_t, t, cond_image).

        Args:
            x_t:        Noisy image at time t  (B, C, H, W)
            t:          Time in [0, 1]          (B,)
            cond_image: Same-galaxy conditioning (B, C, H, W) — other survey
        """
        cond_tokens = self.encoder(cond_image)   # (B, seq_len, cross_attention_dim)
        return self.velocity_model(
            x_t,
            t * 1000,
            encoder_hidden_states=cond_tokens,
        ).sample

    def _unpack_batch(self, batch: tuple):
        """Return (anchor, samegal, metadata) regardless of 4- or 5-tuple format."""
        if len(batch) == 5:
            anchor, samegal, _sameins, _masks, metadata = batch
        else:
            anchor, samegal, _sameins, metadata = batch
        return anchor, samegal, metadata

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        x_1, cond_image, metadata = self._unpack_batch(batch)

        batch_size = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=x_1.device)

        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        target_velocity = x_1 - x_0

        predicted_velocity = self(x_t, t, cond_image)

        if self.mask_center:
            _, _, H, W = predicted_velocity.shape
            s = 48
            sy, sx = (H - s) // 2, (W - s) // 2
            loss = nn.functional.mse_loss(
                predicted_velocity[:, :, sy:sy+s, sx:sx+s],
                target_velocity[:, :, sy:sy+s, sx:sx+s],
                reduction="none",
            )
        else:
            loss = nn.functional.mse_loss(predicted_velocity, target_velocity, reduction="none")

        per_example_loss = loss.mean(dim=(1, 2, 3))

        anchor_surveys = [m["anchor_survey"] for m in metadata]
        is_hsc = torch.tensor([s == "hsc" for s in anchor_surveys], device=per_example_loss.device)
        is_legacy = ~is_hsc

        self._loss_hsc = (
            per_example_loss[is_hsc].mean()
            if is_hsc.any()
            else torch.tensor(float("nan"), device=per_example_loss.device)
        )
        self._loss_legacy = (
            per_example_loss[is_legacy].mean()
            if is_legacy.any()
            else torch.tensor(float("nan"), device=per_example_loss.device)
        )
        self._loss_generative_total = per_example_loss.mean().detach()
        return per_example_loss.mean()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        cond_image: torch.Tensor,
        num_steps: Optional[int] = None,
        x_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler-integrate from noise to target conditioned on cond_image."""
        num_steps = num_steps or self.num_integration_steps
        B = cond_image.shape[0]
        device = cond_image.device

        x = (
            x_noise.to(device)
            if x_noise is not None
            else torch.randn(B, self.in_channels, self.image_size, self.image_size, device=device)
        )
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            x = x + self(x, t, cond_image) * dt
        return x

    @torch.no_grad()
    def compute_mse(
        self,
        target_image: torch.Tensor,
        cond_image: torch.Tensor,
        metadata=None,
        mask_sizes: tuple = (48, 32),
    ):
        samples = self.sample(cond_image)
        diff = target_image - samples
        _, _, H, W = diff.shape

        mse_by_size = {}
        for s in mask_sizes:
            sy, sx = (H - s) // 2, (W - s) // 2
            mse_by_size[s] = torch.mean(diff[:, :, sy:sy+s, sx:sx+s] ** 2)

        s0 = mask_sizes[0]
        sy, sx = (H - s0) // 2, (W - s0) // 2
        crop = diff[:, :, sy:sy+s0, sx:sx+s0]

        mse_hsc = mse_legacy = None
        if metadata is not None:
            surveys = [m["anchor_survey"] for m in metadata]
            hsc_m = torch.tensor([s == "hsc" for s in surveys], device=diff.device)
            legacy_m = ~hsc_m
            mse_hsc = crop[hsc_m].pow(2).mean() if hsc_m.any() else torch.tensor(float("nan"), device=diff.device)
            mse_legacy = crop[legacy_m].pow(2).mean() if legacy_m.any() else torch.tensor(float("nan"), device=diff.device)

        return mse_by_size, mse_hsc, mse_legacy

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, "_loss_generative_total"):
            self.log("train/loss_generative_total", self._loss_generative_total, on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, "_loss_hsc"):
            self.log("train/loss_generative_hsc", self._loss_hsc, on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, "_loss_legacy"):
            self.log("train/loss_generative_legacy", self._loss_legacy, on_step=True, on_epoch=True, sync_dist=True)

        if self.global_step % 100 == 0 and hasattr(self, "_train_start_time") and self.global_step > 0:
            elapsed = time.time() - self._train_start_time
            max_steps = self.trainer.max_steps
            if max_steps > 0:
                remaining = (max_steps - self.global_step) / (self.global_step / elapsed)
                print(
                    f"Step {self.global_step}/{max_steps} "
                    f"({100 * self.global_step / max_steps:.1f}%) | "
                    f"Elapsed: {_fmt_hms(elapsed)} | ETA: {_fmt_hms(remaining)} | "
                    f"{self.global_step / elapsed:.2f} steps/s"
                )
        return loss

    def on_train_start(self):
        self._train_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"[SingleEncoder] Training started — target: {self.trainer.max_steps} steps")
        print(f"H100 detected: {self.is_h100}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

    def on_train_end(self):
        if hasattr(self, "_train_start_time"):
            print(f"\n{'='*60}")
            print(f"Training completed! Total time: {_fmt_hms(time.time() - self._train_start_time)}")
            print(f"{'='*60}\n")

    def on_validation_epoch_start(self):
        self._umap_hsc_batches = []
        self._umap_legacy_batches = []
        self._umap_batch_count = 0

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        if hasattr(self, "_loss_generative_total"):
            self.log("val/loss_generative_total", self._loss_generative_total, on_epoch=True, sync_dist=True)
        if hasattr(self, "_loss_hsc"):
            self.log("val/loss_generative_hsc", self._loss_hsc, on_epoch=True, sync_dist=True)
        if hasattr(self, "_loss_legacy"):
            self.log("val/loss_generative_legacy", self._loss_legacy, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            anchor, samegal, metadata = self._unpack_batch(batch)
            self._val_anchor_batch = anchor[:self.num_sample_images].clone()
            self._val_samegal_batch = samegal[:self.num_sample_images].clone()

            n = min(self.num_mse_images, anchor.shape[0])
            self._val_mse_target = anchor[:n].clone()
            self._val_mse_cond = samegal[:n].clone()
            self._val_mse_metadata = metadata[:n] if metadata else None

        if self._umap_batch_count < self.num_umap_batches:
            anchor, samegal, metadata = self._unpack_batch(batch)
            hsc_imgs, legacy_imgs = reconstruct_hsc_legacy_pairs(anchor, samegal, metadata)
            self._umap_hsc_batches.append(hsc_imgs.cpu())
            self._umap_legacy_batches.append(legacy_imgs.cpu())
            self._umap_batch_count += 1

        return loss

    def on_validation_epoch_end(self):
        if not self.logger or not hasattr(self, "_val_anchor_batch"):
            return

        import matplotlib.pyplot as plt

        num_rows = min(6, len(self._val_anchor_batch))
        num_samples = 5
        num_cols = 2 + num_samples + 1   # samegal | target | samples... | mean

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), squeeze=False)
        col_titles = ["SameGal", "Target"] + [f"Sample {j+1}" for j in range(num_samples)] + ["Mean"]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=10)

        for i in range(num_rows):
            cond = self._val_samegal_batch[i : i + 1].to(self.device)
            target = self._val_anchor_batch[i : i + 1].to(self.device)

            cond_rep = cond.repeat(num_samples, 1, 1, 1)
            samples = self.sample(cond_rep)
            mean_sample = samples.mean(dim=0, keepdim=True)

            def _vis(t):
                t = t.clone()
                t = t - t.min()
                if t.max() > 0:
                    t = t / t.max()
                return t[:3].permute(1, 2, 0).cpu().numpy()

            axes[i, 0].imshow(_vis(cond[0]))
            axes[i, 0].axis("off")
            axes[i, 1].imshow(_vis(target[0]))
            axes[i, 1].axis("off")
            for j in range(num_samples):
                axes[i, 2 + j].imshow(_vis(samples[j]))
                axes[i, 2 + j].axis("off")
            axes[i, -1].imshow(_vis(mean_sample[0]))
            axes[i, -1].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({
            "val/sample_grid": wandb.Image(fig),
            "global_step": self.global_step,
        })
        plt.close(fig)

        if hasattr(self, "_val_mse_target"):
            mse_start = time.time()
            mse_by_size, mse_hsc, mse_legacy = self.compute_mse(
                self._val_mse_target.to(self.device),
                self._val_mse_cond.to(self.device),
                self._val_mse_metadata,
                mask_sizes=(48, 32),
            )
            if not hasattr(self, "_mse_timing_logged"):
                print(f"[MSE metric] took {time.time() - mse_start:.2f}s")
                self._mse_timing_logged = True
            self.log("val/mse", mse_by_size[48], sync_dist=True)
            self.log("val/mse_32", mse_by_size[32], sync_dist=True)
            if mse_hsc is not None:
                self.log("val/mse_hsc", mse_hsc, sync_dist=True)
            if mse_legacy is not None:
                self.log("val/mse_legacy", mse_legacy, sync_dist=True)

        if self._umap_batch_count > 0:
            try:
                hsc = torch.cat(self._umap_hsc_batches, dim=0).to(self.device)
                leg = torch.cat(self._umap_legacy_batches, dim=0).to(self.device)
                umap_path = self._plot_latent_space(hsc, leg)
                print(f"[UMAP] saved to {umap_path}")
            except Exception as e:
                import traceback
                print(f"[UMAP] error: {e}")
                traceback.print_exc()

    @torch.no_grad()
    def _plot_latent_space(self, hsc_batch: torch.Tensor, legacy_batch: torch.Tensor) -> Path:
        import matplotlib.pyplot as plt

        hsc_emb = self.encoder(hsc_batch)      # (B, seq_len, D)
        leg_emb = self.encoder(legacy_batch)

        num_hsc = hsc_emb.shape[0]
        all_emb = torch.cat([hsc_emb, leg_emb], dim=0)  # (2B, seq_len, D)
        seq_len = all_emb.shape[1]

        umap_params = dict(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=42, n_jobs=1)

        fig, axes = plt.subplots(seq_len + 1, 1, figsize=(8, 6 * (seq_len + 1)), squeeze=False)

        for tok in range(seq_len):
            tok_emb = all_emb[:, tok, :].cpu().numpy()
            reducer = umap.UMAP(**umap_params)
            proj = reducer.fit_transform(tok_emb)
            ax = axes[tok, 0]
            ax.scatter(proj[:num_hsc, 0], proj[:num_hsc, 1], s=5, alpha=0.6, c="blue", label="HSC")
            ax.scatter(proj[num_hsc:, 0], proj[num_hsc:, 1], s=5, alpha=0.6, c="orange", label="Legacy")
            ax.set_title(f"Single Encoder — Token {tok}", fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Combined
        all_flat = all_emb.flatten(start_dim=1).cpu().numpy()
        proj_combined = umap.UMAP(**umap_params).fit_transform(all_flat)
        ax_c = axes[seq_len, 0]
        ax_c.scatter(proj_combined[:num_hsc, 0], proj_combined[:num_hsc, 1], s=5, alpha=0.6, c="blue", label="HSC")
        ax_c.scatter(proj_combined[num_hsc:, 0], proj_combined[num_hsc:, 1], s=5, alpha=0.6, c="orange", label="Legacy")
        ax_c.set_title("Single Encoder — Combined (All Tokens)", fontsize=10)
        ax_c.legend()
        ax_c.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / f"umap_step{self.global_step}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.log({
                "latent_space/umap": wandb.Image(str(path)),
                "global_step": self.global_step,
            })
        return path

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _is_h100_gpu() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        return any("h100" in torch.cuda.get_device_name(i).lower() for i in range(torch.cuda.device_count()))
    except Exception:
        return False
