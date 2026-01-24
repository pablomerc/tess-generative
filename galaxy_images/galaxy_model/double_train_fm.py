import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import timm
import time
import sys
import shutil
import datetime
from pathlib import Path
from diffusers import UNet2DConditionModel, UNet2DModel
from typing import Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class ResNetEncoder(nn.Module):
    """
    ResNet18 encoder from timm that produces spatial feature maps for conditioning.
    Uses feature extraction to get intermediate spatial features for cross-attention.
    """

    def __init__(
        self,
        in_channels: int = 4,
        cross_attention_dim: int = 256,
        pretrained: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),  # Get features from layer2, layer3, layer4
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
        """
        Args:
            x: Conditioning image (B, C, H, W)
        Returns:
            Spatial embeddings (B, seq_len, cross_attention_dim) for cross-attention
        """
        features = self.backbone(x)
        feat = features[-1]  # (B, 512, H/32, W/32)
        feat = self.proj(feat)  # (B, cross_attention_dim, H', W')

        B, D, H, W = feat.shape
        feat = feat.view(B, D, H * W).permute(0, 2, 1)
        return feat


class ConditionalFlowMatchingModule(pl.LightningModule):
    """
    Conditional Flow Matching model using optimal transport conditional paths.

    Conditions on a second image (e.g., Legacy Survey) to generate the target
    image (e.g., HSC).

    The interpolation is: x_t = (1 - t) * x_0 + t * x_1
    where x_0 ~ N(0, I) (noise), x_1 ~ target data.

    The target velocity is: v(x_t, t, c) = x_1 - x_0
    where c is the conditioning image embedding.

    Args:
        concat_conditioning: If True, concatenate conditioning image directly to
            input (no encoder, uses UNet2DModel). If False, use ResNet encoder
            with cross-attention (uses UNet2DConditionModel).
    """

    def __init__(
        self,
        #DATA PARAMS
        in_channels: int = 4, # Channels in the target domain (image)
        cond_channels: int = 4, # Encoder input channels or concatenated channels for conditioning
        image_size: int = 64, # Spatial size of the image
        #UNET PARAMS
        model_channels: int = 128, # Base channel width for the unet
        channel_mult: tuple = (1, 2, 4, 4), # channel multiplier for each block. we downsample spatially and increase the channels
        layers_per_block: int = 2, # resnet-like layers in each unet block
        attention_head_dim: int = 8, # head dimension used by attention blocks inside diffusers unet
        # Conditioning params
        cross_attention_dim: int = 256, # cross-attention mode (conditionning mode). must match the resnet encoder cross att dim and the unet encoding dim
        pretrained_encoder: bool = False, # load pretrained imagenet weights
        concat_conditioning: bool = False, # if true -> no encoder, conditioning is concatenated as extra channels to the input
        # Optimization params
        lr: float = 1e-4,
        num_sample_images: int = 8, # number of exmaples cached for first validation batch for W&B
        num_integration_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_sample_images = num_sample_images
        self.num_integration_steps = num_integration_steps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.image_size = image_size
        self.concat_conditioning = concat_conditioning

        block_out_channels = tuple(model_channels * m for m in channel_mult)

        if concat_conditioning:
            raise ValueError("Concat conditioning is not supported for the double encoder case")
        else:
            self.encoder_1 = ResNetEncoder(
                in_channels=cond_channels,
                cross_attention_dim=cross_attention_dim,
                pretrained=pretrained_encoder,
            )

            self.encoder_2 = ResNetEncoder(
                in_channels=cond_channels,
                cross_attention_dim=cross_attention_dim,
                pretrained=pretrained_encoder,
            )

            self.velocity_model = UNet2DConditionModel(
                sample_size=image_size,
                in_channels=in_channels,
                out_channels=in_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=(
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=cross_attention_dim,
                attention_head_dim=attention_head_dim,
            )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_image_samegal: torch.Tensor,
        cond_image_sameins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t, c).

        Args:
            x_t: Noisy image at time t (B, C, H, W)
            t: Time in [0, 1] (B,)
            cond_image_samegal: Conditioning image (B, C, H, W)
            cond_image_sameins: Set of conditioning images (B, k, C, H, W)
        """
        timesteps = t * 1000

        cond_gal_embedding = self.encoder_1(cond_image_samegal)  # (B, seq_len, embed_dim)

        B, k, C, H, W = cond_image_sameins.shape

        cond_image_sameins_flat = cond_image_sameins.flatten(0, 1)          # (B*k, C, H, W)
        cond_ins_embedding_flat = self.encoder_2(cond_image_sameins_flat)      # (B*k, seq_len, embed_dim)

        cond_ins_embedding = cond_ins_embedding_flat.unflatten(0, (B, k))    # (B, k, seq_len, embed_dim)
        cond_ins_embedding = cond_ins_embedding.flatten(1, 2)                # (B, k*seq_len, embed_dim)

        cond_embedding = torch.cat([cond_gal_embedding, cond_ins_embedding], dim=1)
        # (B, (1+k)*seq_len, embed_dim)

        return self.velocity_model(
            x_t,
            timesteps,
            encoder_hidden_states=cond_embedding,
        ).sample

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """Compute conditional flow matching loss.

        Args:
            Batch: (anchor_image, same_galaxy, same_instrument, metadata)

        """
        x_1, cond_image_samegal, cond_image_sameins, metadata = batch

        batch_size = x_1.shape[0]

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=x_1.device)

        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        target_velocity = x_1 - x_0

        predicted_velocity = self(x_t, t, cond_image_samegal, cond_image_sameins)

        loss = nn.functional.mse_loss(predicted_velocity, target_velocity, reduction='none')

        # Reduce to per-example losses: (B, C, H, W) -> (B,)
        per_example_loss = loss.mean(dim=(1, 2, 3))

        # Extract anchor_survey values
        anchor_surveys = [m['anchor_survey'] for m in metadata]

        # Create boolean masks (on the same device as your loss tensor)
        is_hsc = torch.tensor([s == 'hsc' for s in anchor_surveys], device=per_example_loss.device)
        is_legacy = torch.tensor([s == 'legacy' for s in anchor_surveys], device=per_example_loss.device)

        # Compute mean losses for each group
        loss_hsc = per_example_loss[is_hsc].mean() if is_hsc.any() else torch.tensor(float('nan'), device=per_example_loss.device)
        loss_legacy = per_example_loss[is_legacy].mean() if is_legacy.any() else torch.tensor(float('nan'), device=per_example_loss.device)

        # Total loss (scalar) for gradients
        total_loss = per_example_loss.mean()

        # Store separate losses for logging (detached to be explicit they're not used for gradients)
        self._loss_hsc = loss_hsc.detach()
        self._loss_legacy = loss_legacy.detach()

        return total_loss

    #TODO: Remove time logging (added for debugging purposes)
    def _format_time_hms(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def on_train_start(self):
        """Record training start time."""
        self._train_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training started - Target: {self.trainer.max_steps} steps")
        print(f"{'='*60}\n")

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # Log separate losses for hsc and legacy anchors
        if hasattr(self, '_loss_hsc'):
            self.log("train/loss_hsc", self._loss_hsc, on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_legacy'):
            self.log("train/loss_legacy", self._loss_legacy, on_step=True, on_epoch=True, sync_dist=True)

        # Print time estimates periodically (every 100 steps)
        if self.global_step % 100 == 0 and hasattr(self, '_train_start_time') and self.global_step > 0:
            elapsed_time = time.time() - self._train_start_time
            max_steps = self.trainer.max_steps

            if max_steps > 0:
                steps_per_second = self.global_step / elapsed_time
                remaining_steps = max_steps - self.global_step
                estimated_remaining = remaining_steps / steps_per_second
                progress = (self.global_step / max_steps) * 100

                elapsed_str = self._format_time_hms(elapsed_time)
                remaining_str = self._format_time_hms(estimated_remaining)

                print(f"Step {self.global_step}/{max_steps} ({progress:.1f}%) | "
                      f"Elapsed: {elapsed_str} | ETA: {remaining_str} | "
                      f"Speed: {steps_per_second:.2f} steps/s")

        return loss

    def on_train_epoch_start(self):
        """Record epoch start time."""
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        """Print epoch time at the end of each epoch."""
        if hasattr(self, '_epoch_start_time'):
            epoch_time = time.time() - self._epoch_start_time
            epoch_str = self._format_time_hms(epoch_time)
            print(f"Epoch {self.current_epoch} completed in {epoch_str}")

    def on_train_end(self):
        """Print total training time at the end."""
        if hasattr(self, '_train_start_time'):
            total_time = time.time() - self._train_start_time
            total_str = self._format_time_hms(total_time)

            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Total training time: {total_str}")
            print(f"Total steps: {self.global_step}")
            print(f"{'='*60}\n")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Log separate losses for hsc and legacy anchors
        if hasattr(self, '_loss_hsc'):
            self.log("val/loss_hsc", self._loss_hsc, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_legacy'):
            self.log("val/loss_legacy", self._loss_legacy, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            anchor_image, same_galaxy, same_instrument, _ = batch
            self._val_anchor_batch = anchor_image[:self.num_sample_images].clone()
            self._val_samegal_batch = same_galaxy[:self.num_sample_images].clone()
            self._val_sameins_batch = same_instrument[:self.num_sample_images].clone()

        return loss

    @torch.no_grad()
    def sample(
        self,
        cond_image_samegal: torch.Tensor,
        cond_image_sameins: torch.Tensor,
        num_steps: Optional[int] = None,
        x_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples conditioned on input images using Euler integration.

        If a noise sample is generated outside before calling this method, it should follow
        x = torch.randn(
                num_samples, self.in_channels, self.image_size, self.image_size,
                device=device,
            )

        Args:
            cond_image_samegal: Same galaxy conditioning images (B, C, H, W)
            cond_image_sameins: Same instrument conditioning images (B, k, C, H, W)
            num_steps: Number of integration steps
            x_noise: Optional noise sample (B, C, H, W). If provided, must match batch size
                and be on the same device as cond_image_samegal.
        """
        num_steps = num_steps or self.num_integration_steps
        num_samples = cond_image_samegal.shape[0]
        device = cond_image_samegal.device

        if x_noise is None:
            x = torch.randn(
                num_samples, self.in_channels, self.image_size, self.image_size,
                device=device,
            )
        else:
            # Ensure x_noise is on the correct device and has correct shape
            x = x_noise.to(device)
            expected_shape = (num_samples, self.in_channels, self.image_size, self.image_size)
            if x.shape != expected_shape:
                raise ValueError(
                    f"x_noise shape {x.shape} does not match expected shape {expected_shape}"
                )

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            velocity = self(x, t, cond_image_samegal, cond_image_sameins)
            x = x + velocity * dt

        return x

    def _normalize_for_vis(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] for visualization."""
        img = img.clone()
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return img

    def on_validation_epoch_end(self) -> None:
        """Log sampled images as a grid to W&B.

        Creates a grid where each row corresponds to one conditioning image:
        [SameGal | SameIns (first) | Target | Sample1 | Sample2 | ... | SampleN | Mean]
        """
        if not self.logger or not hasattr(self, "_val_anchor_batch"):
            return

        import matplotlib.pyplot as plt
        import torch
        import wandb

        num_cond_images = min(6, len(self._val_anchor_batch))
        num_samples_per_cond = 5
        num_cols = 3 + num_samples_per_cond + 1  # samegal + sameins_first + target + samples + mean

        def _row_scale_rgb(x_chw: torch.Tensor, vmin, vmax) -> torch.Tensor:
            """
            Scale a (3,H,W) tensor to (H,W,3) in [0,1] using fixed per-channel vmin/vmax.
            vmin/vmax: tensor-like shape (3,)
            """
            x = x_chw[:3]
            vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
            vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
            y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
            y = y.clamp(0, 1)
            return y.permute(1, 2, 0)

        # --- ORIGINAL GRID ---
        fig_orig, axes_orig = plt.subplots(
            num_cond_images, num_cols,
            figsize=(2 * num_cols, 2 * num_cond_images),
            squeeze=False,
        )
        col_titles = ["SameGal", "SameIns (1st)", "Target"] + [f"Sample {j+1}" for j in range(num_samples_per_cond)] + ["Mean"]
        for j, title in enumerate(col_titles):
            axes_orig[0, j].set_title(title, fontsize=10)

        # --- ROW-SCALED GRID (new) ---
        fig_row, axes_row = plt.subplots(
            num_cond_images, num_cols,
            figsize=(2 * num_cols, 2 * num_cond_images),
            squeeze=False,
        )
        for j, title in enumerate(col_titles):
            axes_row[0, j].set_title(title, fontsize=10)

        for i in range(num_cond_images):
            samegal = self._val_samegal_batch[i : i + 1].to(self.device)
            target = self._val_anchor_batch[i : i + 1].to(self.device)
            sameins = self._val_sameins_batch[i : i + 1].to(self.device)  # (1, k, C, H, W)
            sameins_first = sameins[:, 0:1]  # (1, 1, C, H, W) - first same instrument image

            # Repeat samegal and sameins for multiple samples
            samegal_repeated = samegal.repeat(num_samples_per_cond, 1, 1, 1)
            sameins_repeated = sameins.repeat(num_samples_per_cond, 1, 1, 1, 1)  # (num_samples_per_cond, k, C, H, W)

            samples = self.sample(samegal_repeated, sameins_repeated)
            mean_sample = samples.mean(dim=0, keepdim=True)

            # =========================
            # (A) ORIGINAL PLOTTING ROW
            # =========================
            # SameGal column
            samegal_rgb = self._normalize_for_vis(samegal[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, 0].imshow(samegal_rgb)
            axes_orig[i, 0].axis("off")

            # SameIns (first) column
            sameins_first_rgb = self._normalize_for_vis(sameins_first[0, 0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, 1].imshow(sameins_first_rgb)
            axes_orig[i, 1].axis("off")

            # Target column
            target_rgb = self._normalize_for_vis(target[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, 2].imshow(target_rgb)
            axes_orig[i, 2].axis("off")

            # Sample columns
            for j in range(num_samples_per_cond):
                sample_rgb = self._normalize_for_vis(samples[j, :3]).cpu().permute(1, 2, 0).numpy()
                axes_orig[i, 3 + j].imshow(sample_rgb)
                axes_orig[i, 3 + j].axis("off")

            # Mean column
            mean_rgb = self._normalize_for_vis(mean_sample[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, -1].imshow(mean_rgb)
            axes_orig[i, -1].axis("off")

            # =========================
            # (B) ROW-SCALED PLOTTING
            # =========================
            # Compute per-channel vmin/vmax from the TARGET for this row
            target_chw = target[0, :3]  # (3,H,W)

            # Use min/max (exactly matches your "if flux=10 is max in target" idea)
            vmin = target_chw.amin(dim=(1, 2))  # (3,)
            vmax = target_chw.amax(dim=(1, 2))  # (3,)

            # If you prefer robust scaling (optional), replace above with:
            # flat = target_chw.flatten(1)  # (3, H*W)
            # vmin = torch.quantile(flat, 0.01, dim=1)
            # vmax = torch.quantile(flat, 0.99, dim=1)

            # SameGal column
            samegal_vis = _row_scale_rgb(samegal[0, :3], vmin, vmax).detach().cpu().numpy()
            axes_row[i, 0].imshow(samegal_vis)
            axes_row[i, 0].axis("off")

            # SameIns (first) column
            sameins_first_vis = _row_scale_rgb(sameins_first[0, 0, :3], vmin, vmax).detach().cpu().numpy()
            axes_row[i, 1].imshow(sameins_first_vis)
            axes_row[i, 1].axis("off")

            # Target column
            target_vis = _row_scale_rgb(target[0, :3], vmin, vmax).detach().cpu().numpy()
            axes_row[i, 2].imshow(target_vis)
            axes_row[i, 2].axis("off")

            # Sample columns
            for j in range(num_samples_per_cond):
                samp_vis = _row_scale_rgb(samples[j, :3], vmin, vmax).detach().cpu().numpy()
                axes_row[i, 3 + j].imshow(samp_vis)
                axes_row[i, 3 + j].axis("off")

            # Mean column
            mean_vis = _row_scale_rgb(mean_sample[0, :3], vmin, vmax).detach().cpu().numpy()
            axes_row[i, -1].imshow(mean_vis)
            axes_row[i, -1].axis("off")

        plt.figure(fig_orig.number)
        plt.tight_layout()
        plt.figure(fig_row.number)
        plt.tight_layout()

        self.logger.experiment.log({
            "val/sample_grid": wandb.Image(fig_orig),
            "val/sample_grid_row_scaled": wandb.Image(fig_row),
            "global_step": self.global_step,
        })

        plt.close(fig_orig)
        plt.close(fig_row)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    # Set up snapshot + tee logging before anything else in main runs

    from pytorch_lightning.loggers import WandbLogger
    from torch.utils.data import DataLoader, TensorDataset
    from data import HSCLegacyTripletDataset, BalancedAnchorBatchSampler, custom_collate_fn

    batch_size = 64
    wandb_project = "galaxy-flow-matching"  # Change this to your desired wandb project name

    train_dataset = HSCLegacyTripletDataset(
        hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
        idx_list=list(range(95_000)),
    )
    val_dataset = HSCLegacyTripletDataset(
        hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
        idx_list=list(range(95_000, 100_000)),
        deterministic_anchor_survey=True,  # Make validation batches consistent
    )

    # train_dataset = HSCLegacyTripletDataset(
    #     hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_48x48_laptop.h5',
    #     idx_list=list(range(5000)),
    # )
    # val_dataset = HSCLegacyTripletDataset(
    #     hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_48x48_laptop.h5',
    #     idx_list=list(range(5000, 5140)),
    #     deterministic_anchor_survey=True,  # Make validation batches consistent
    # )

    # TODO: (Future) - Fix BalancedAnchorBatchSampler and use it here instead
    # train_batch_sampler = BalancedAnchorBatchSampler(
    #     num_samples=len(train_dataset),
    #     batch_size=batch_size,
    #     drop_last=True,
    #     seed=0
    # )

    # val_batch_sampler = BalancedAnchorBatchSampler(
    #     num_samples=len(val_dataset),
    #     batch_size=batch_size,
    #     drop_last=True,
    #     seed=0
    # )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # batch_sampler=train_batch_sampler,  # important: use batch_sampler, not batch_size/shuffle
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Use same collate function
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        # batch_sampler=val_batch_sampler,  # important: use batch_sampler, not batch_size/shuffle
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Use same collate function
    )


    concat_conditioning = False
    model = ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        image_size=48,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=64,
        pretrained_encoder=False,
        concat_conditioning=concat_conditioning,
        lr=1e-4,
        num_sample_images=10,
        num_integration_steps=250,
    )

    if concat_conditioning:
        name="conditional-unet2d-concatenated"
    else:
        name="double-encoder-resnet18-triplet-zdim64"
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=name,
        log_model=False,
    )

    n_devices = 4
    trainer = pl.Trainer(
        max_steps=300_000/n_devices,
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
       log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model, train_loader, val_loader)
