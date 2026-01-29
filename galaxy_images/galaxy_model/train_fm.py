from numpy._core.numeric import False_
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


def setup_run_snapshot() -> Path:
    """
    Create a timestamped run directory, copy key source files into it,
    and tee stdout/stderr into a log file while still printing to terminal.

    Returns:
        Path to the created run directory.
    """

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    runs_dir = script_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(exist_ok=True)

    # Snapshot key source files
    src_files = [
        script_dir / "train_fm.py",
        script_dir / "data.py",
    ]
    for src in src_files:
        if src.exists():
            shutil.copy2(src, run_dir / src.name)

    # Set up tee-style logging
    log_path = run_dir / "train.log"

    class _Tee:
        def __init__(self, stream, log_file):
            self._stream = stream
            self._log_file = log_file

        def write(self, data):
            self._stream.write(data)
            self._log_file.write(data)

        def flush(self):
            self._stream.flush()
            self._log_file.flush()

    # Line-buffered text file for immediate writes
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    print(f"[run snapshot] Logging to {log_path}")
    print(f"[run snapshot] Source snapshot stored in {run_dir}")

    return run_dir


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
        num_mse_images: int = 64, # number of examples cached for MSE tracking
        num_integration_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_sample_images = num_sample_images
        self.num_mse_images = num_mse_images
        self.num_integration_steps = num_integration_steps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.image_size = image_size
        self.concat_conditioning = concat_conditioning

        block_out_channels = tuple(model_channels * m for m in channel_mult)

        if concat_conditioning:
            # Don't create encoder attribute when using concatenation
            # Setting it to None causes DDP parameter count inconsistencies
            self.velocity_model = UNet2DModel(
                sample_size=image_size,
                in_channels=in_channels + cond_channels,  # x_t + cond concatenated
                out_channels=in_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                ),
                attention_head_dim=attention_head_dim,
            )
        else:
            self.encoder = ResNetEncoder(
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
        cond_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t, c).

        Args:
            x_t: Noisy image at time t (B, C, H, W)
            t: Time in [0, 1] (B,)
            cond_image: Conditioning image (B, C, H, W)
        """
        timesteps = t * 1000

        if self.concat_conditioning:
            x_input = torch.cat([x_t, cond_image], dim=1)
            return self.velocity_model(x_input, timesteps).sample
        else:
            cond_embedding = self.encoder(cond_image)  # (B, seq_len, embed_dim)
            return self.velocity_model(
                x_t,
                timesteps,
                encoder_hidden_states=cond_embedding,
            ).sample

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """Compute conditional flow matching loss."""
        x_1, cond_image = batch
        batch_size = x_1.shape[0]

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=x_1.device)

        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        target_velocity = x_1 - x_0

        predicted_velocity = self(x_t, t, cond_image)

        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

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


    @torch.no_grad()
    def compute_mse(self, target_image, cond_image):
        '''Compute reconstruction MSE on a batch of given images
        Args:
            target_image (B,C,H,W)
            cond_image (B,C,H,W)
        '''
        samples = self.sample(cond_image)

        diff = target_image - samples

        mse_error = torch.mean(diff**2)

        return mse_error


    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self._val_cond_batch = batch[1][:self.num_sample_images].clone()
            self._val_target_batch = batch[0][:self.num_sample_images].clone()

            batch_size = batch[0].shape[0]
            num_mse_images = (self.num_mse_images if self.num_mse_images <= batch_size else batch_size)
            self._val_mse_cond_batch = batch[1][:num_mse_images].clone()
            self._val_mse_target_batch = batch[0][:num_mse_images].clone()

        return loss

    @torch.no_grad()
    def sample(
        self,
        cond_images: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples conditioned on input images using Euler integration.

        Args:
            cond_images: Conditioning images (B, C, H, W)
            num_steps: Number of integration steps
        """
        num_steps = num_steps or self.num_integration_steps
        num_samples = cond_images.shape[0]
        device = cond_images.device

        x = torch.randn(
            num_samples, self.in_channels, self.image_size, self.image_size,
            device=device,
        )

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            velocity = self(x, t, cond_images)
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
        [Cond | Target | Sample1 | Sample2 | ... | SampleN | Mean]
        """
        if not self.logger or not hasattr(self, "_val_cond_batch"):
            return

        import matplotlib.pyplot as plt
        import torch
        import wandb

        num_cond_images = min(6, len(self._val_cond_batch))
        num_samples_per_cond = 5
        num_cols = 2 + num_samples_per_cond + 1  # cond + target + samples + mean

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
        col_titles = ["Cond", "Target"] + [f"Sample {j+1}" for j in range(num_samples_per_cond)] + ["Mean"]
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
            cond = self._val_cond_batch[i : i + 1].to(self.device)
            target = self._val_target_batch[i : i + 1].to(self.device)

            cond_repeated = cond.repeat(num_samples_per_cond, 1, 1, 1)
            samples = self.sample(cond_repeated)
            mean_sample = samples.mean(dim=0, keepdim=True)

            # =========================
            # (A) ORIGINAL PLOTTING ROW
            # =========================
            cond_rgb = self._normalize_for_vis(cond[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, 0].imshow(cond_rgb)
            axes_orig[i, 0].axis("off")

            target_rgb = self._normalize_for_vis(target[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, 1].imshow(target_rgb)
            axes_orig[i, 1].axis("off")

            for j in range(num_samples_per_cond):
                sample_rgb = self._normalize_for_vis(samples[j, :3]).cpu().permute(1, 2, 0).numpy()
                axes_orig[i, 2 + j].imshow(sample_rgb)
                axes_orig[i, 2 + j].axis("off")

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

            cond_vis = _row_scale_rgb(cond[0, :3], vmin, vmax).detach().cpu().numpy()
            axes_row[i, 0].imshow(cond_vis)
            axes_row[i, 0].axis("off")

            target_vis = _row_scale_rgb(target[0, :3], vmin, vmax).detach().cpu().numpy()
            axes_row[i, 1].imshow(target_vis)
            axes_row[i, 1].axis("off")

            for j in range(num_samples_per_cond):
                samp_vis = _row_scale_rgb(samples[j, :3], vmin, vmax).detach().cpu().numpy()
                axes_row[i, 2 + j].imshow(samp_vis)
                axes_row[i, 2 + j].axis("off")

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

        # Compute MSE metric
        if hasattr(self, '_val_mse_cond_batch') and hasattr(self, '_val_mse_target_batch'):
            mse_start_time = time.time()
            mse = self.compute_mse(
                self._val_mse_target_batch.to(self.device),
                self._val_mse_cond_batch.to(self.device)
            )
            mse_time = time.time() - mse_start_time

            # Print timing on first validation run
            if not hasattr(self, '_mse_timing_logged'):
                print(f"[MSE metric] Computation took {mse_time:.2f} seconds")
                self._mse_timing_logged = True

            self.log("val/mse", mse, sync_dist=True)



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    # Set up snapshot + tee logging before anything else in main runs
    setup_run_snapshot()

    from pytorch_lightning.loggers import WandbLogger
    from torch.utils.data import DataLoader, TensorDataset
    from data import HSCLegacyDataset

    batch_size = 64
    wandb_project = "galaxy-flow-matching"  # Change this to your desired wandb project name

    train_dataset = HSCLegacyDataset(
        hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
        idx_list=list(range(95_000)),
    )
    val_dataset = HSCLegacyDataset(
        hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
        idx_list=list(range(95_000, 100_000)),
    )

    # train_dataset = HSCLegacyDataset(
    #     hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_48x48_laptop.h5',
    #     idx_list=list(range(5000)),
    # )
    # val_dataset = HSCLegacyDataset(
    #     hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_48x48_laptop.h5',
    #     idx_list=list(range(5000, 5140)),
    # )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    concat_conditioning = True
    model = ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        image_size=48,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=512,
        pretrained_encoder=False,
        concat_conditioning=concat_conditioning,
        lr=1e-4,
        num_sample_images=6,
        num_integration_steps=250,
    )

    if concat_conditioning:
        name="conditional-unet2d-concatenated-100k"
    else:
        name="conditional-unet2d-resnet18-100k-z_dim128"
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
        # strategy="ddp_find_unused_parameters_true", #TODO: Remove this if not needed -- only to show that we get no conditioning if there's no cross-attention
        log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model, train_loader, val_loader)
