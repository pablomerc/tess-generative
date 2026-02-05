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
import geomloss
import umap
import numpy as np



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

    def intermediate_states(self, x: torch.Tensor) -> list:
        features = self.backbone(x)
        for i, element in enumerate(features):
            print(f'Shape of features element {i} is {element.shape}')
        return features


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
        lambda_generative: float = 1.0, # weight for generative loss
        lambda_geometric: float = 0.3, # weight for geometric loss
        num_umap_batches: int = 8, # number of validation batches to collect for UMAP visualization
        mask_center: bool = False, # if true -> mask the center of the image
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
        self.lambda_generative = lambda_generative
        self.lambda_geometric = lambda_geometric
        self.num_umap_batches = num_umap_batches
        self.mask_center = mask_center

        # Detect H100 GPU
        self.is_h100 = is_h100_gpu()

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

        # Initialize geometric loss function once (reused across all training steps)
        if self.lambda_geometric > 0:
            self.geom_loss_fn = geomloss.SamplesLoss(
                loss='sinkhorn',
                p=2,
                blur=0.01,
                backend='tensorized',
                debias=True)
        else:
            self.geom_loss_fn = None

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

        ### Generative loss
        if self.lambda_generative > 0:
            batch_size = x_1.shape[0]

            x_0 = torch.randn_like(x_1)
            t = torch.rand(batch_size, device=x_1.device)

            t_expanded = t[:, None, None, None]
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

            target_velocity = x_1 - x_0

            predicted_velocity = self(x_t, t, cond_image_samegal, cond_image_sameins)

            if self.mask_center:
                mask_size = 48
                _, _, height, width = predicted_velocity.shape
                start_x = (width - mask_size) // 2
                start_y = (height - mask_size) // 2
                loss = nn.functional.mse_loss(predicted_velocity[:, :, start_y:start_y+mask_size, start_x:start_x+mask_size], target_velocity[:, :, start_y:start_y+mask_size, start_x:start_x+mask_size], reduction='none')
            else:
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
            generative_loss = per_example_loss.mean()

            # Store separate losses for logging (detached to be explicit they're not used for gradients)
            self._loss_generative_total = generative_loss.detach()
            self._loss_hsc = loss_hsc.detach()
            self._loss_legacy = loss_legacy.detach()
        else:
            # Skip computation when lambda_generative is 0
            device = x_1.device
            dtype = x_1.dtype
            generative_loss = torch.tensor(0.0, device=device, dtype=dtype)
            self._loss_generative_total = generative_loss.detach()
            self._loss_hsc = torch.tensor(float('nan'), device=device)
            self._loss_legacy = torch.tensor(float('nan'), device=device)

        ### Geometric loss
        if self.lambda_geometric > 0:
            embeds_target = self.encoder_1(x_1).contiguous()  # (B, seq_len, embed_dim)
            embeds_samegal = self.encoder_1(cond_image_samegal).contiguous()  # (B, seq_len, embed_dim)

            # Flatten embeddings before computing geometric loss
            embeds_target = embeds_target.flatten(start_dim=1)  # (B, seq_len * embed_dim)
            embeds_samegal = embeds_samegal.flatten(start_dim=1)  # (B, seq_len * embed_dim)

            # Compute geometric loss (scalar for the entire batch)
            total_geom_loss = self.geom_loss_fn(embeds_target, embeds_samegal)

            # Store geometric loss for logging
            self._loss_geom_total = total_geom_loss.detach()
        else:
            # Skip computation when lambda_geometric is 0
            device = x_1.device
            dtype = x_1.dtype
            total_geom_loss = torch.tensor(0.0, device=device, dtype=dtype)
            self._loss_geom_total = total_geom_loss.detach()

        total_loss = self.lambda_generative * generative_loss + self.lambda_geometric * total_geom_loss

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
        print(f"H100 GPU detected: {self.is_h100}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

        # Explicitly log important hyperparameters to wandb
        if self.logger and hasattr(self.logger, 'experiment'):
            if hasattr(self.logger.experiment, 'config'):
                self.logger.experiment.config.update({
                    "cross_attention_dim": self.hparams.cross_attention_dim,
                    "lambda_generative": self.lambda_generative,
                    "lambda_geometric": self.lambda_geometric,
                    "is_h100": self.is_h100,
                })

        # Cache lens batch from second val dataloader when present (for 48x48 lens validation)
        val_loaders = getattr(self.trainer, 'val_dataloaders', None)
        if val_loaders is not None and len(val_loaders) > 1:
            self._lense_batch = next(iter(val_loaders[1]))
            self._lense_batch_size = self._lense_batch[0].shape[0]
        else:
            self._lense_batch = None
            self._lense_batch_size = 0

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # Log generative losses
        if hasattr(self, '_loss_generative_total'):
            self.log("train/loss_generative_total", self._loss_generative_total, on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_hsc'):
            self.log("train/loss_generative_hsc", self._loss_hsc, on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_legacy'):
            self.log("train/loss_generative_legacy", self._loss_legacy, on_step=True, on_epoch=True, sync_dist=True)

        # Log geometric losses
        if hasattr(self, '_loss_geom_total'):
            self.log("train/loss_geom_total", self._loss_geom_total, on_step=True, on_epoch=True, sync_dist=True)

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

    def on_validation_epoch_start(self):
        """Initialize lists for collecting batches for UMAP visualization."""
        self._umap_hsc_batches = []
        self._umap_legacy_batches = []
        self._umap_batch_count = 0

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

    def validation_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        # Second dataloader (dataloader_idx=1) is the lens batch: only cache for row-scaled plot
        if dataloader_idx == 1:
            anchor_image, same_galaxy, same_instrument, metadata = batch
            self._val_lens_anchor_batch = anchor_image.clone()
            self._val_lens_samegal_batch = same_galaxy.clone()
            self._val_lens_sameins_batch = same_instrument.clone()
            loss = self.compute_loss(batch)
            self.log("val/loss_lenses", loss, prog_bar=False, on_epoch=True, sync_dist=True)
            return loss

        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Log generative losses
        if hasattr(self, '_loss_generative_total'):
            self.log("val/loss_generative_total", self._loss_generative_total, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_hsc'):
            self.log("val/loss_generative_hsc", self._loss_hsc, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_legacy'):
            self.log("val/loss_generative_legacy", self._loss_legacy, on_epoch=True, sync_dist=True)

        # Log geometric losses
        if hasattr(self, '_loss_geom_total'):
            self.log("val/loss_geom_total", self._loss_geom_total, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            anchor_image, same_galaxy, same_instrument, metadata = batch
            self._val_anchor_batch = anchor_image[:self.num_sample_images].clone()
            self._val_samegal_batch = same_galaxy[:self.num_sample_images].clone()
            self._val_sameins_batch = same_instrument[:self.num_sample_images].clone()

            batch_size = anchor_image.shape[0]
            num_mse_images = (self.num_mse_images if self.num_mse_images <= batch_size else batch_size)
            self._val_mse_target_batch = anchor_image[:num_mse_images].clone()
            self._val_mse_samegal_batch = same_galaxy[:num_mse_images].clone()
            self._val_mse_sameins_batch = same_instrument[:num_mse_images].clone()
            self._val_mse_metadata = metadata[:num_mse_images] if metadata else None

        # Collect batches for UMAP visualization
        if (hasattr(self, '_umap_batch_count') and
            self._umap_batch_count < self.num_umap_batches):
            anchor_image, same_galaxy, same_instrument, metadata = batch

            # Separate HSC and Legacy images based on anchor_survey
            anchor_surveys = [m['anchor_survey'] for m in metadata]
            hsc_mask = torch.tensor([s == 'hsc' for s in anchor_surveys], device=anchor_image.device)
            legacy_mask = torch.tensor([s == 'legacy' for s in anchor_surveys], device=anchor_image.device)

            # Collect HSC images (anchor_image when anchor_survey == 'hsc')
            if hsc_mask.any():
                hsc_images = anchor_image[hsc_mask]
                self._umap_hsc_batches.append(hsc_images.cpu())

            # Collect Legacy images (anchor_image when anchor_survey == 'legacy')
            if legacy_mask.any():
                legacy_images = anchor_image[legacy_mask]
                self._umap_legacy_batches.append(legacy_images.cpu())

            self._umap_batch_count += 1

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

    @torch.no_grad()
    def compute_mse(self, target_image, cond_image_samegal, cond_image_sameins, metadata=None, mask_sizes=(48, 32)):
        '''Compute reconstruction MSE on a batch of given images for one or more center crop sizes.
        Args:
            target_image (B,C,H,W)
            cond_image_samegal (B,C,H,W)
            cond_image_sameins (B,k,C,H,W)
            metadata: Optional list of metadata dicts for HSC/Legacy separation
            mask_sizes: Tuple of center crop sizes for MSE (default 48 and 32).
        Returns:
            mse_by_size: Dict mapping mask size (int) -> total MSE over that center crop
            mse_hsc: MSE for HSC samples with first mask size (if metadata provided and HSC samples exist)
            mse_legacy: MSE for Legacy samples with first mask size (if metadata provided and Legacy samples exist)
        '''
        samples = self.sample(cond_image_samegal, cond_image_sameins)
        diff = target_image - samples
        _, _, height, width = diff.shape
        device = diff.device

        mse_by_size = {}
        for mask_size in mask_sizes:
            start_x = (width - mask_size) // 2
            start_y = (height - mask_size) // 2
            diff_crop = diff[:, :, start_y:start_y+mask_size, start_x:start_x+mask_size]
            mse_by_size[mask_size] = torch.mean(diff_crop**2)

        # HSC/Legacy breakdown using first mask size (e.g. 48)
        primary_mask_size = mask_sizes[0]
        start_x = (width - primary_mask_size) // 2
        start_y = (height - primary_mask_size) // 2
        diff_primary = diff[:, :, start_y:start_y+primary_mask_size, start_x:start_x+primary_mask_size]

        mse_hsc = None
        mse_legacy = None
        if metadata is not None:
            anchor_surveys = [m['anchor_survey'] for m in metadata]
            hsc_mask = torch.tensor([s == 'hsc' for s in anchor_surveys], device=device)
            legacy_mask = torch.tensor([s == 'legacy' for s in anchor_surveys], device=device)

            if hsc_mask.any():
                mse_hsc = torch.mean(diff_primary[hsc_mask]**2)
            else:
                mse_hsc = torch.tensor(float('nan'), device=device)

            if legacy_mask.any():
                mse_legacy = torch.mean(diff_primary[legacy_mask]**2)
            else:
                mse_legacy = torch.tensor(float('nan'), device=device)

        return mse_by_size, mse_hsc, mse_legacy

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

        # Lens batch: extra row-scaled validation plot (when lens dataloader was used)
        if hasattr(self, '_val_lens_anchor_batch') and hasattr(self, '_val_lens_samegal_batch') and hasattr(self, '_val_lens_sameins_batch'):
            num_lens = min(6, len(self._val_lens_anchor_batch))
            num_samples_per_cond = 5
            num_cols = 3 + num_samples_per_cond + 1
            col_titles = ["SameGal", "SameIns (1st)", "Target"] + [f"Sample {j+1}" for j in range(num_samples_per_cond)] + ["Mean"]

            fig_lens, axes_lens = plt.subplots(
                num_lens, num_cols,
                figsize=(2 * num_cols, 2 * num_lens),
                squeeze=False,
            )
            for j, title in enumerate(col_titles):
                axes_lens[0, j].set_title(title, fontsize=10)

            def _row_scale_rgb_lens(x_chw, vmin, vmax):
                x = x_chw[:3]
                vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
                vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
                y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
                y = y.clamp(0, 1)
                return y.permute(1, 2, 0)

            for i in range(num_lens):
                samegal = self._val_lens_samegal_batch[i : i + 1].to(self.device)
                target = self._val_lens_anchor_batch[i : i + 1].to(self.device)
                sameins = self._val_lens_sameins_batch[i : i + 1].to(self.device)
                sameins_first = sameins[:, 0:1]

                samegal_repeated = samegal.repeat(num_samples_per_cond, 1, 1, 1)
                sameins_repeated = sameins.repeat(num_samples_per_cond, 1, 1, 1, 1)
                samples = self.sample(samegal_repeated, sameins_repeated)
                mean_sample = samples.mean(dim=0, keepdim=True)

                target_chw = target[0, :3]
                vmin = target_chw.amin(dim=(1, 2))
                vmax = target_chw.amax(dim=(1, 2))

                samegal_vis = _row_scale_rgb_lens(samegal[0, :3], vmin, vmax).detach().cpu().numpy()
                axes_lens[i, 0].imshow(samegal_vis)
                axes_lens[i, 0].axis("off")
                sameins_first_vis = _row_scale_rgb_lens(sameins_first[0, 0, :3], vmin, vmax).detach().cpu().numpy()
                axes_lens[i, 1].imshow(sameins_first_vis)
                axes_lens[i, 1].axis("off")
                target_vis = _row_scale_rgb_lens(target[0, :3], vmin, vmax).detach().cpu().numpy()
                axes_lens[i, 2].imshow(target_vis)
                axes_lens[i, 2].axis("off")
                for j in range(num_samples_per_cond):
                    samp_vis = _row_scale_rgb_lens(samples[j, :3], vmin, vmax).detach().cpu().numpy()
                    axes_lens[i, 3 + j].imshow(samp_vis)
                    axes_lens[i, 3 + j].axis("off")
                mean_vis = _row_scale_rgb_lens(mean_sample[0, :3], vmin, vmax).detach().cpu().numpy()
                axes_lens[i, -1].imshow(mean_vis)
                axes_lens[i, -1].axis("off")

            plt.figure(fig_lens.number)
            plt.tight_layout()
            self.logger.experiment.log({
                "val/sample_grid_row_scaled_lenses": wandb.Image(fig_lens),
                "global_step": self.global_step,
            })
            plt.close(fig_lens)

            # Lens triple comparison: Target | Sample | Legacy (same style as lenses_indeces.ipynb)
            # 3 rows (Target HSC, Sample, Legacy samegal), 5 cols (4 bands + RGB), PercentileInterval + AsinhStretch
            try:
                from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

                def _normed_imshow_lens(ax, data2d, interval_obj, stretch_obj, title=None):
                    data = np.asarray(data2d, dtype=np.float32)
                    mask = np.isfinite(data)
                    if not np.any(mask):
                        ax.axis("off")
                        return
                    vmin, vmax = interval_obj.get_limits(data[mask])
                    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_obj, clip=True)
                    ax.imshow(data, origin="lower", norm=norm, cmap="magma")
                    if title:
                        ax.set_title(title, fontsize=9, fontweight="bold")
                    ax.set_xticks([])
                    ax.set_yticks([])

                def make_rgb_lens(img, r, g, b, interval_obj, stretch_obj):
                    # img (C, H, W)
                    channels_available = img.shape[0]
                    r, g, b = [min(x, channels_available - 1) for x in [r, g, b]]
                    rgb = np.stack([img[r], img[g], img[b]], axis=-1).astype(np.float32)
                    out = np.zeros_like(rgb)
                    for k in range(3):
                        ch = rgb[..., k]
                        mask = np.isfinite(ch)
                        if not np.any(mask):
                            continue
                        vmin, vmax = interval_obj.get_limits(ch[mask])
                        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_obj, clip=True)
                        out[..., k] = norm(ch)
                    mx = np.nanmax(out)
                    if mx > 0:
                        out /= mx
                    return np.clip(out, 0, 1)

                interval_obj = PercentileInterval(99.5)
                stretch_obj = AsinhStretch()
                band_names = ["g", "r", "i", "z"]

                # First lens: target, one sample, legacy (samegal)
                samegal_0 = self._val_lens_samegal_batch[0:1].to(self.device)
                target_0 = self._val_lens_anchor_batch[0:1].to(self.device)
                sameins_0 = self._val_lens_sameins_batch[0:1].to(self.device)
                sample_0 = self.sample(samegal_0, sameins_0)  # (1, C, H, W)
                target_np = target_0[0].cpu().numpy()   # (C, H, W)
                sample_np = sample_0[0].cpu().numpy()   # (C, H, W)
                legacy_np = samegal_0[0].cpu().numpy() # (C, H, W)

                fig_triple, gs = plt.subplots(3, 5, figsize=(14, 10), constrained_layout=True)
                fig_triple.suptitle("Lens validation: Target (HSC) | Sample | Legacy (samegal)", fontsize=12, y=1.02)

                # Row 0: Target
                for c in range(4):
                    _normed_imshow_lens(gs[0, c], target_np[c], interval_obj, stretch_obj, title=f"Target {band_names[c]}")
                gs[0, 4].imshow(make_rgb_lens(target_np, 2, 1, 0, interval_obj, stretch_obj), origin="lower")
                gs[0, 4].set_title("Target RGB (irg)", fontsize=9, fontweight="bold")
                gs[0, 4].axis("off")

                # Row 1: Sample
                for c in range(4):
                    _normed_imshow_lens(gs[1, c], sample_np[c], interval_obj, stretch_obj, title=f"Sample {band_names[c]}")
                gs[1, 4].imshow(make_rgb_lens(sample_np, 2, 1, 0, interval_obj, stretch_obj), origin="lower")
                gs[1, 4].set_title("Sample RGB (irg)", fontsize=9, fontweight="bold")
                gs[1, 4].axis("off")

                # Row 2: Legacy (samegal)
                for c in range(4):
                    _normed_imshow_lens(gs[2, c], legacy_np[c], interval_obj, stretch_obj, title=f"Legacy {band_names[c]}")
                gs[2, 4].imshow(make_rgb_lens(legacy_np, 2, 1, 0, interval_obj, stretch_obj), origin="lower")
                gs[2, 4].set_title("Legacy RGB (irg)", fontsize=9, fontweight="bold")
                gs[2, 4].axis("off")

                self.logger.experiment.log({
                    "val/lens_triple_target_sample_legacy": wandb.Image(fig_triple),
                    "global_step": self.global_step,
                })
                plt.close(fig_triple)
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[Lens triple plot] Skipped (astropy or plot error): {e}")

        # Compute MSE metric (48x48 center = "MSE", 32x32 center = "MSE 32")
        if hasattr(self, '_val_mse_target_batch') and hasattr(self, '_val_mse_samegal_batch') and hasattr(self, '_val_mse_sameins_batch'):
            mse_start_time = time.time()
            mse_by_size, mse_hsc, mse_legacy = self.compute_mse(
                self._val_mse_target_batch.to(self.device),
                self._val_mse_samegal_batch.to(self.device),
                self._val_mse_sameins_batch.to(self.device),
                self._val_mse_metadata,
                mask_sizes=(48, 32),
            )
            mse_time = time.time() - mse_start_time

            # Print timing on first validation run
            if not hasattr(self, '_mse_timing_logged'):
                print(f"[MSE metric] Computation took {mse_time:.2f} seconds")
                self._mse_timing_logged = True

            self.log("val/mse", mse_by_size[48], sync_dist=True)
            self.log("val/mse_32", mse_by_size[32], sync_dist=True)
            if mse_hsc is not None:
                self.log("val/mse_hsc", mse_hsc, sync_dist=True)
            if mse_legacy is not None:
                self.log("val/mse_legacy", mse_legacy, sync_dist=True)

        # Generate UMAP visualization if we collected enough batches
        if (hasattr(self, '_umap_hsc_batches') and hasattr(self, '_umap_legacy_batches') and
            len(self._umap_hsc_batches) > 0 and len(self._umap_legacy_batches) > 0):
            try:
                # Concatenate all collected batches
                hsc_mega_batch = torch.cat(self._umap_hsc_batches, dim=0).to(self.device)
                legacy_mega_batch = torch.cat(self._umap_legacy_batches, dim=0).to(self.device)

                # Call plot_latent_space
                umap_path = self.plot_latent_space(hsc_mega_batch, legacy_mega_batch)
                print(f"[UMAP] Visualization saved to {umap_path}")
            except Exception as e:
                print(f"[UMAP] Error generating UMAP visualization: {e}")
                import traceback
                traceback.print_exc()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    @torch.no_grad()
    def plot_latent_space(self, hsc_batch, legacy_batch):
        """
        Generate UMAP visualizations for both encoders, showing token-specific and combined embeddings.

        Creates a 5x2 grid:
        - Rows 0-3: UMAP for each token position (0, 1, 2, 3)
        - Row 4: Combined UMAP (all tokens flattened)
        - Columns: Encoder 1 (Same Galaxy) and Encoder 2 (Same Instrument)

        Args:
            hsc_batch: HSC images (B, C, H, W)
            legacy_batch: Legacy images (B, C, H, W)
        """
        import matplotlib.pyplot as plt

        # Encode images with both encoders
        hsc_embeddings_1 = self.encoder_1(hsc_batch)  # (B, seq_len, embed_dim)
        legacy_embeddings_1 = self.encoder_1(legacy_batch)
        hsc_embeddings_2 = self.encoder_2(hsc_batch)
        legacy_embeddings_2 = self.encoder_2(legacy_batch)

        num_hsc = hsc_embeddings_1.shape[0]
        seq_len = hsc_embeddings_1.shape[1]

        # Prepare embeddings for each encoder (keep token structure)
        all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0)  # (B_total, seq_len, embed_dim)
        all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0)  # (B_total, seq_len, embed_dim)

        # Create figures directory
        figures_dir = Path('/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)

        # UMAP parameters
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 2,
            'metric': 'euclidean',
            'random_state': 42,
        }

        # Create figure with 5 rows (4 token positions + 1 combined) and 2 columns (encoder_1, encoder_2)
        fig, axes = plt.subplots(5, 2, figsize=(16, 24))

        # Process each token position (0, 1, 2, 3)
        for token_idx in range(seq_len):
            # Extract embeddings for this token position
            token_embeddings_1 = all_embeddings_1[:, token_idx, :].cpu().numpy()  # (B_total, embed_dim)
            token_embeddings_2 = all_embeddings_2[:, token_idx, :].cpu().numpy()

            # Split into HSC and Legacy
            hsc_token_1 = token_embeddings_1[:num_hsc]
            legacy_token_1 = token_embeddings_1[num_hsc:]
            hsc_token_2 = token_embeddings_2[:num_hsc]
            legacy_token_2 = token_embeddings_2[num_hsc:]

            # ===== Encoder 1 UMAP for this token =====
            reducer_1 = umap.UMAP(**umap_params)
            embedding_1_umap = reducer_1.fit_transform(token_embeddings_1)

            hsc_embedding_1_umap = embedding_1_umap[:num_hsc]
            legacy_embedding_1_umap = embedding_1_umap[num_hsc:]

            # ===== Encoder 2 UMAP for this token =====
            reducer_2 = umap.UMAP(**umap_params)
            embedding_2_umap = reducer_2.fit_transform(token_embeddings_2)

            hsc_embedding_2_umap = embedding_2_umap[:num_hsc]
            legacy_embedding_2_umap = embedding_2_umap[num_hsc:]

            # Plot Encoder 1 (left column)
            ax1 = axes[token_idx, 0]
            ax1.scatter(hsc_embedding_1_umap[:, 0], hsc_embedding_1_umap[:, 1],
                        s=5, label='HSC', alpha=0.6, c='blue')
            ax1.scatter(legacy_embedding_1_umap[:, 0], legacy_embedding_1_umap[:, 1],
                        s=5, label='Legacy', alpha=0.6, c='orange')
            ax1.set_title(f'Encoder 1 (Same Galaxy) - Token {token_idx}', fontsize=10)
            ax1.set_xlabel('UMAP Component 1')
            ax1.set_ylabel('UMAP Component 2')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot Encoder 2 (right column)
            ax2 = axes[token_idx, 1]
            ax2.scatter(hsc_embedding_2_umap[:, 0], hsc_embedding_2_umap[:, 1],
                        s=5, label='HSC', alpha=0.6, c='blue')
            ax2.scatter(legacy_embedding_2_umap[:, 0], legacy_embedding_2_umap[:, 1],
                        s=5, label='Legacy', alpha=0.6, c='orange')
            ax2.set_title(f'Encoder 2 (Same Instrument) - Token {token_idx}', fontsize=10)
            ax2.set_xlabel('UMAP Component 1')
            ax2.set_ylabel('UMAP Component 2')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # ===== Combined UMAP (Row 4) =====
        # Flatten all tokens for combined visualization
        all_embeddings_1_flat = all_embeddings_1.flatten(start_dim=1).cpu().numpy()  # (B_total, seq_len * embed_dim)
        all_embeddings_2_flat = all_embeddings_2.flatten(start_dim=1).cpu().numpy()

        # Encoder 1 combined UMAP
        reducer_1_combined = umap.UMAP(**umap_params)
        embedding_1_combined_umap = reducer_1_combined.fit_transform(all_embeddings_1_flat)
        hsc_embedding_1_combined = embedding_1_combined_umap[:num_hsc]
        legacy_embedding_1_combined = embedding_1_combined_umap[num_hsc:]

        # Encoder 2 combined UMAP
        reducer_2_combined = umap.UMAP(**umap_params)
        embedding_2_combined_umap = reducer_2_combined.fit_transform(all_embeddings_2_flat)
        hsc_embedding_2_combined = embedding_2_combined_umap[:num_hsc]
        legacy_embedding_2_combined = embedding_2_combined_umap[num_hsc:]

        # Plot combined Encoder 1 (left column, row 4)
        ax1_combined = axes[4, 0]
        ax1_combined.scatter(hsc_embedding_1_combined[:, 0], hsc_embedding_1_combined[:, 1],
                             s=5, label='HSC', alpha=0.6, c='blue')
        ax1_combined.scatter(legacy_embedding_1_combined[:, 0], legacy_embedding_1_combined[:, 1],
                             s=5, label='Legacy', alpha=0.6, c='orange')
        ax1_combined.set_title('Encoder 1 (Same Galaxy) - Combined (All Tokens)', fontsize=10)
        ax1_combined.set_xlabel('UMAP Component 1')
        ax1_combined.set_ylabel('UMAP Component 2')
        ax1_combined.legend()
        ax1_combined.grid(True, alpha=0.3)

        # Plot combined Encoder 2 (right column, row 4)
        ax2_combined = axes[4, 1]
        ax2_combined.scatter(hsc_embedding_2_combined[:, 0], hsc_embedding_2_combined[:, 1],
                             s=5, label='HSC', alpha=0.6, c='blue')
        ax2_combined.scatter(legacy_embedding_2_combined[:, 0], legacy_embedding_2_combined[:, 1],
                             s=5, label='Legacy', alpha=0.6, c='orange')
        ax2_combined.set_title('Encoder 2 (Same Instrument) - Combined (All Tokens)', fontsize=10)
        ax2_combined.set_xlabel('UMAP Component 1')
        ax2_combined.set_ylabel('UMAP Component 2')
        ax2_combined.legend()
        ax2_combined.grid(True, alpha=0.3)

        # Add column labels at the top
        col_labels = ['Encoder 1 (Physics)', 'Encoder 2 (Instrument)']
        for col_idx, label in enumerate(col_labels):
            axes[0, col_idx].text(0.5, 1.15, label, transform=axes[0, col_idx].transAxes,
                                  ha='center', va='bottom', fontsize=12, weight='bold')

        plt.suptitle('UMAP Visualization by Token Position', fontsize=14, y=0.995)
        plt.tight_layout()

        # Add horizontal line separator between token-specific and combined plots
        # Draw line across both columns between row 3 and row 4
        # Get the position of axes[3, 0] (bottom of token plots) and axes[4, 0] (top of combined plot)
        ax_bottom = axes[3, 0]
        ax_top = axes[4, 0]

        # Get positions in figure coordinates (after tight_layout)
        bbox_bottom = ax_bottom.get_position()
        bbox_top = ax_top.get_position()

        # Calculate y position for the separator line (midpoint between bottom of row 3 and top of row 4)
        y_separator = (bbox_bottom.y0 + bbox_top.y1) / 2

        # Draw horizontal line across full width
        line = plt.Line2D(
            [0, 1],  # Full width in figure coordinates
            [y_separator, y_separator],
            transform=fig.transFigure,
            color='black',
            linewidth=2,
            linestyle='--',
            alpha=0.5,
            zorder=10
        )
        fig.lines.append(line)

        # Save figure
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / f'umap_latent_space_step{self.global_step}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Log to wandb if logger is available
        if self.logger and hasattr(self.logger, 'experiment'):
            import wandb
            self.logger.experiment.log({
                "latent_space/umap_grid": wandb.Image(str(save_path)),
                "global_step": self.global_step,
            })

        return save_path


def is_h100_gpu() -> bool:
    """
    Detect if any available GPU is an H100.

    Returns:
        True if at least one H100 GPU is detected, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i).lower()
        if 'h100' in device_name:
            return True

    return False


if __name__ == "__main__":
    # Set up snapshot + tee logging before anything else in main runs

    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset
    from data import HSCLegacyTripletDataset, BalancedAnchorBatchSampler, custom_collate_fn, HSCLegacyTripletDatasetZoom, HSCLegacyTripletDatasetMask, HSCLegacyTripletDatasetZoomLenses

    # Seed everything for reproducibility
    seed = 42  # Set to None for non-deterministic behavior
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    lambda_generative = 1
    lambda_geometric = 7.5e-4 # 0.075, 0.3, 0.0075=7.5e-3, 0.00075=7.5e-4 -- >
    # lambda_geometric = 0

    # Detect H100 for custom logic
    is_h100 = is_h100_gpu()

    # Set default values
    batch_size = 32
    precision_setting = "16-mixed"
    lr = 1e-4
    num_steps = 300_000
    # image_size: 48 (default 48x48 behavior), 96 (96x96), or 64 (to be added)
    image_size = 48
    mask_center = False

    # Override with H100-specific settings if detected
    if is_h100:
        # print("H100 GPU detected - applying H100-specific settings")
        # batch_size = 64  # in the future try changing to 256
        # precision_setting = 'bf16-mixed'
        # lr = 1e-4
        # print(f"  Using batch_size={batch_size}, precision={precision_setting}")
        print("H100 GPU detected - applying H100-specific settings")
        multiplier = 1
        batch_size = 64 * multiplier  # in the future try changing to 256
        precision_setting = 'bf16-mixed'
        lr = 1e-4 * multiplier
        num_steps = 300_000 / multiplier

        print(f"  Using batch_size={batch_size}, precision={precision_setting}, num_steps={num_steps}")

    wandb_project = "galaxy-flow-matching"  # Change this to your desired wandb project name

    lense_dataset = None
    if image_size == 48:
        print("Not masking center and using 48x48 images")
        train_dataset = HSCLegacyTripletDatasetZoom(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
            idx_list=list(range(95_000)),
            is96=False,
        )
        val_dataset = HSCLegacyTripletDatasetZoom(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
            idx_list=list(range(95_000, 100_000)),
            deterministic_anchor_survey=True,  # Make validation batches consistent
            is96=False,
        )
        lense_indices = [3199, 3298, 4368, 4556, 8357, 9503, 19076, 20869, 26247, 40506, 51839, 53037, 60565, 60980, 64245, 72326, 74053, 77857, 99695]
        lense_dataset = HSCLegacyTripletDatasetZoomLenses(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
            lense_indices=lense_indices,
            is96=False,
        )


    elif image_size == 64 and not mask_center:
        print("Not masking center and using 64x64 images")
        train_dataset = HSCLegacyTripletDatasetZoom(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_64x64_half.h5',
            idx_list=list(range(50_000)),
            is96=False,
        )
        val_dataset = HSCLegacyTripletDatasetZoom(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_64x64_half.h5',
            idx_list=list(range(50_000, 51_400)),
            deterministic_anchor_survey=True,  # Make validation batches consistent
            is96=False,
        )


    elif image_size == 64 and mask_center:
        print("Masking center and using 64x64 images")
        train_dataset = HSCLegacyTripletDatasetMask(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_64x64_half.h5',
            idx_list=list(range(50_000)),
            is96=False,
        )
        val_dataset = HSCLegacyTripletDatasetMask(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_64x64_half.h5',
            idx_list=list(range(50_000, 51_400)),
            deterministic_anchor_survey=True,  # Make validation batches consistent
            is96=False,
        )

    elif image_size == 96 and mask_center:
        # print("Masking center and using 96x96 images")
        # train_dataset = HSCLegacyTripletDatasetMask(
        #     hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_96x96_half.h5',
        #     idx_list=list(range(50_000)),
        #     is96=True,
        # )
        # val_dataset = HSCLegacyTripletDatasetMask(
        #     hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_96x96_half.h5',
        #     idx_list=list(range(50_000, 51_400)),
        #     deterministic_anchor_survey=True,  # Make validation batches consistent
        #     is96=True,
        # )

        print("Masking center and using 96x96 images")
        train_dataset = HSCLegacyTripletDatasetMask(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_96x96_half.h5',
            idx_list=list(range(30_000)),
            is96=True,
        )
        val_dataset = HSCLegacyTripletDatasetMask(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_96x96_half.h5',
            idx_list=list(range(30_000, 32_140)),
            deterministic_anchor_survey=True,  # Make validation batches consistent
            is96=True,
        )

    elif image_size == 96:
        print("Not masking center and using 96x96 images")
        train_dataset = HSCLegacyTripletDatasetZoom(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_96x96_half.h5', #10280
            idx_list=list(range(50_000)),
            is96=True,  # Use 96x96 normalization values
        )
        val_dataset = HSCLegacyTripletDatasetZoom(
            hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_96x96_half.h5',
            idx_list=list(range(50_000, 51_400)),
            deterministic_anchor_survey=True,  # Make validation batches consistent
            is96=True,  # Use 96x96 normalization values
        )

    else:
        raise NotImplementedError(f"image_size={image_size} not yet implemented; use 48, 96, or add support for 64")

    # train_dataset = HSCLegacyTripletDatasetZoom(
    #     hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_48x48_laptop.h5',
    #     idx_list=list(range(5000)),
    # )
    # val_dataset = HSCLegacyTripletDatasetZoom(
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
        num_workers=8,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Use same collate function
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        # batch_sampler=val_batch_sampler,  # important: use batch_sampler, not batch_size/shuffle
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Use same collate function
    )

    if lense_dataset is not None:
        lense_loader = DataLoader(
            lense_dataset,
            batch_size=19,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )
        val_loaders = [val_loader, lense_loader]
    else:
        val_loaders = val_loader

    concat_conditioning = False
    model = ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        image_size=image_size,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=16,
        pretrained_encoder=False,
        concat_conditioning=concat_conditioning,
        lr=lr,
        num_sample_images=10,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=lambda_generative,
        lambda_geometric=lambda_geometric,
        mask_center=mask_center,
    )

    if concat_conditioning:
        name="conditional-unet2d-concatenated"
    else:
        name="encoder-zoom-64x64-zdim16-masked-50kimages"
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=name,
        log_model=False,
    )

    # Log batch_size and precision_setting to wandb config
    if wandb_logger.experiment:
        wandb_logger.experiment.config.update({
            "batch_size": batch_size,
            "precision": precision_setting,
            "is_h100": is_h100,
        })

    # Checkpoint callback for best model (based on validation loss).
    # With multiple val dataloaders (e.g. main + lens), Lightning suffixes keys: val/loss -> val/loss/dataloader_idx_0
    val_loss_monitor = "val/loss/dataloader_idx_0" if lense_dataset is not None else "val/loss"
    best_checkpoint = ModelCheckpoint(
        monitor=val_loss_monitor,
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )

    # Checkpoint callback for periodic saves (every 1000 steps, replaces previous)
    periodic_checkpoint = ModelCheckpoint(
        every_n_train_steps=1000,
        save_top_k=1,  # Only keep the latest one (replaces previous)
        filename="latest-step={step}",
        save_last=False,  # We're using every_n_train_steps instead
    )

    n_devices = 4
    trainer = pl.Trainer(
        max_steps=num_steps/n_devices,
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        strategy="ddp_find_unused_parameters_true",  # Required when lambda_generative=0 (UNet/encoder_2 unused)
        log_every_n_steps=10,
        precision=precision_setting,  # Use H100-specific precision if detected
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint],
    )

    trainer.fit(model, train_loader, val_loaders)
