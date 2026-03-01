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

#TODO: Implement the multiple instrument pairs with attn rather than concatenation


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
        script_dir / "double_train_fm_no-attn.py",
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
    ResNet18 encoder from timm that produces a single embedding vector.
    Used for class embedding projection instead of cross-attention.

    Args:
        output_dim: Output dimension of the embedding. If None, uses the default ResNet18
            output dimension (512). If specified, adds a linear projection layer.
    """

    def __init__(
        self,
        in_channels: int = 4,
        cross_attention_dim: int = 256,  # Not used, kept for compatibility
        pretrained: bool = False,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            num_classes=0,  # This removes the final FC layer and returns the 512 pool output
            in_chans=in_channels
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

        # Add linear projection if output_dim is specified
        self.output_dim = output_dim if output_dim is not None else 512
        if output_dim is not None and output_dim != 512:
            self.projection = nn.Linear(512, output_dim)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns shape (B, 512) or (B, output_dim) if projection is used
        features = self.backbone(x)  # (B, 512)
        if self.projection is not None:
            features = self.projection(features)  # (B, output_dim)
        return features


class ConditionalFlowMatchingModule(pl.LightningModule):
    """
    Conditional Flow Matching model using optimal transport conditional paths.

    Conditions on two sets of images:
    1. Same galaxy conditioning (e.g., Legacy Survey)
    2. Same instrument conditioning (e.g., multiple HSC images)

    The interpolation is: x_t = (1 - t) * x_0 + t * x_1
    where x_0 ~ N(0, I) (noise), x_1 ~ target data.

    The target velocity is: v(x_t, t, c) = x_1 - x_0
    where c is the conditioning image embedding.

    Uses class embedding projection instead of cross-attention.
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
        cross_attention_dim: int = 256, # Not used in no-attn mode, kept for compatibility
        pretrained_encoder: bool = False, # load pretrained imagenet weights
        concat_conditioning: bool = False, # if true -> no encoder, conditioning is concatenated as extra channels to the input
        encoder_output_dim: int = 512, # Output dimension of each encoder
        num_same_instrument: int = 5, # Number of same-instrument images (k). Combined dimension will be encoder_output_dim * (1 + k)
        # Optimization params
        lr: float = 1e-4,
        num_sample_images: int = 8, # number of exmaples cached for first validation batch for W&B
        num_mse_images: int = 64, # number of examples cached for MSE tracking
        num_integration_steps: int = 500,
        lambda_generative: float = 1.0, # weight for generative loss
        lambda_geometric: float = 0.3, # weight for geometric loss
        num_umap_batches: int = 8, # number of validation batches to collect for UMAP visualization
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
        self.num_same_instrument = num_same_instrument

        block_out_channels = tuple(model_channels * m for m in channel_mult)

        if concat_conditioning:
            raise ValueError("Concat conditioning is not supported for the double encoder case")
        else:
            self.encoder_1 = ResNetEncoder(
                in_channels=cond_channels,
                cross_attention_dim=cross_attention_dim,
                pretrained=pretrained_encoder,
                output_dim=encoder_output_dim,
            )

            self.encoder_2 = ResNetEncoder(
                in_channels=cond_channels,
                cross_attention_dim=cross_attention_dim,
                pretrained=pretrained_encoder,
                output_dim=encoder_output_dim,
            )

            self.velocity_model = UNet2DConditionModel(
                sample_size=image_size,
                in_channels=in_channels,
                out_channels=in_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                ),
                mid_block_type='UNetMidBlock2D',
                up_block_types=(
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                class_embed_type="projection",
                projection_class_embeddings_input_dim=encoder_output_dim * (1 + num_same_instrument),  # encoder_1 (1) + encoder_2 (k) concatenated
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

        Returns:
            Predicted velocity (B, C, H, W)
        """
        timesteps = t * 1000

        cond_gal_embedding = self.encoder_1(cond_image_samegal)  # (B, encoder_output_dim)

        B, k, C, H, W = cond_image_sameins.shape

        cond_image_sameins_flat = cond_image_sameins.flatten(0, 1)  # (B*k, C, H, W)
        cond_ins_embedding_flat = self.encoder_2(cond_image_sameins_flat)  # (B*k, encoder_output_dim)

        cond_ins_embedding = cond_ins_embedding_flat.unflatten(0, (B, k))  # (B, k, encoder_output_dim)

        # Concatenate all k same-instrument embeddings along feature dimension
        cond_ins_embedding = cond_ins_embedding.flatten(1)  # (B, k * encoder_output_dim)

        # Concatenate both embeddings to get (B, encoder_output_dim + k * encoder_output_dim)
        cond_embedding = torch.cat([cond_gal_embedding, cond_ins_embedding], dim=1)  # (B, encoder_output_dim * (1 + k))

        # Create dummy encoder_hidden_states since it's required but not used without cross-attention
        batch_size = x_t.shape[0]
        dummy_encoder_hidden_states = torch.zeros(
            batch_size, 1, 1280, device=x_t.device, dtype=x_t.dtype
        )

        return self.velocity_model(
            x_t,
            timesteps,
            encoder_hidden_states=dummy_encoder_hidden_states,
            class_labels=cond_embedding,  # Combined embeddings from both encoders
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
        generative_loss = per_example_loss.mean()

        # Store separate losses for logging (detached to be explicit they're not used for gradients)
        self._loss_generative_total = generative_loss.detach()
        self._loss_hsc = loss_hsc.detach()
        self._loss_legacy = loss_legacy.detach()


        ### Geometric loss
        if self.lambda_geometric > 0:
            embeds_target = self.encoder_1(x_1).contiguous()  # (B, encoder_output_dim)
            embeds_samegal = self.encoder_1(cond_image_samegal).contiguous()  # (B, encoder_output_dim)

            # Compute geometric loss (scalar for the entire batch)
            total_geom_loss = self.geom_loss_fn(embeds_target, embeds_samegal)

            # Store geometric loss for logging
            self._loss_geom_total = total_geom_loss.detach()
        else:
            # Skip computation when lambda_geometric is 0
            total_geom_loss = torch.tensor(0.0, device=generative_loss.device, dtype=generative_loss.dtype)
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
        print(f"{'='*60}\n")

        # Explicitly log important hyperparameters to wandb
        if self.logger and hasattr(self.logger, 'experiment'):
            if hasattr(self.logger.experiment, 'config'):
                self.logger.experiment.config.update({
                    "encoder_output_dim": self.hparams.encoder_output_dim,
                    "lambda_generative": self.lambda_generative,
                    "lambda_geometric": self.lambda_geometric,
                })

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

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
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
    def compute_mse(self, target_image, cond_image_samegal, cond_image_sameins, metadata=None):
        '''Compute reconstruction MSE on a batch of given images
        Args:
            target_image (B,C,H,W)
            cond_image_samegal (B,C,H,W)
            cond_image_sameins (B,k,C,H,W)
            metadata: Optional list of metadata dicts for HSC/Legacy separation
        Returns:
            mse_total: Total MSE across all samples
            mse_hsc: MSE for HSC samples (if metadata provided and HSC samples exist)
            mse_legacy: MSE for Legacy samples (if metadata provided and Legacy samples exist)
        '''
        samples = self.sample(cond_image_samegal, cond_image_sameins)

        diff = target_image - samples
        mse_total = torch.mean(diff**2)

        mse_hsc = None
        mse_legacy = None

        if metadata is not None:
            # Extract anchor_survey from metadata and compute separate MSEs
            anchor_surveys = [m['anchor_survey'] for m in metadata]
            device = diff.device
            hsc_mask = torch.tensor([s == 'hsc' for s in anchor_surveys], device=device)
            legacy_mask = torch.tensor([s == 'legacy' for s in anchor_surveys], device=device)

            # Compute MSE for HSC samples
            if hsc_mask.any():
                diff_hsc = diff[hsc_mask]
                mse_hsc = torch.mean(diff_hsc**2)
            else:
                mse_hsc = torch.tensor(float('nan'), device=device)

            # Compute MSE for Legacy samples
            if legacy_mask.any():
                diff_legacy = diff[legacy_mask]
                mse_legacy = torch.mean(diff_legacy**2)
            else:
                mse_legacy = torch.tensor(float('nan'), device=device)

        return mse_total, mse_hsc, mse_legacy

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

        # Compute MSE metric
        if hasattr(self, '_val_mse_target_batch') and hasattr(self, '_val_mse_samegal_batch') and hasattr(self, '_val_mse_sameins_batch'):
            mse_start_time = time.time()
            mse_total, mse_hsc, mse_legacy = self.compute_mse(
                self._val_mse_target_batch.to(self.device),
                self._val_mse_samegal_batch.to(self.device),
                self._val_mse_sameins_batch.to(self.device),
                self._val_mse_metadata
            )
            mse_time = time.time() - mse_start_time

            # Print timing on first validation run
            if not hasattr(self, '_mse_timing_logged'):
                print(f"[MSE metric] Computation took {mse_time:.2f} seconds")
                self._mse_timing_logged = True

            self.log("val/mse", mse_total, sync_dist=True)
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
        Generate UMAP visualizations for both encoders.

        Creates a 1x2 grid:
        - Left: Encoder 1 (Same Galaxy) with HSC and Legacy points superimposed
        - Right: Encoder 2 (Same Instrument) with HSC and Legacy points superimposed

        Args:
            hsc_batch: HSC images (B, C, H, W)
            legacy_batch: Legacy images (B, C, H, W)
        """
        import matplotlib.pyplot as plt

        # Encode images with both encoders
        hsc_embeddings_1 = self.encoder_1(hsc_batch)  # (B, 512)
        legacy_embeddings_1 = self.encoder_1(legacy_batch)  # (B, 512)
        hsc_embeddings_2 = self.encoder_2(hsc_batch)  # (B, 512)
        legacy_embeddings_2 = self.encoder_2(legacy_batch)  # (B, 512)

        num_hsc = hsc_embeddings_1.shape[0]

        # Prepare embeddings for each encoder
        all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0)  # (B_total, 512)
        all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0)  # (B_total, 512)

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

        # Create figure with 1 row and 2 columns (encoder_1, encoder_2)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # ===== Encoder 1 UMAP =====
        reducer_1 = umap.UMAP(**umap_params)
        embedding_1_umap = reducer_1.fit_transform(all_embeddings_1.cpu().numpy())

        hsc_embedding_1_umap = embedding_1_umap[:num_hsc]
        legacy_embedding_1_umap = embedding_1_umap[num_hsc:]

        # Plot Encoder 1 with both HSC and Legacy points
        ax1 = axes[0]
        ax1.scatter(hsc_embedding_1_umap[:, 0], hsc_embedding_1_umap[:, 1],
                    s=5, label='HSC', alpha=0.6, c='blue')
        ax1.scatter(legacy_embedding_1_umap[:, 0], legacy_embedding_1_umap[:, 1],
                    s=5, label='Legacy', alpha=0.6, c='orange')
        ax1.set_title('Encoder 1 (Same Galaxy)', fontsize=12)
        ax1.set_xlabel('UMAP Component 1')
        ax1.set_ylabel('UMAP Component 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ===== Encoder 2 UMAP =====
        reducer_2 = umap.UMAP(**umap_params)
        embedding_2_umap = reducer_2.fit_transform(all_embeddings_2.cpu().numpy())

        hsc_embedding_2_umap = embedding_2_umap[:num_hsc]
        legacy_embedding_2_umap = embedding_2_umap[num_hsc:]

        # Plot Encoder 2 with both HSC and Legacy points
        ax2 = axes[1]
        ax2.scatter(hsc_embedding_2_umap[:, 0], hsc_embedding_2_umap[:, 1],
                    s=5, label='HSC', alpha=0.6, c='blue')
        ax2.scatter(legacy_embedding_2_umap[:, 0], legacy_embedding_2_umap[:, 1],
                    s=5, label='Legacy', alpha=0.6, c='orange')
        ax2.set_title('Encoder 2 (Same Instrument)', fontsize=12)
        ax2.set_xlabel('UMAP Component 1')
        ax2.set_ylabel('UMAP Component 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('UMAP Visualization of Encoder Embeddings', fontsize=14, y=0.995)
        plt.tight_layout()

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

if __name__ == "__main__":
    # Set up snapshot + tee logging before anything else in main runs
    setup_run_snapshot()

    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset
    from data import HSCLegacyTripletDataset, BalancedAnchorBatchSampler, custom_collate_fn, HSCLegacyTripletDatasetZoom

    # Seed everything for reproducibility
    seed = 42  # Set to None for non-deterministic behavior
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    lambda_generative = 1
    lambda_geometric = 7.5e-4  # 0.075, 0

    batch_size = 64
    wandb_project = "galaxy-flow-matching"  # Change this to your desired wandb project name

    train_dataset = HSCLegacyTripletDatasetZoom(
        hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
        idx_list=list(range(95_000)),
    )
    val_dataset = HSCLegacyTripletDatasetZoom(
        hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
        idx_list=list(range(95_000, 100_000)),
        deterministic_anchor_survey=True,  # Make validation batches consistent
    )

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
        cross_attention_dim=8,  # Not used in no-attn mode, kept for compatibility
        pretrained_encoder=False,
        concat_conditioning=concat_conditioning,
        encoder_output_dim=64,  # Output dimension of each encoder
        num_same_instrument=5,  # Number of same-instrument images (k). Combined dimension will be encoder_output_dim * (1 + k) = 64 * 6 = 384
        lr=1e-4,
        num_sample_images=10,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=lambda_generative,
        lambda_geometric=lambda_geometric,
        num_umap_batches=16,  # Increase this to collect more batches for UMAP visualization
    )

    if concat_conditioning:
        name="conditional-unet2d-concatenated"
    else:
        name="double-encoder-resnet18-triplet-no-attn-zoom_zdim64"
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=name,
        log_model=False,
    )

    # Checkpoint callback for best model (based on validation loss)
    best_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}-val_loss={val/loss:.4f}",
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
        max_steps=300_000/n_devices,
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        log_every_n_steps=10,
        precision="16-mixed",
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint],
    )

    trainer.fit(model, train_loader, val_loader)
