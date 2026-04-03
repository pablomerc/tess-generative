"""
Hierarchical attention variant of the neighbors flow matching model.

Uses ConfigurableEncoder (multi-scale spatial tokens + AdaGN global embedding)
for the galaxy encoder (encoder_1), while keeping a simple ResNetEncoder for
the neighbor instrument encoder (encoder_2). Neighbor tokens are concatenated
at each cross-attention level.

Run via the standalone training script:
    python neighbours_train_hier.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import time
import sys
import numpy as np
from pathlib import Path
from typing import Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import geomloss
import umap

from galaxy_images.galaxy_model.hierarchical_attention.train_experiments import (
    RotaryEmbedding2D,
    ConfigurableEncoder,
    ConditionedUNet,
    build_unet_and_level_map,
    EXPERIMENTS,
)
from galaxy_images.galaxy_model.double_train_fm_neighbors import (
    ResNetEncoder,
    is_h100_gpu,
)


# =============================================================================
# RoPE processor that handles extra (neighbor) tokens in K
# =============================================================================

class RoPEWithNeighborTokensProcessor:
    """
    Attn2 processor applying 2D RoPE to Q and the spatial portion of K,
    while leaving extra tokens (neighbor embeddings) unpositioned in K.

    Q has exactly h*w tokens (UNet spatial features).
    K may have h*w + N_extra tokens (galaxy spatial + neighbor tokens).
    RoPE is applied to Q and the first h*w tokens of K only.
    """

    def __init__(self, head_dim: int, resolution: int):
        self.rope = RotaryEmbedding2D(dim=head_dim)
        self.resolution = resolution
        self.n_spatial = resolution * resolution

    def __call__(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.reshape(B, C, H * W).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        h = w = self.resolution
        query = self.rope(query, h, w)

        key_spatial = key[:, :, :self.n_spatial, :]
        key_extra = key[:, :, self.n_spatial:, :]
        key_spatial = self.rope(key_spatial, h, w)
        key = torch.cat([key_spatial, key_extra], dim=2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=attn.dropout, training=attn.training)

        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(batch_size, C, H, W)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def _setup_rope_with_neighbors(conditioned_unet, level_grid_sizes, level_rope):
    """Install RoPEWithNeighborTokensProcessor on cross-attention blocks."""
    from diffusers.models.attention_processor import Attention

    block_info = {}
    for block_name, level_idx in conditioned_unet.level_map.items():
        if level_idx is not None and level_idx in level_grid_sizes:
            block_info[block_name] = (
                level_grid_sizes[level_idx],
                level_rope.get(level_idx, True),
            )

    attn_procs = {}
    for name, proc in conditioned_unet.unet.attn_processors.items():
        if "attn2" not in name:
            attn_procs[name] = proc
            continue

        block_name = conditioned_unet._attn_to_block(name)
        if block_name in block_info:
            resolution, use_rope = block_info[block_name]
            if use_rope:
                layer = conditioned_unet._get_layer(name)
                hd = layer.inner_dim // layer.heads
                attn_procs[name] = RoPEWithNeighborTokensProcessor(
                    head_dim=hd, resolution=resolution,
                )
                print(f"  RoPE (neighbor-aware) on {name} "
                      f"(res={resolution}, head_dim={hd})")
            else:
                attn_procs[name] = proc
                print(f"  Standard xattn on {name} (no RoPE, grid mismatch)")
        else:
            attn_procs[name] = proc

    conditioned_unet.unet.set_attn_processor(attn_procs)


# =============================================================================
# Hierarchical Flow Matching Module with neighbor conditioning
# =============================================================================

class HierarchicalFlowMatchingModule(pl.LightningModule):
    """
    Conditional Flow Matching with hierarchical attention for galaxy conditioning
    and ResNet encoder for neighbor (same-instrument) conditioning.

    encoder_1 (galaxy):     ConfigurableEncoder → multi-scale spatial tokens + global
    encoder_2 (neighbors):  ResNetEncoder → flat tokens (concatenated at each level)
    UNet:                   ConditionedUNet with level routing + AdaGN

    The interpolation is: x_t = (1 - t) * x_0 + t * x_1
    where x_0 ~ N(0, I) (noise), x_1 ~ target data.
    The target velocity is: v(x_t, t, c) = x_1 - x_0.
    """

    def __init__(
        self,
        experiment_config: dict,
        in_channels: int = 4,
        cond_channels: int = 4,
        image_size: int = 48,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        layers_per_block: int = 2,
        attention_head_dim: int = 64,
        instrument_zdim: int = None,
        pretrained_encoder: bool = False,
        lr: float = 1e-4,
        num_sample_images: int = 8,
        num_mse_images: int = 64,
        num_integration_steps: int = 250,
        lambda_generative: float = 1.0,
        lambda_geometric: float = 0.0,
        num_umap_batches: int = 8,
        mask_center: bool = False,
        figures_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(experiment_config, str):
            experiment_config = EXPERIMENTS[experiment_config]
        cfg = experiment_config
        token_dim = cfg["token_dim"]

        self.lr = lr
        self.num_sample_images = num_sample_images
        self.num_mse_images = num_mse_images
        self.num_integration_steps = num_integration_steps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.image_size = image_size
        self.lambda_generative = lambda_generative
        self.lambda_geometric = lambda_geometric
        self.num_umap_batches = num_umap_batches
        self.mask_center = mask_center
        self.figures_dir = (
            Path(figures_dir) if figures_dir
            else (Path(__file__).resolve().parent / "figures")
        )
        self.instrument_zdim = instrument_zdim if instrument_zdim is not None else token_dim
        self.is_h100 = is_h100_gpu()

        # ---- Encoder 1: Hierarchical (same-galaxy conditioning) ----
        self.encoder_1 = ConfigurableEncoder(
            in_channels=cond_channels,
            spatial_indices=cfg["spatial_indices"],
            reductions=cfg.get("reductions", {}),
            token_dim=token_dim,
            global_dim=cfg["global_dim"],
            pretrained=pretrained_encoder,
        )

        # ---- Encoder 2: ResNet (same-instrument neighbors) ----
        self.encoder_2 = ResNetEncoder(
            in_channels=cond_channels,
            cross_attention_dim=self.instrument_zdim,
            pretrained=pretrained_encoder,
            mean_pool=False,
        )

        if self.instrument_zdim != token_dim:
            self.ins_proj = nn.Linear(self.instrument_zdim, token_dim)
        else:
            self.ins_proj = None

        # ---- UNet with level routing ----
        unet, level_map, effective_hd = build_unet_and_level_map(
            config=cfg,
            in_channels=in_channels,
            image_size=image_size,
            model_channels=model_channels,
            channel_mult=channel_mult,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
        )
        self.conditioned_unet = ConditionedUNet(unet=unet, level_map=level_map)

        # ---- Probe encoder + setup RoPE with neighbor support ----
        print(f"\nEncoder 1 (galaxy) configuration:")
        with torch.no_grad():
            dummy = torch.zeros(1, cond_channels, image_size, image_size)
            spatial_levels, global_vec, rope_flags = self.encoder_1(dummy)

            grid_sizes = {}
            rope_enabled = {}
            total_tokens = 0
            total_spatial_values = 0
            for i, (tokens, gh, gw) in enumerate(spatial_levels):
                n_tok = tokens.shape[1]
                total_tokens += n_tok
                total_spatial_values += n_tok * token_dim
                grid_sizes[i] = gh
                rope_enabled[i] = rope_flags[i]
                print(f"  Level {i}: {gh}x{gw} = {n_tok} tokens x {token_dim}d"
                      f" = {n_tok * token_dim} values"
                      f" | RoPE={'yes' if rope_flags[i] else 'no'}")

            total_cond = total_spatial_values + cfg["global_dim"]
            input_vals = cond_channels * image_size * image_size
            ratio = input_vals / total_cond
            tag = (f"{ratio:.1f}x compression" if ratio > 1
                   else f"{1/ratio:.1f}x expansion")
            print(f"  Global: {cfg['global_dim']}d")
            print(f"  Total: {total_tokens} tokens + global = "
                  f"{total_cond} values ({tag} from {input_vals})")
            print(f"  Attention head dim: {effective_hd}")

            # Probe encoder_2 output shape
            enc2_out = self.encoder_2(dummy)
            enc2_seq = enc2_out.shape[1]
            print(f"\nEncoder 2 (neighbors): {enc2_seq} tokens x "
                  f"{self.instrument_zdim}d per neighbor image"
                  f" (projected to {token_dim}d)"
                  if self.ins_proj else
                  f"\nEncoder 2 (neighbors): {enc2_seq} tokens x "
                  f"{self.instrument_zdim}d per neighbor image")

        has_xattn = any(v is not None for v in level_map.values())
        if has_xattn:
            print("Setting up cross-attention processors (neighbor-aware RoPE)...")
            _setup_rope_with_neighbors(self.conditioned_unet, grid_sizes, rope_enabled)
        print("Done.\n")

        # ---- Geometric loss ----
        if self.lambda_geometric > 0:
            self.geom_loss_fn = geomloss.SamplesLoss(
                loss='sinkhorn', p=2, blur=0.01,
                backend='tensorized', debias=True,
            )
        else:
            self.geom_loss_fn = None

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_galaxy_embeddings_flat(self, images: torch.Tensor) -> torch.Tensor:
        """Flat embedding from encoder_1 (for geometric loss / UMAP)."""
        spatial_levels, _global_vec, _ = self.encoder_1(images)
        all_tokens = torch.cat([t for t, _, _ in spatial_levels], dim=1)
        return all_tokens.flatten(start_dim=1)

    def _encode_neighbors(
        self, cond_image_sameins: torch.Tensor, masks: torch.Tensor,
    ) -> torch.Tensor:
        """Encode neighbor images and return masked, flattened tokens (B, k*seq, D)."""
        B, k, C, H, W = cond_image_sameins.shape
        flat = cond_image_sameins.flatten(0, 1)
        tokens_flat = self.encoder_2(flat)

        if self.ins_proj is not None:
            tokens_flat = self.ins_proj(tokens_flat)

        tokens = tokens_flat.unflatten(0, (B, k))
        mask_expanded = masks.view(B, k, 1, 1).to(tokens.dtype)
        tokens = tokens * mask_expanded
        return tokens.flatten(1, 2)

    def _augment_spatial_levels(self, spatial_levels, neighbor_tokens):
        """Concatenate neighbor tokens to each spatial level's token sequence."""
        augmented = []
        for tokens, h, w in spatial_levels:
            combined = torch.cat([tokens, neighbor_tokens], dim=1)
            augmented.append((combined, h, w))
        return augmented

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_image_samegal: torch.Tensor,
        cond_image_sameins: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t, c).

        Args:
            x_t: Noisy image at time t (B, C, H, W)
            t: Time in [0, 1] (B,)
            cond_image_samegal: Conditioning image (B, C, H, W)
            cond_image_sameins: Set of conditioning images (B, k, C, H, W)
            masks: Valid neighbor mask (B, k), 1 = real, 0 = padding
        """
        timesteps = t * 1000

        spatial_levels, global_vec, _ = self.encoder_1(cond_image_samegal)
        neighbor_tokens = self._encode_neighbors(cond_image_sameins, masks)
        augmented_levels = self._augment_spatial_levels(spatial_levels, neighbor_tokens)

        return self.conditioned_unet(
            sample=x_t,
            timestep=timesteps,
            spatial_levels=augmented_levels,
            class_labels=global_vec,
        )

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """Compute conditional flow matching loss.

        Batch: (anchor, same_galaxy, same_instrument, masks, metadata)
        or legacy 4-tuple without masks.
        """
        if len(batch) == 5:
            x_1, cond_image_samegal, cond_image_sameins, masks, metadata = batch
        else:
            x_1, cond_image_samegal, cond_image_sameins, metadata = batch
            B, k, _, _, _ = cond_image_sameins.shape
            masks = torch.ones(
                (B, k), device=cond_image_sameins.device, dtype=torch.bool,
            )

        # --- Generative loss ---
        if self.lambda_generative > 0:
            batch_size = x_1.shape[0]
            x_0 = torch.randn_like(x_1)
            t = torch.rand(batch_size, device=x_1.device)

            t_expanded = t[:, None, None, None]
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            target_velocity = x_1 - x_0

            predicted_velocity = self(
                x_t, t, cond_image_samegal, cond_image_sameins, masks,
            )

            if self.mask_center:
                mask_size = 48
                _, _, height, width = predicted_velocity.shape
                start_x = (width - mask_size) // 2
                start_y = (height - mask_size) // 2
                loss = F.mse_loss(
                    predicted_velocity[:, :, start_y:start_y+mask_size,
                                      start_x:start_x+mask_size],
                    target_velocity[:, :, start_y:start_y+mask_size,
                                   start_x:start_x+mask_size],
                    reduction='none',
                )
            else:
                loss = F.mse_loss(predicted_velocity, target_velocity, reduction='none')

            per_example_loss = loss.mean(dim=(1, 2, 3))
            anchor_surveys = [m['anchor_survey'] for m in metadata]
            is_hsc = torch.tensor(
                [s == 'hsc' for s in anchor_surveys],
                device=per_example_loss.device,
            )
            is_legacy = torch.tensor(
                [s == 'legacy' for s in anchor_surveys],
                device=per_example_loss.device,
            )
            loss_hsc = (per_example_loss[is_hsc].mean() if is_hsc.any()
                        else torch.tensor(float('nan'), device=per_example_loss.device))
            loss_legacy = (per_example_loss[is_legacy].mean() if is_legacy.any()
                           else torch.tensor(float('nan'), device=per_example_loss.device))
            generative_loss = per_example_loss.mean()

            self._loss_generative_total = generative_loss.detach()
            self._loss_hsc = loss_hsc.detach()
            self._loss_legacy = loss_legacy.detach()
        else:
            device = x_1.device
            dtype = x_1.dtype
            generative_loss = torch.tensor(0.0, device=device, dtype=dtype)
            self._loss_generative_total = generative_loss.detach()
            self._loss_hsc = torch.tensor(float('nan'), device=device)
            self._loss_legacy = torch.tensor(float('nan'), device=device)

        # --- Geometric loss ---
        if self.lambda_geometric > 0:
            embeds_target = self._get_galaxy_embeddings_flat(x_1).contiguous()
            embeds_samegal = self._get_galaxy_embeddings_flat(
                cond_image_samegal,
            ).contiguous()
            total_geom_loss = self.geom_loss_fn(embeds_target, embeds_samegal)
            self._loss_geom_total = total_geom_loss.detach()
        else:
            device = x_1.device
            dtype = x_1.dtype
            total_geom_loss = torch.tensor(0.0, device=device, dtype=dtype)
            self._loss_geom_total = total_geom_loss.detach()

        return (self.lambda_generative * generative_loss
                + self.lambda_geometric * total_geom_loss)

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        cond_image_samegal: torch.Tensor,
        cond_image_sameins: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        x_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using Euler integration."""
        num_steps = num_steps or self.num_integration_steps
        num_samples = cond_image_samegal.shape[0]
        device = cond_image_samegal.device

        if masks is None:
            B, k, _, _, _ = cond_image_sameins.shape
            masks = torch.ones((B, k), device=device, dtype=torch.bool)

        if x_noise is None:
            x = torch.randn(
                num_samples, self.in_channels, self.image_size, self.image_size,
                device=device,
            )
        else:
            x = x_noise.to(device)
            expected_shape = (
                num_samples, self.in_channels, self.image_size, self.image_size,
            )
            if x.shape != expected_shape:
                raise ValueError(
                    f"x_noise shape {x.shape} != expected {expected_shape}"
                )

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            velocity = self(x, t, cond_image_samegal, cond_image_sameins, masks)
            x = x + velocity * dt

        return x

    @torch.no_grad()
    def compute_mse(
        self, target_image, cond_image_samegal, cond_image_sameins,
        metadata=None, masks=None, mask_sizes=(48, 32),
    ):
        """Compute reconstruction MSE on center crops."""
        samples = self.sample(cond_image_samegal, cond_image_sameins, masks=masks)
        diff = target_image - samples
        _, _, height, width = diff.shape
        device = diff.device

        mse_by_size = {}
        for mask_size in mask_sizes:
            start_x = (width - mask_size) // 2
            start_y = (height - mask_size) // 2
            diff_crop = diff[:, :, start_y:start_y+mask_size,
                            start_x:start_x+mask_size]
            mse_by_size[mask_size] = torch.mean(diff_crop ** 2)

        primary_mask_size = mask_sizes[0]
        start_x = (width - primary_mask_size) // 2
        start_y = (height - primary_mask_size) // 2
        diff_primary = diff[:, :, start_y:start_y+primary_mask_size,
                           start_x:start_x+primary_mask_size]

        mse_hsc = None
        mse_legacy = None
        if metadata is not None:
            anchor_surveys = [m['anchor_survey'] for m in metadata]
            hsc_mask = torch.tensor(
                [s == 'hsc' for s in anchor_surveys], device=device,
            )
            legacy_mask = torch.tensor(
                [s == 'legacy' for s in anchor_surveys], device=device,
            )
            mse_hsc = (torch.mean(diff_primary[hsc_mask] ** 2)
                       if hsc_mask.any()
                       else torch.tensor(float('nan'), device=device))
            mse_legacy = (torch.mean(diff_primary[legacy_mask] ** 2)
                          if legacy_mask.any()
                          else torch.tensor(float('nan'), device=device))

        return mse_by_size, mse_hsc, mse_legacy

    # -----------------------------------------------------------------
    # Time formatting
    # -----------------------------------------------------------------

    def _format_time_hms(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    # -----------------------------------------------------------------
    # Lightning hooks
    # -----------------------------------------------------------------

    def on_train_start(self):
        self._train_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training started - Target: {self.trainer.max_steps} steps")
        print(f"H100 GPU detected: {self.is_h100}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

        if (self.trainer.is_global_zero and self.logger
                and hasattr(self.logger, 'experiment')):
            self.logger.experiment.config.update({
                "experiment_config": self.hparams.experiment_config,
                "instrument_zdim": self.instrument_zdim,
                "lambda_generative": self.lambda_generative,
                "lambda_geometric": self.lambda_geometric,
                "is_h100": self.is_h100,
            }, allow_val_change=True)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True,
                 on_epoch=True, sync_dist=True)

        if hasattr(self, '_loss_generative_total'):
            self.log("train/loss_generative_total",
                     self._loss_generative_total,
                     on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_hsc'):
            self.log("train/loss_generative_hsc", self._loss_hsc,
                     on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_legacy'):
            self.log("train/loss_generative_legacy", self._loss_legacy,
                     on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_geom_total'):
            self.log("train/loss_geom_total", self._loss_geom_total,
                     on_step=True, on_epoch=True, sync_dist=True)

        if (self.global_step % 100 == 0
                and hasattr(self, '_train_start_time')
                and self.global_step > 0):
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
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if hasattr(self, '_epoch_start_time'):
            epoch_time = time.time() - self._epoch_start_time
            print(f"Epoch {self.current_epoch} completed in "
                  f"{self._format_time_hms(epoch_time)}")

    def on_validation_epoch_start(self):
        self._umap_hsc_batches = []
        self._umap_legacy_batches = []
        self._umap_batch_count = 0

    def on_train_end(self):
        if hasattr(self, '_train_start_time'):
            total_time = time.time() - self._train_start_time
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Total training time: {self._format_time_hms(total_time)}")
            print(f"Total steps: {self.global_step}")
            print(f"{'='*60}\n")

    def _unpack_batch(self, batch: tuple):
        if len(batch) == 5:
            return batch[0], batch[1], batch[2], batch[3], batch[4]
        anchor_image, same_galaxy, same_instrument, metadata = batch
        return anchor_image, same_galaxy, same_instrument, None, metadata

    def validation_step(self, batch: tuple, batch_idx: int,
                        dataloader_idx: int = 0) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if hasattr(self, '_loss_generative_total'):
            self.log("val/loss_generative_total",
                     self._loss_generative_total, on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_hsc'):
            self.log("val/loss_generative_hsc", self._loss_hsc,
                     on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_legacy'):
            self.log("val/loss_generative_legacy", self._loss_legacy,
                     on_epoch=True, sync_dist=True)
        if hasattr(self, '_loss_geom_total'):
            self.log("val/loss_geom_total", self._loss_geom_total,
                     on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            (anchor_image, same_galaxy, same_instrument,
             masks, metadata) = self._unpack_batch(batch)
            self._val_anchor_batch = anchor_image[:self.num_sample_images].clone()
            self._val_samegal_batch = same_galaxy[:self.num_sample_images].clone()
            self._val_sameins_batch = same_instrument[:self.num_sample_images].clone()
            self._val_masks_batch = (
                masks[:self.num_sample_images].clone()
                if masks is not None else None
            )

            batch_size = anchor_image.shape[0]
            num_mse = min(self.num_mse_images, batch_size)
            self._val_mse_target_batch = anchor_image[:num_mse].clone()
            self._val_mse_samegal_batch = same_galaxy[:num_mse].clone()
            self._val_mse_sameins_batch = same_instrument[:num_mse].clone()
            self._val_mse_masks_batch = (
                masks[:num_mse].clone() if masks is not None else None
            )
            self._val_mse_metadata = metadata[:num_mse] if metadata else None

        if (hasattr(self, '_umap_batch_count')
                and self._umap_batch_count < self.num_umap_batches):
            (anchor_image, same_galaxy, same_instrument,
             _masks, metadata) = self._unpack_batch(batch)
            anchor_surveys = [m['anchor_survey'] for m in metadata]
            hsc_mask = torch.tensor(
                [s == 'hsc' for s in anchor_surveys], device=anchor_image.device,
            )
            legacy_mask = torch.tensor(
                [s == 'legacy' for s in anchor_surveys], device=anchor_image.device,
            )
            if hsc_mask.any():
                self._umap_hsc_batches.append(anchor_image[hsc_mask].cpu())
            if legacy_mask.any():
                self._umap_legacy_batches.append(anchor_image[legacy_mask].cpu())
            self._umap_batch_count += 1

        return loss

    # -----------------------------------------------------------------
    # Visualization helpers
    # -----------------------------------------------------------------

    def _normalize_for_vis(self, img: torch.Tensor) -> torch.Tensor:
        img = img.clone()
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return img

    def on_validation_epoch_end(self) -> None:
        if not self.logger or not hasattr(self, "_val_anchor_batch"):
            return

        import matplotlib.pyplot as plt

        num_cond_images = min(6, len(self._val_anchor_batch))
        num_samples_per_cond = 5
        num_cols = 3 + num_samples_per_cond + 1

        def _row_scale_rgb(x_chw, vmin, vmax):
            x = x_chw[:3]
            vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
            vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
            y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
            return y.clamp(0, 1).permute(1, 2, 0)

        # --- Original grid ---
        fig_orig, axes_orig = plt.subplots(
            num_cond_images, num_cols,
            figsize=(2 * num_cols, 2 * num_cond_images), squeeze=False,
        )
        col_titles = (
            ["SameGal", "SameIns (1st)", "Target"]
            + [f"Sample {j+1}" for j in range(num_samples_per_cond)]
            + ["Mean"]
        )
        for j, title in enumerate(col_titles):
            axes_orig[0, j].set_title(title, fontsize=10)

        # --- Row-scaled grid ---
        fig_row, axes_row = plt.subplots(
            num_cond_images, num_cols,
            figsize=(2 * num_cols, 2 * num_cond_images), squeeze=False,
        )
        for j, title in enumerate(col_titles):
            axes_row[0, j].set_title(title, fontsize=10)

        for i in range(num_cond_images):
            samegal = self._val_samegal_batch[i:i+1].to(self.device)
            target = self._val_anchor_batch[i:i+1].to(self.device)
            sameins = self._val_sameins_batch[i:i+1].to(self.device)
            sameins_first = sameins[:, 0:1]

            samegal_rep = samegal.repeat(num_samples_per_cond, 1, 1, 1)
            sameins_rep = sameins.repeat(num_samples_per_cond, 1, 1, 1, 1)

            masks_i = None
            if (hasattr(self, '_val_masks_batch')
                    and self._val_masks_batch is not None):
                masks_i = self._val_masks_batch[i:i+1].to(
                    self.device,
                ).repeat(num_samples_per_cond, 1)
            samples = self.sample(samegal_rep, sameins_rep, masks=masks_i)
            mean_sample = samples.mean(dim=0, keepdim=True)

            # Original plot row
            samegal_rgb = self._normalize_for_vis(
                samegal[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, 0].imshow(samegal_rgb)
            axes_orig[i, 0].axis("off")

            sameins_rgb = self._normalize_for_vis(
                sameins_first[0, 0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, 1].imshow(sameins_rgb)
            axes_orig[i, 1].axis("off")

            target_rgb = self._normalize_for_vis(
                target[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, 2].imshow(target_rgb)
            axes_orig[i, 2].axis("off")

            for j in range(num_samples_per_cond):
                s_rgb = self._normalize_for_vis(
                    samples[j, :3]).cpu().permute(1, 2, 0).numpy()
                axes_orig[i, 3 + j].imshow(s_rgb)
                axes_orig[i, 3 + j].axis("off")

            mean_rgb = self._normalize_for_vis(
                mean_sample[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes_orig[i, -1].imshow(mean_rgb)
            axes_orig[i, -1].axis("off")

            # Row-scaled plot row
            target_chw = target[0, :3]
            vmin = target_chw.amin(dim=(1, 2))
            vmax = target_chw.amax(dim=(1, 2))

            axes_row[i, 0].imshow(
                _row_scale_rgb(samegal[0, :3], vmin, vmax).detach().cpu().numpy())
            axes_row[i, 0].axis("off")

            axes_row[i, 1].imshow(
                _row_scale_rgb(sameins_first[0, 0, :3], vmin, vmax).detach().cpu().numpy())
            axes_row[i, 1].axis("off")

            axes_row[i, 2].imshow(
                _row_scale_rgb(target[0, :3], vmin, vmax).detach().cpu().numpy())
            axes_row[i, 2].axis("off")

            for j in range(num_samples_per_cond):
                axes_row[i, 3 + j].imshow(
                    _row_scale_rgb(samples[j, :3], vmin, vmax).detach().cpu().numpy())
                axes_row[i, 3 + j].axis("off")

            axes_row[i, -1].imshow(
                _row_scale_rgb(mean_sample[0, :3], vmin, vmax).detach().cpu().numpy())
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

        # MSE metric
        if (hasattr(self, '_val_mse_target_batch')
                and hasattr(self, '_val_mse_samegal_batch')
                and hasattr(self, '_val_mse_sameins_batch')):
            mse_start_time = time.time()
            mse_masks = None
            if (hasattr(self, '_val_mse_masks_batch')
                    and self._val_mse_masks_batch is not None):
                mse_masks = self._val_mse_masks_batch.to(self.device)
            mse_by_size, mse_hsc, mse_legacy = self.compute_mse(
                self._val_mse_target_batch.to(self.device),
                self._val_mse_samegal_batch.to(self.device),
                self._val_mse_sameins_batch.to(self.device),
                self._val_mse_metadata,
                masks=mse_masks,
                mask_sizes=(48, 32),
            )
            mse_time = time.time() - mse_start_time
            if not hasattr(self, '_mse_timing_logged'):
                print(f"[MSE metric] Computation took {mse_time:.2f} seconds")
                self._mse_timing_logged = True

            self.log("val/mse", mse_by_size[48], sync_dist=True)
            self.log("val/mse_32", mse_by_size[32], sync_dist=True)
            if mse_hsc is not None:
                self.log("val/mse_hsc", mse_hsc, sync_dist=True)
            if mse_legacy is not None:
                self.log("val/mse_legacy", mse_legacy, sync_dist=True)

        # UMAP visualization
        if (hasattr(self, '_umap_hsc_batches')
                and hasattr(self, '_umap_legacy_batches')
                and len(self._umap_hsc_batches) > 0
                and len(self._umap_legacy_batches) > 0):
            try:
                hsc_mega = torch.cat(self._umap_hsc_batches, dim=0).to(self.device)
                legacy_mega = torch.cat(self._umap_legacy_batches, dim=0).to(self.device)
                umap_path = self.plot_latent_space(hsc_mega, legacy_mega)
                print(f"[UMAP] Visualization saved to {umap_path}")
            except Exception as e:
                print(f"[UMAP] Error generating visualization: {e}")
                import traceback
                traceback.print_exc()

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # -----------------------------------------------------------------
    # UMAP
    # -----------------------------------------------------------------

    @torch.no_grad()
    def plot_latent_space(self, hsc_batch, legacy_batch):
        """
        UMAP visualization for both encoders.

        2x2 grid:
          (0,0) Encoder 1 — spatial tokens (combined)
          (0,1) Encoder 2 — tokens (combined)
          (1,0) Encoder 1 — global embedding
          (1,1) (unused)
        """
        import matplotlib.pyplot as plt

        # Encoder 1: spatial tokens + global
        spatial_hsc, global_hsc, _ = self.encoder_1(hsc_batch)
        spatial_leg, global_leg, _ = self.encoder_1(legacy_batch)

        hsc_spatial_flat = torch.cat(
            [t for t, _, _ in spatial_hsc], dim=1,
        ).flatten(1).cpu().numpy()
        leg_spatial_flat = torch.cat(
            [t for t, _, _ in spatial_leg], dim=1,
        ).flatten(1).cpu().numpy()
        hsc_global = global_hsc.cpu().numpy()
        leg_global = global_leg.cpu().numpy()

        # Encoder 2: flat tokens
        hsc_enc2 = self.encoder_2(hsc_batch).flatten(1).cpu().numpy()
        leg_enc2 = self.encoder_2(legacy_batch).flatten(1).cpu().numpy()

        num_hsc = len(hsc_batch)

        figures_dir = self.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)

        umap_params = {
            'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 2,
            'metric': 'euclidean', 'random_state': 42, 'n_jobs': 1,
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (0, 0) Encoder 1 spatial tokens
        combined_1 = np.concatenate([hsc_spatial_flat, leg_spatial_flat], axis=0)
        umap_1 = umap.UMAP(**umap_params).fit_transform(combined_1)
        axes[0, 0].scatter(umap_1[:num_hsc, 0], umap_1[:num_hsc, 1],
                           s=5, label='HSC', alpha=0.6, c='blue')
        axes[0, 0].scatter(umap_1[num_hsc:, 0], umap_1[num_hsc:, 1],
                           s=5, label='Legacy', alpha=0.6, c='orange')
        axes[0, 0].set_title('Encoder 1 (Galaxy) — Spatial Tokens', fontsize=10)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # (0, 1) Encoder 2 tokens
        combined_2 = np.concatenate([hsc_enc2, leg_enc2], axis=0)
        umap_2 = umap.UMAP(**umap_params).fit_transform(combined_2)
        axes[0, 1].scatter(umap_2[:num_hsc, 0], umap_2[:num_hsc, 1],
                           s=5, label='HSC', alpha=0.6, c='blue')
        axes[0, 1].scatter(umap_2[num_hsc:, 0], umap_2[num_hsc:, 1],
                           s=5, label='Legacy', alpha=0.6, c='orange')
        axes[0, 1].set_title('Encoder 2 (Instrument) — Tokens', fontsize=10)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # (1, 0) Encoder 1 global embedding
        combined_g = np.concatenate([hsc_global, leg_global], axis=0)
        umap_g = umap.UMAP(**umap_params).fit_transform(combined_g)
        axes[1, 0].scatter(umap_g[:num_hsc, 0], umap_g[:num_hsc, 1],
                           s=5, label='HSC', alpha=0.6, c='blue')
        axes[1, 0].scatter(umap_g[num_hsc:, 0], umap_g[num_hsc:, 1],
                           s=5, label='Legacy', alpha=0.6, c='orange')
        axes[1, 0].set_title('Encoder 1 (Galaxy) — Global Embedding', fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # (1, 1) empty
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5, 0.5, '(Hierarchical attention variant)',
            transform=axes[1, 1].transAxes,
            ha='center', va='center', fontsize=12, alpha=0.5,
        )

        plt.suptitle('UMAP Latent Space Visualization', fontsize=14, y=0.995)
        plt.tight_layout()

        save_path = figures_dir / f'umap_latent_space_step{self.global_step}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({
                "latent_space/umap_grid": wandb.Image(str(save_path)),
                "global_step": self.global_step,
            })

        return save_path


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _train_script = _script_dir / "neighbours_train_hier.py"
    print("This module is not intended to be run directly.", file=sys.stderr)
    print(f"Run the training script instead: {_train_script}", file=sys.stderr)
    sys.exit(1)
