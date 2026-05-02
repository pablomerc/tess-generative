"""
Hierarchical neighbors variant with a separate global instrument path.

The same-galaxy image is encoded hierarchically and routed through UNet
cross-attention. Same-instrument neighbors are encoded independently with a
global ResNet encoder and pooled with a masked mean before being injected
through a second global projection path.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import geomloss
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from galaxy_images.galaxy_model.double_train_fm_neighbors import (
    ResNetEncoder,
    is_h100_gpu,
)
from galaxy_images.galaxy_model.hierarchical_attention.train_experiments import (
    ConditionedUNet,
    ConfigurableEncoder,
    EXPERIMENTS,
    build_unet_and_level_map,
)
from galaxy_images.galaxy_model.validation_pairs import reconstruct_hsc_legacy_pairs


class DualGlobalConditionedUNet(ConditionedUNet):
    """Conditioned UNet with separate galaxy-global and instrument-global paths."""

    def __init__(
        self,
        unet,
        level_map,
        instrument_global_dim: int,
    ):
        super().__init__(unet=unet, level_map=level_map)
        time_embed_dim = self.unet.time_embedding.linear_2.out_features
        self.instrument_global_proj = nn.Sequential(
            nn.LayerNorm(instrument_global_dim),
            nn.Linear(instrument_global_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(
        self,
        sample,
        timestep,
        spatial_levels,
        class_labels=None,
        instrument_global_labels=None,
    ):
        ts = timestep
        if not torch.is_tensor(ts):
            ts = torch.tensor([ts], dtype=torch.long, device=sample.device)
        elif ts.ndim == 0:
            ts = ts[None].to(sample.device)
        ts = ts.expand(sample.shape[0])

        t_emb = self.unet.time_proj(ts).to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb)
        if self.unet.class_embedding is not None and class_labels is not None:
            emb = emb + self.unet.class_embedding(class_labels).to(dtype=sample.dtype)
        if instrument_global_labels is not None:
            emb = emb + self.instrument_global_proj(instrument_global_labels).to(
                dtype=sample.dtype
            )

        sample = self.unet.conv_in(sample)

        down_res = (sample,)
        for i, block in enumerate(self.unet.down_blocks):
            lvl = self.level_map.get(f"down_{i}")
            enc = spatial_levels[lvl][0] if lvl is not None else None
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample, res = block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=enc,
                )
            else:
                sample, res = block(hidden_states=sample, temb=emb)
            down_res += res

        lvl = self.level_map.get("mid")
        enc = spatial_levels[lvl][0] if lvl is not None else None
        if (
            hasattr(self.unet.mid_block, "has_cross_attention")
            and self.unet.mid_block.has_cross_attention
        ):
            sample = self.unet.mid_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=enc,
            )
        else:
            sample = self.unet.mid_block(hidden_states=sample, temb=emb)

        for i, block in enumerate(self.unet.up_blocks):
            n = len(block.resnets)
            res = down_res[-n:]
            down_res = down_res[:-n]
            lvl = self.level_map.get(f"up_{i}")
            enc = spatial_levels[lvl][0] if lvl is not None else None
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample = block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res,
                    encoder_hidden_states=enc,
                )
            else:
                sample = block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res,
                )

        if self.unet.conv_norm_out:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        return self.unet.conv_out(sample)


class HierarchicalGlobalInstrumentFlowMatchingModule(pl.LightningModule):
    """
    Hierarchical same-galaxy conditioning with pooled global instrument context.
    """

    def __init__(
        self,
        experiment_config,
        in_channels: int = 4,
        cond_channels: int = 4,
        image_size: int = 48,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        layers_per_block: int = 2,
        attention_head_dim: int = 64,
        instrument_zdim: int = None,
        instrument_pooling: str = "masked_mean",
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
        global_dim = cfg["global_dim"]

        if instrument_pooling != "masked_mean":
            raise ValueError(
                f"Unsupported instrument_pooling={instrument_pooling!r}. "
                "Only 'masked_mean' is implemented."
            )

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
        self.instrument_pooling = instrument_pooling
        self.figures_dir = (
            Path(figures_dir)
            if figures_dir
            else (Path(__file__).resolve().parent / "figures")
        )
        self.token_dim = token_dim
        self.global_dim = global_dim
        self.instrument_zdim = (
            instrument_zdim if instrument_zdim is not None else token_dim
        )
        self.is_h100 = is_h100_gpu()

        self.encoder_1 = ConfigurableEncoder(
            in_channels=cond_channels,
            spatial_indices=cfg["spatial_indices"],
            reductions=cfg.get("reductions", {}),
            token_dim=token_dim,
            global_dim=global_dim,
            pretrained=pretrained_encoder,
        )

        self.encoder_2 = ResNetEncoder(
            in_channels=cond_channels,
            cross_attention_dim=self.instrument_zdim,
            pretrained=pretrained_encoder,
            mean_pool=True,
        )

        unet, level_map, effective_hd = build_unet_and_level_map(
            config=cfg,
            in_channels=in_channels,
            image_size=image_size,
            model_channels=model_channels,
            channel_mult=channel_mult,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
        )
        self.conditioned_unet = DualGlobalConditionedUNet(
            unet=unet,
            level_map=level_map,
            instrument_global_dim=self.instrument_zdim,
        )

        print("\nEncoder 1 (galaxy) configuration:")
        with torch.no_grad():
            dummy = torch.zeros(1, cond_channels, image_size, image_size)
            spatial_levels, _global_vec, rope_flags = self.encoder_1(dummy)

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
                print(
                    f"  Level {i}: {gh}x{gw} = {n_tok} tokens x {token_dim}d"
                    f" = {n_tok * token_dim} values"
                    f" | RoPE={'yes' if rope_flags[i] else 'no'}"
                )

            total_cond = total_spatial_values + global_dim
            input_vals = cond_channels * image_size * image_size
            ratio = input_vals / total_cond
            tag = (
                f"{ratio:.1f}x compression"
                if ratio > 1
                else f"{1 / ratio:.1f}x expansion"
            )
            print(f"  Global: {global_dim}d")
            print(
                f"  Total: {total_tokens} tokens + global = "
                f"{total_cond} values ({tag} from {input_vals})"
            )
            print(f"  Attention head dim: {effective_hd}")

            enc2_out = self.encoder_2(dummy)
            print(
                f"\nEncoder 2 (instrument): {enc2_out.shape[1]} token x "
                f"{self.instrument_zdim}d per image"
            )

        has_xattn = any(v is not None for v in level_map.values())
        if has_xattn:
            print("Setting up cross-attention processors (standard hierarchical RoPE)...")
            self.conditioned_unet.setup_rope_processors(grid_sizes, rope_enabled)
        print("Done.\n")

        if self.lambda_geometric > 0:
            self.geom_loss_fn = geomloss.SamplesLoss(
                loss="sinkhorn",
                p=2,
                blur=0.01,
                backend="tensorized",
                debias=True,
            )
        else:
            self.geom_loss_fn = None

    def _get_galaxy_embeddings_flat(self, images: torch.Tensor) -> torch.Tensor:
        spatial_levels, _global_vec, _ = self.encoder_1(images)
        all_tokens = torch.cat([tokens for tokens, _, _ in spatial_levels], dim=1)
        return all_tokens.flatten(start_dim=1)

    def encode_image(self, image: torch.Tensor) -> dict:
        if image.ndim != 4:
            raise ValueError(
                f"encode_image expects input of shape (B, C, H, W), got {tuple(image.shape)}."
            )

        spatial_levels, global_vec, rope_flags = self.encoder_1(image)
        physics_levels = []
        level_flats = []
        for (tokens, height, width), rope in zip(spatial_levels, rope_flags):
            physics_levels.append(
                {
                    "tokens": tokens,
                    "height": height,
                    "width": width,
                    "rope": rope,
                }
            )
            level_flats.append(tokens.flatten(start_dim=1))

        if physics_levels:
            spatial_concat = torch.cat(
                [level["tokens"] for level in physics_levels],
                dim=1,
            )
            spatial_flat = spatial_concat.flatten(start_dim=1)
        else:
            batch_size = image.shape[0]
            spatial_concat = image.new_zeros((batch_size, 0, self.token_dim))
            spatial_flat = image.new_zeros((batch_size, 0))

        instrument_tokens = self.encoder_2(image)

        return {
            "physics": {
                "spatial_levels": physics_levels,
                "level_flats": level_flats,
                "spatial_concat": spatial_concat,
                "spatial_flat": spatial_flat,
                "global_vec": global_vec,
            },
            "instrument": {
                "tokens": instrument_tokens,
                "flat": instrument_tokens.squeeze(1),
            },
        }

    @staticmethod
    def _masked_mean_pool(tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        mask = masks.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (tokens * mask).sum(dim=1) / denom

    def _pool_instrument_conditioning(
        self,
        cond_image_sameins: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.instrument_pooling != "masked_mean":
            raise ValueError(
                f"Unsupported instrument_pooling={self.instrument_pooling!r}."
            )

        batch_size, num_neighbors = cond_image_sameins.shape[:2]
        if num_neighbors == 0:
            return cond_image_sameins.new_zeros((batch_size, self.instrument_zdim))

        flat_images = cond_image_sameins.flatten(0, 1)
        instrument_tokens = self.encoder_2(flat_images)
        if instrument_tokens.ndim != 3 or instrument_tokens.shape[1] != 1:
            raise ValueError(
                "Expected mean-pooled instrument encoder output of shape (B, 1, D)."
            )
        instrument_tokens = instrument_tokens.squeeze(1).unflatten(
            0,
            (batch_size, num_neighbors),
        )
        return self._masked_mean_pool(instrument_tokens, masks)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_image_samegal: torch.Tensor,
        cond_image_sameins: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = t * 1000

        spatial_levels, global_vec, _ = self.encoder_1(cond_image_samegal)
        instrument_global = self._pool_instrument_conditioning(
            cond_image_sameins,
            masks,
        )

        return self.conditioned_unet(
            sample=x_t,
            timestep=timesteps,
            spatial_levels=spatial_levels,
            class_labels=global_vec,
            instrument_global_labels=instrument_global,
        )

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        if len(batch) == 5:
            x_1, cond_image_samegal, cond_image_sameins, masks, metadata = batch
        else:
            x_1, cond_image_samegal, cond_image_sameins, metadata = batch
            batch_size, num_neighbors = cond_image_sameins.shape[:2]
            masks = torch.ones(
                (batch_size, num_neighbors),
                device=cond_image_sameins.device,
                dtype=torch.bool,
            )

        if self.lambda_generative > 0:
            batch_size = x_1.shape[0]
            x_0 = torch.randn_like(x_1)
            t = torch.rand(batch_size, device=x_1.device)

            t_expanded = t[:, None, None, None]
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            target_velocity = x_1 - x_0

            predicted_velocity = self(
                x_t,
                t,
                cond_image_samegal,
                cond_image_sameins,
                masks,
            )

            if self.mask_center:
                mask_size = 48
                _, _, height, width = predicted_velocity.shape
                start_x = (width - mask_size) // 2
                start_y = (height - mask_size) // 2
                loss = F.mse_loss(
                    predicted_velocity[
                        :,
                        :,
                        start_y : start_y + mask_size,
                        start_x : start_x + mask_size,
                    ],
                    target_velocity[
                        :,
                        :,
                        start_y : start_y + mask_size,
                        start_x : start_x + mask_size,
                    ],
                    reduction="none",
                )
            else:
                loss = F.mse_loss(
                    predicted_velocity,
                    target_velocity,
                    reduction="none",
                )

            per_example_loss = loss.mean(dim=(1, 2, 3))
            anchor_surveys = [m["anchor_survey"] for m in metadata]
            is_hsc = torch.tensor(
                [survey == "hsc" for survey in anchor_surveys],
                device=per_example_loss.device,
            )
            is_legacy = torch.tensor(
                [survey == "legacy" for survey in anchor_surveys],
                device=per_example_loss.device,
            )
            loss_hsc = (
                per_example_loss[is_hsc].mean()
                if is_hsc.any()
                else torch.tensor(float("nan"), device=per_example_loss.device)
            )
            loss_legacy = (
                per_example_loss[is_legacy].mean()
                if is_legacy.any()
                else torch.tensor(float("nan"), device=per_example_loss.device)
            )
            generative_loss = per_example_loss.mean()

            self._loss_generative_total = generative_loss.detach()
            self._loss_hsc = loss_hsc.detach()
            self._loss_legacy = loss_legacy.detach()
        else:
            device = x_1.device
            dtype = x_1.dtype
            generative_loss = torch.tensor(0.0, device=device, dtype=dtype)
            self._loss_generative_total = generative_loss.detach()
            self._loss_hsc = torch.tensor(float("nan"), device=device)
            self._loss_legacy = torch.tensor(float("nan"), device=device)

        if self.lambda_geometric > 0:
            embeds_target = self._get_galaxy_embeddings_flat(x_1).contiguous()
            embeds_samegal = self._get_galaxy_embeddings_flat(cond_image_samegal).contiguous()
            total_geom_loss = self.geom_loss_fn(embeds_target, embeds_samegal)
            self._loss_geom_total = total_geom_loss.detach()
        else:
            device = x_1.device
            dtype = x_1.dtype
            total_geom_loss = torch.tensor(0.0, device=device, dtype=dtype)
            self._loss_geom_total = total_geom_loss.detach()

        return (
            self.lambda_generative * generative_loss
            + self.lambda_geometric * total_geom_loss
        )

    @torch.no_grad()
    def sample(
        self,
        cond_image_samegal: torch.Tensor,
        cond_image_sameins: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        x_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_steps = num_steps or self.num_integration_steps
        num_samples = cond_image_samegal.shape[0]
        device = cond_image_samegal.device

        if masks is None:
            batch_size, num_neighbors = cond_image_sameins.shape[:2]
            masks = torch.ones((batch_size, num_neighbors), device=device, dtype=torch.bool)

        if x_noise is None:
            x = torch.randn(
                num_samples,
                self.in_channels,
                self.image_size,
                self.image_size,
                device=device,
            )
        else:
            x = x_noise.to(device)
            expected_shape = (
                num_samples,
                self.in_channels,
                self.image_size,
                self.image_size,
            )
            if x.shape != expected_shape:
                raise ValueError(f"x_noise shape {x.shape} != expected {expected_shape}")

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            velocity = self(x, t, cond_image_samegal, cond_image_sameins, masks)
            x = x + velocity * dt

        return x

    @torch.no_grad()
    def compute_mse(
        self,
        target_image,
        cond_image_samegal,
        cond_image_sameins,
        metadata=None,
        masks=None,
        mask_sizes=(48, 32),
    ):
        samples = self.sample(cond_image_samegal, cond_image_sameins, masks=masks)
        diff = target_image - samples
        _, _, height, width = diff.shape
        device = diff.device

        mse_by_size = {}
        for mask_size in mask_sizes:
            start_x = (width - mask_size) // 2
            start_y = (height - mask_size) // 2
            diff_crop = diff[
                :,
                :,
                start_y : start_y + mask_size,
                start_x : start_x + mask_size,
            ]
            mse_by_size[mask_size] = torch.mean(diff_crop**2)

        primary_mask_size = mask_sizes[0]
        start_x = (width - primary_mask_size) // 2
        start_y = (height - primary_mask_size) // 2
        diff_primary = diff[
            :,
            :,
            start_y : start_y + primary_mask_size,
            start_x : start_x + primary_mask_size,
        ]

        mse_hsc = None
        mse_legacy = None
        if metadata is not None:
            anchor_surveys = [m["anchor_survey"] for m in metadata]
            hsc_mask = torch.tensor(
                [survey == "hsc" for survey in anchor_surveys],
                device=device,
            )
            legacy_mask = torch.tensor(
                [survey == "legacy" for survey in anchor_surveys],
                device=device,
            )
            mse_hsc = (
                torch.mean(diff_primary[hsc_mask] ** 2)
                if hsc_mask.any()
                else torch.tensor(float("nan"), device=device)
            )
            mse_legacy = (
                torch.mean(diff_primary[legacy_mask] ** 2)
                if legacy_mask.any()
                else torch.tensor(float("nan"), device=device)
            )

        return mse_by_size, mse_hsc, mse_legacy

    def _format_time_hms(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def on_train_start(self):
        self._train_start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"Training started - Target: {self.trainer.max_steps} steps")
        print(f"H100 GPU detected: {self.is_h100}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'=' * 60}\n")

        if (
            self.trainer.is_global_zero
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "config")
        ):
            self.logger.experiment.config.update(
                {
                    "experiment_config": self.hparams.experiment_config,
                    "instrument_zdim": self.instrument_zdim,
                    "instrument_pooling": self.instrument_pooling,
                    "lambda_generative": self.lambda_generative,
                    "lambda_geometric": self.lambda_geometric,
                    "is_h100": self.is_h100,
                },
                allow_val_change=True,
            )

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        if hasattr(self, "_loss_generative_total"):
            self.log(
                "train/loss_generative_total",
                self._loss_generative_total,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        if hasattr(self, "_loss_hsc"):
            self.log(
                "train/loss_generative_hsc",
                self._loss_hsc,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        if hasattr(self, "_loss_legacy"):
            self.log(
                "train/loss_generative_legacy",
                self._loss_legacy,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        if hasattr(self, "_loss_geom_total"):
            self.log(
                "train/loss_geom_total",
                self._loss_geom_total,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        if (
            self.global_step % 100 == 0
            and hasattr(self, "_train_start_time")
            and self.global_step > 0
        ):
            elapsed_time = time.time() - self._train_start_time
            max_steps = self.trainer.max_steps
            if max_steps > 0:
                steps_per_second = self.global_step / elapsed_time
                remaining_steps = max_steps - self.global_step
                estimated_remaining = remaining_steps / steps_per_second
                progress = (self.global_step / max_steps) * 100
                elapsed_str = self._format_time_hms(elapsed_time)
                remaining_str = self._format_time_hms(estimated_remaining)
                print(
                    f"Step {self.global_step}/{max_steps} ({progress:.1f}%) | "
                    f"Elapsed: {elapsed_str} | ETA: {remaining_str} | "
                    f"Speed: {steps_per_second:.2f} steps/s"
                )

        return loss

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if hasattr(self, "_epoch_start_time"):
            epoch_time = time.time() - self._epoch_start_time
            print(
                f"Epoch {self.current_epoch} completed in "
                f"{self._format_time_hms(epoch_time)}"
            )

    def on_validation_epoch_start(self):
        self._umap_hsc_batches = []
        self._umap_legacy_batches = []
        self._umap_batch_count = 0

    def on_train_end(self):
        if hasattr(self, "_train_start_time"):
            total_time = time.time() - self._train_start_time
            print(f"\n{'=' * 60}")
            print("Training completed!")
            print(f"Total training time: {self._format_time_hms(total_time)}")
            print(f"Total steps: {self.global_step}")
            print(f"{'=' * 60}\n")

    def _unpack_batch(self, batch: tuple):
        if len(batch) == 5:
            return batch[0], batch[1], batch[2], batch[3], batch[4]
        anchor_image, same_galaxy, same_instrument, metadata = batch
        return anchor_image, same_galaxy, same_instrument, None, metadata

    def validation_step(
        self,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if hasattr(self, "_loss_generative_total"):
            self.log(
                "val/loss_generative_total",
                self._loss_generative_total,
                on_epoch=True,
                sync_dist=True,
            )
        if hasattr(self, "_loss_hsc"):
            self.log(
                "val/loss_generative_hsc",
                self._loss_hsc,
                on_epoch=True,
                sync_dist=True,
            )
        if hasattr(self, "_loss_legacy"):
            self.log(
                "val/loss_generative_legacy",
                self._loss_legacy,
                on_epoch=True,
                sync_dist=True,
            )
        if hasattr(self, "_loss_geom_total"):
            self.log(
                "val/loss_geom_total",
                self._loss_geom_total,
                on_epoch=True,
                sync_dist=True,
            )

        if batch_idx == 0:
            (
                anchor_image,
                same_galaxy,
                same_instrument,
                masks,
                metadata,
            ) = self._unpack_batch(batch)
            self._val_anchor_batch = anchor_image[: self.num_sample_images].clone()
            self._val_samegal_batch = same_galaxy[: self.num_sample_images].clone()
            self._val_sameins_batch = same_instrument[: self.num_sample_images].clone()
            self._val_masks_batch = (
                masks[: self.num_sample_images].clone()
                if masks is not None
                else None
            )

            batch_size = anchor_image.shape[0]
            num_mse = min(self.num_mse_images, batch_size)
            self._val_mse_target_batch = anchor_image[:num_mse].clone()
            self._val_mse_samegal_batch = same_galaxy[:num_mse].clone()
            self._val_mse_sameins_batch = same_instrument[:num_mse].clone()
            self._val_mse_masks_batch = masks[:num_mse].clone() if masks is not None else None
            self._val_mse_metadata = metadata[:num_mse] if metadata else None

        if (
            hasattr(self, "_umap_batch_count")
            and self._umap_batch_count < self.num_umap_batches
        ):
            (
                anchor_image,
                same_galaxy,
                _same_instrument,
                _masks,
                metadata,
            ) = self._unpack_batch(batch)
            hsc_images, legacy_images = reconstruct_hsc_legacy_pairs(
                anchor_image,
                same_galaxy,
                metadata,
            )
            self._umap_hsc_batches.append(hsc_images.cpu())
            self._umap_legacy_batches.append(legacy_images.cpu())
            self._umap_batch_count += 1

        return loss

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

        fig_orig, axes_orig = plt.subplots(
            num_cond_images,
            num_cols,
            figsize=(2 * num_cols, 2 * num_cond_images),
            squeeze=False,
        )
        col_titles = (
            ["SameGal", "SameIns (1st)", "Target"]
            + [f"Sample {j + 1}" for j in range(num_samples_per_cond)]
            + ["Mean"]
        )
        for j, title in enumerate(col_titles):
            axes_orig[0, j].set_title(title, fontsize=10)

        fig_row, axes_row = plt.subplots(
            num_cond_images,
            num_cols,
            figsize=(2 * num_cols, 2 * num_cond_images),
            squeeze=False,
        )
        for j, title in enumerate(col_titles):
            axes_row[0, j].set_title(title, fontsize=10)

        for i in range(num_cond_images):
            samegal = self._val_samegal_batch[i : i + 1].to(self.device)
            target = self._val_anchor_batch[i : i + 1].to(self.device)
            sameins = self._val_sameins_batch[i : i + 1].to(self.device)
            sameins_first = sameins[:, 0:1]

            samegal_rep = samegal.repeat(num_samples_per_cond, 1, 1, 1)
            sameins_rep = sameins.repeat(num_samples_per_cond, 1, 1, 1, 1)

            masks_i = None
            if hasattr(self, "_val_masks_batch") and self._val_masks_batch is not None:
                masks_i = self._val_masks_batch[i : i + 1].to(self.device).repeat(
                    num_samples_per_cond,
                    1,
                )
            samples = self.sample(samegal_rep, sameins_rep, masks=masks_i)
            mean_sample = samples.mean(dim=0, keepdim=True)

            samegal_rgb = (
                self._normalize_for_vis(samegal[0, :3]).cpu().permute(1, 2, 0).numpy()
            )
            axes_orig[i, 0].imshow(samegal_rgb)
            axes_orig[i, 0].axis("off")

            sameins_rgb = (
                self._normalize_for_vis(sameins_first[0, 0, :3])
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            axes_orig[i, 1].imshow(sameins_rgb)
            axes_orig[i, 1].axis("off")

            target_rgb = (
                self._normalize_for_vis(target[0, :3]).cpu().permute(1, 2, 0).numpy()
            )
            axes_orig[i, 2].imshow(target_rgb)
            axes_orig[i, 2].axis("off")

            for j in range(num_samples_per_cond):
                sample_rgb = (
                    self._normalize_for_vis(samples[j, :3])
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                )
                axes_orig[i, 3 + j].imshow(sample_rgb)
                axes_orig[i, 3 + j].axis("off")

            mean_rgb = (
                self._normalize_for_vis(mean_sample[0, :3])
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            axes_orig[i, -1].imshow(mean_rgb)
            axes_orig[i, -1].axis("off")

            target_chw = target[0, :3]
            vmin = target_chw.amin(dim=(1, 2))
            vmax = target_chw.amax(dim=(1, 2))

            axes_row[i, 0].imshow(
                _row_scale_rgb(samegal[0, :3], vmin, vmax).detach().cpu().numpy()
            )
            axes_row[i, 0].axis("off")

            axes_row[i, 1].imshow(
                _row_scale_rgb(sameins_first[0, 0, :3], vmin, vmax)
                .detach()
                .cpu()
                .numpy()
            )
            axes_row[i, 1].axis("off")

            axes_row[i, 2].imshow(
                _row_scale_rgb(target[0, :3], vmin, vmax).detach().cpu().numpy()
            )
            axes_row[i, 2].axis("off")

            for j in range(num_samples_per_cond):
                axes_row[i, 3 + j].imshow(
                    _row_scale_rgb(samples[j, :3], vmin, vmax).detach().cpu().numpy()
                )
                axes_row[i, 3 + j].axis("off")

            axes_row[i, -1].imshow(
                _row_scale_rgb(mean_sample[0, :3], vmin, vmax).detach().cpu().numpy()
            )
            axes_row[i, -1].axis("off")

        plt.figure(fig_orig.number)
        plt.tight_layout()
        plt.figure(fig_row.number)
        plt.tight_layout()

        self.logger.experiment.log(
            {
                "val/sample_grid": wandb.Image(fig_orig),
                "val/sample_grid_row_scaled": wandb.Image(fig_row),
                "global_step": self.global_step,
            }
        )
        plt.close(fig_orig)
        plt.close(fig_row)

        if (
            hasattr(self, "_val_mse_target_batch")
            and hasattr(self, "_val_mse_samegal_batch")
            and hasattr(self, "_val_mse_sameins_batch")
        ):
            mse_start_time = time.time()
            mse_masks = None
            if hasattr(self, "_val_mse_masks_batch") and self._val_mse_masks_batch is not None:
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
            if not hasattr(self, "_mse_timing_logged"):
                print(f"[MSE metric] Computation took {mse_time:.2f} seconds")
                self._mse_timing_logged = True

            self.log("val/mse", mse_by_size[48], sync_dist=True)
            self.log("val/mse_32", mse_by_size[32], sync_dist=True)
            if mse_hsc is not None:
                self.log("val/mse_hsc", mse_hsc, sync_dist=True)
            if mse_legacy is not None:
                self.log("val/mse_legacy", mse_legacy, sync_dist=True)

        if (
            hasattr(self, "_umap_hsc_batches")
            and hasattr(self, "_umap_legacy_batches")
            and len(self._umap_hsc_batches) > 0
            and len(self._umap_legacy_batches) > 0
        ):
            try:
                hsc_mega = torch.cat(self._umap_hsc_batches, dim=0).to(self.device)
                legacy_mega = torch.cat(self._umap_legacy_batches, dim=0).to(self.device)
                umap_path = self.plot_latent_space(hsc_mega, legacy_mega)
                print(f"[UMAP] Visualization saved to {umap_path}")
            except Exception as e:
                print(f"[UMAP] Error generating visualization: {e}")
                import traceback

                traceback.print_exc()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def _fit_umap(features: np.ndarray) -> np.ndarray:
        if features.shape[0] < 3:
            raise ValueError("UMAP requires at least 3 points.")
        umap_params = {
            "n_neighbors": min(15, features.shape[0] - 1),
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
            "n_jobs": 1,
        }
        return umap.UMAP(**umap_params).fit_transform(features)

    @torch.no_grad()
    def plot_latent_space(self, hsc_batch, legacy_batch):
        import matplotlib.pyplot as plt

        hsc_encoded = self.encode_image(hsc_batch)
        legacy_encoded = self.encode_image(legacy_batch)

        num_hsc = hsc_batch.shape[0]
        hsc_physics = hsc_encoded["physics"]
        legacy_physics = legacy_encoded["physics"]
        hsc_instrument = hsc_encoded["instrument"]
        legacy_instrument = legacy_encoded["instrument"]

        panels = []
        for level_idx, (hsc_level, legacy_level) in enumerate(
            zip(hsc_physics["level_flats"], legacy_physics["level_flats"])
        ):
            panels.append(
                (
                    f"Physics - Level {level_idx}",
                    np.concatenate(
                        [
                            hsc_level.cpu().numpy(),
                            legacy_level.cpu().numpy(),
                        ],
                        axis=0,
                    ),
                )
            )

        panels.extend(
            [
                (
                    "Physics - Combined Spatial",
                    np.concatenate(
                        [
                            hsc_physics["spatial_flat"].cpu().numpy(),
                            legacy_physics["spatial_flat"].cpu().numpy(),
                        ],
                        axis=0,
                    ),
                ),
                (
                    "Physics - Global",
                    np.concatenate(
                        [
                            hsc_physics["global_vec"].cpu().numpy(),
                            legacy_physics["global_vec"].cpu().numpy(),
                        ],
                        axis=0,
                    ),
                ),
                (
                    "Instrument - Global",
                    np.concatenate(
                        [
                            hsc_instrument["flat"].cpu().numpy(),
                            legacy_instrument["flat"].cpu().numpy(),
                        ],
                        axis=0,
                    ),
                ),
            ]
        )

        figures_dir = self.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)

        num_panels = len(panels)
        num_cols = 2
        num_rows = math.ceil(num_panels / num_cols)
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(8 * num_cols, 5 * num_rows),
            squeeze=False,
        )

        for ax, (title, features) in zip(axes.flat, panels):
            umap_points = self._fit_umap(features)
            ax.scatter(
                umap_points[:num_hsc, 0],
                umap_points[:num_hsc, 1],
                s=6,
                label="HSC",
                alpha=0.6,
                c="blue",
            )
            ax.scatter(
                umap_points[num_hsc:, 0],
                umap_points[num_hsc:, 1],
                s=6,
                label="Legacy",
                alpha=0.6,
                c="orange",
            )
            ax.set_title(title, fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[num_panels:]:
            ax.axis("off")

        plt.suptitle("UMAP Latent Space Visualization", fontsize=14, y=0.995)
        plt.tight_layout()

        save_path = figures_dir / f"umap_latent_space_step{self.global_step}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.log(
                {
                    "latent_space/umap_grid": wandb.Image(str(save_path)),
                    "global_step": self.global_step,
                }
            )

        return save_path


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _train_script = _script_dir / "neighbours_train_hier_global_ins.py"
    print("This module is not intended to be run directly.", file=sys.stderr)
    print(f"Run the training script instead: {_train_script}", file=sys.stderr)
    sys.exit(1)
