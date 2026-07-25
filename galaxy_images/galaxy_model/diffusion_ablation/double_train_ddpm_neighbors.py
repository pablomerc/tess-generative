"""Conditional DDPM module — diffusion-objective ablation of ConditionalFlowMatchingModule.

Overrides only the generative objective (corruption + target) and the sampler.
Encoders, UNet, validation, UMAP, and optimizers are inherited unchanged.
"""
from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule


class ConditionalDDPMModule(ConditionalFlowMatchingModule):
    """DDPM ε-/v-prediction training with DDIM sampling on the shared dual-encoder UNet."""

    def __init__(
        self,
        prediction_type: str = "epsilon",
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        **kwargs,
    ):
        # ModelConfig may contain keys the parent does not accept. Our **kwargs makes
        # filter_supported_model_kwargs pass everything through, so peel extras here.
        parent_params = inspect.signature(ConditionalFlowMatchingModule.__init__).parameters
        parent_kwargs = {
            k: v for k, v in kwargs.items() if k in parent_params and k != "self"
        }
        super().__init__(**parent_kwargs)
        if getattr(self, "lambda_geometric", 0.0):
            raise ValueError(
                "ConditionalDDPMModule omits the Sinkhorn/geometric branch; "
                "lambda_geometric must be 0.0"
            )
        # REQUIRED: subclass args must land in hparams for load_from_checkpoint.
        self.save_hyperparameters()
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False,  # data is z-scored, not in [-1, 1]
        )
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False,
        )

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """Compute DDPM ε-/v-prediction MSE (same logging contract as the FM parent)."""
        if len(batch) == 5:
            x_1, cond_image_samegal, cond_image_sameins, masks, metadata = batch
        else:
            x_1, cond_image_samegal, cond_image_sameins, metadata = batch
            B, k, _, _, _ = cond_image_sameins.shape
            masks = torch.ones((B, k), device=cond_image_sameins.device, dtype=torch.bool)

        if self.lambda_generative > 0:
            noise = torch.randn_like(x_1)
            t = torch.randint(
                0,
                self.train_scheduler.config.num_train_timesteps,
                (x_1.shape[0],),
                device=x_1.device,
            )
            x_t = self.train_scheduler.add_noise(x_1, noise, t)
            if self.hparams.prediction_type == "epsilon":
                target = noise
            else:  # "v_prediction"
                target = self.train_scheduler.get_velocity(x_1, noise, t)

            # Inherited forward multiplies t by 1000; pass t/1000 to deliver integer timesteps.
            pred = self(
                x_t,
                t.float() / 1000.0,
                cond_image_samegal,
                cond_image_sameins,
                masks,
            )

            if self.mask_center:
                mask_size = 48
                _, _, height, width = pred.shape
                start_x = (width - mask_size) // 2
                start_y = (height - mask_size) // 2
                loss = nn.functional.mse_loss(
                    pred[:, :, start_y : start_y + mask_size, start_x : start_x + mask_size],
                    target[:, :, start_y : start_y + mask_size, start_x : start_x + mask_size],
                    reduction="none",
                )
            else:
                loss = nn.functional.mse_loss(pred, target, reduction="none")

            per_example_loss = loss.mean(dim=(1, 2, 3))

            anchor_surveys = [m["anchor_survey"] for m in metadata]
            is_hsc = torch.tensor(
                [s == "hsc" for s in anchor_surveys], device=per_example_loss.device
            )
            is_legacy = torch.tensor(
                [s == "legacy" for s in anchor_surveys], device=per_example_loss.device
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

        # Sinkhorn / lambda_geometric branch omitted (always 0.0 in these runs).
        total_geom_loss = torch.tensor(0.0, device=x_1.device, dtype=x_1.dtype)
        self._loss_geom_total = total_geom_loss.detach()

        return self.lambda_generative * generative_loss + self.lambda_geometric * total_geom_loss

    @torch.no_grad()
    def sample(
        self,
        cond_image_samegal: torch.Tensor,
        cond_image_sameins: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        x_noise: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """DDIM sample. ``eta=0`` is deterministic; ``eta>0`` uses ``generator`` for noise."""
        num_steps = num_steps or self.num_integration_steps
        num_samples = cond_image_samegal.shape[0]
        device = cond_image_samegal.device

        if masks is None:
            B, k, _, _, _ = cond_image_sameins.shape
            masks = torch.ones((B, k), device=device, dtype=torch.bool)

        sched = self.inference_scheduler
        sched.set_timesteps(num_steps, device=device)

        if x_noise is None:
            x = torch.randn(
                num_samples,
                self.in_channels,
                self.image_size,
                self.image_size,
                device=device,
                generator=generator,
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
                raise ValueError(
                    f"x_noise shape {x.shape} does not match expected shape {expected_shape}"
                )

        x = x * sched.init_noise_sigma

        if eta > 0 and generator is None:
            raise ValueError(
                "eta>0 requires a seeded `generator` for reproducible stochastic sampling"
            )

        for t in sched.timesteps:
            t_batch = t.expand(num_samples).float() / 1000.0
            pred = self(x, t_batch, cond_image_samegal, cond_image_sameins, masks)
            step_kwargs = {"eta": eta}
            if eta > 0 and generator is not None:
                step_kwargs["generator"] = generator
            x = sched.step(pred, t, x, **step_kwargs).prev_sample

        return x
