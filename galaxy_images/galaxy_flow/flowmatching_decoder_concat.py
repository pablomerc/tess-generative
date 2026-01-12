import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEncoder(nn.Module):
    """Time embedding using Fourier features."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,)
        Returns:
            embeddings: (batch_size, dim)
        """
        t = t.view(-1, 1)  # (batch_size, 1)
        t = torch.clamp(t, 0.0, 1.0)

        freqs = t * self.weights * 2 * math.pi  # (batch_size, half_dim)
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        result = torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)

        if torch.isnan(result).any() or torch.isinf(result).any():
            print("WARNING: NaN/Inf in FourierEncoder. Replaced with finite values.")
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

        return result


class ResidualBlockTimeOnly(nn.Module):
    """
    Simple residual block:
      - 2 x (Conv2d + BatchNorm + SiLU)
      - Adds a projected time embedding, no FiLM, no z conditioning.
    """

    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # Project time embedding to channel dimension
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels),
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            t_embed: (B, time_embed_dim)
        """
        residual = x

        x = self.block1(x)

        # Add time conditioning
        t_out = self.time_proj(t_embed).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        x = x + t_out

        x = self.block2(x)
        x = x + residual
        return x


class EncoderBlock(nn.Module):
    """U-Net encoder block with time-only residual blocks."""

    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, time_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualBlockTimeOnly(channels_in, time_embed_dim) for _ in range(num_residual_layers)]
        )
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed)
        x = self.downsample(x)
        return x


class MidBlock(nn.Module):
    """Middle block of U-Net, time-only residual blocks."""

    def __init__(self, channels: int, num_residual_layers: int, time_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualBlockTimeOnly(channels, time_embed_dim) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed)
        return x


class DecoderBlock(nn.Module):
    """U-Net decoder block with time-only residual blocks."""

    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, time_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlockTimeOnly(channels_out, time_embed_dim) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for block in self.res_blocks:
            x = block(x, t_embed)
        return x


class UNetVelocityFieldConcat(nn.Module):
    """
    U-Net velocity field for image-like data with **channel concatenation** conditioning.

    - No latent z
    - No FiLM
    - Conditioning is purely via concatenating conditioning images along the channel dimension.
    """

    def __init__(
        self,
        output_dim: int,
        image_size: int = 28,
        num_channels: int = 1,
        cond_channels: int = 4,
        channels: List[int] = [32, 64, 128],
        num_residual_layers: int = 2,
        time_embed_dim: int = 40,
    ):
        super().__init__()

        self.image_size = image_size
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.cond_channels = cond_channels

        expected_dim = num_channels * image_size * image_size
        if output_dim != expected_dim:
            raise ValueError(f"output_dim {output_dim} must equal num_channels * image_size² = {expected_dim}")

        # Initial conv sees target + conditioning channels
        in_ch = num_channels + cond_channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        self.time_encoder = FourierEncoder(time_embed_dim)

        # U-Net encoder / decoder
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(EncoderBlock(curr_c, next_c, num_residual_layers, time_embed_dim))
            decoders.append(DecoderBlock(next_c, curr_c, num_residual_layers, time_embed_dim))

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.mid = MidBlock(channels[-1], num_residual_layers, time_embed_dim)
        self.final_conv = nn.Conv2d(channels[0], num_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, output_dim)  - flattened target image
            t: (B,) or scalar   - time
            cond_images: (B, cond_channels, H, W) - conditioning images
        Returns:
            velocity: (B, output_dim)
        """
        batch_size = x.shape[0]

        # Reshape target
        x = x.view(batch_size, self.num_channels, self.image_size, self.image_size)

        # Channel concatenation conditioning
        x = torch.cat([x, cond_images], dim=1)  # (B, num_channels + cond_channels, H, W)

        # Handle time
        if t.dim() == 0:
            t = t.expand(batch_size)
        elif t.shape[0] != batch_size:
            t = t.expand(batch_size)

        t_embed = self.time_encoder(t)  # (B, time_embed_dim)

        # U-Net
        x = self.init_conv(x)
        residuals = []
        for enc in self.encoders:
            x = enc(x, t_embed)
            residuals.append(x.clone())

        x = self.mid(x, t_embed)

        for dec in self.decoders:
            res = residuals.pop()
            x = x + res
            x = dec(x, t_embed)

        x = self.final_conv(x)  # (B, num_channels, H, W)
        x = x.view(batch_size, self.output_dim)
        return x


class FlowMatchingDecoderConcat(nn.Module):
    """
    Flow Matching decoder with **channel concatenation** image conditioning.

    - Always uses a UNetVelocityFieldConcat
    - No latent z
    - Conditioning is via `cond_images` only
    """

    def __init__(
        self,
        output_dim: int,
        image_size: int = 28,
        num_channels: int = 1,
        cond_channels: int = 4,
        unet_channels: List[int] = [32, 64, 128],
        num_residual_layers: int = 2,
        time_embed_dim: int = 40,
        n_integration_steps: int = 100,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.num_channels = num_channels
        self.cond_channels = cond_channels
        self.n_integration_steps = n_integration_steps

        self.vector_field = UNetVelocityFieldConcat(
            output_dim=output_dim,
            image_size=image_size,
            num_channels=num_channels,
            cond_channels=cond_channels,
            channels=unet_channels,
            num_residual_layers=num_residual_layers,
            time_embed_dim=time_embed_dim,
        )

    @staticmethod
    def get_conditional_flow(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation: x_t = t * x1 + (1 - t) * x0."""
        return t.unsqueeze(-1) * x1 + (1.0 - t.unsqueeze(-1)) * x0

    def compute_cfm_loss(self, x1: torch.Tensor, cond_images: torch.Tensor) -> torch.Tensor:
        """
        Conditional Flow Matching loss with image conditioning.

        Args:
            x1: (B, output_dim) - target sample (flattened)
            cond_images: (B, cond_channels, H, W) - conditioning images
        """
        batch_size = x1.shape[0]
        device = x1.device

        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=device)
        x_t = self.get_conditional_flow(x0, x1, t)
        v_target = x1 - x0

        v_pred = self.vector_field(x_t, t, cond_images)
        loss = F.mse_loss(v_pred, v_target)
        return loss

    def sample_ode(
        self,
        cond_images: torch.Tensor,
        n_samples: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample by integrating the learned ODE from noise to data, conditioned on images.

        Args:
            cond_images: (B, cond_channels, H, W)
            n_samples: number of stochastic samples per conditioning image
        Returns:
            x_T:
                - shape (B, output_dim) if n_samples == 1
                - shape (n_samples, B, output_dim) if n_samples > 1
        """
        device = cond_images.device
        batch_size = cond_images.shape[0]

        if n_samples == 1:
            actual_batch = batch_size
            cond_rep = cond_images
        else:
            actual_batch = n_samples * batch_size
            cond_rep = cond_images.unsqueeze(0).expand(n_samples, -1, -1, -1, -1)
            cond_rep = cond_rep.reshape(
                -1,
                cond_images.shape[1],
                cond_images.shape[2],
                cond_images.shape[3],
            )

        x = torch.randn(actual_batch, self.output_dim, device=device, generator=generator)
        dt = 1.0 / self.n_integration_steps

        for i in range(self.n_integration_steps):
            t = torch.full((actual_batch,), i * dt, device=device)
            v = self.vector_field(x, t, cond_rep)
            x = x + dt * v

        if n_samples > 1:
            x = x.view(n_samples, batch_size, self.output_dim)
        return x

    def sample(
        self,
        cond_images: torch.Tensor,
        n_samples: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Convenience wrapper around `sample_ode`."""
        return self.sample_ode(cond_images=cond_images, n_samples=n_samples, generator=generator)

    def get_loss(self, x: torch.Tensor, cond_images: torch.Tensor) -> torch.Tensor:
        """Public loss API used during training."""
        return self.compute_cfm_loss(x1=x, cond_images=cond_images)


