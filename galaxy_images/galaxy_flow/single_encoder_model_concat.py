"""
Single Encoder Flow Matching Architecture (Concat Variant) for Galaxy Images

This model uses direct channel-concatenation conditioning:
- No encoder
- No latent z
- Conditioning is done by concatenating the conditioning images along the channel dimension.
"""

import torch
import torch.nn as nn

from galaxy_images.galaxy_flow.flowmatching_decoder_concat import FlowMatchingDecoderConcat
from galaxy_images.galaxy_flow import single_encoder_config as cfg


class SingleEncoderGalaxyFlowConcat(nn.Module):
    """
    Flow Matching model for galaxy images with channel-concatenation conditioning.

    - Target: preprocessed galaxy images (flattened)
    - Conditioning: counterfactual images, concatenated along channel dimension in the decoder
    """

    def __init__(
        self,
        image_size: int = None,
        output_dim: int = None,
        num_channels: int = None,
        # U-Net params
        concat_unet_channels: list = None,
        num_residual_layers: int = None,
        t_embed_dim: int = None,
        n_integration_steps: int = None,
        cond_channels: int = None,
    ):
        super().__init__()

        image_size = image_size or cfg.IMAGE_SIZE
        num_channels = num_channels if num_channels is not None else cfg.NUM_CHANNELS
        output_dim = output_dim or cfg.OUTPUT_DIM

        # Concat-specific config
        concat_unet_channels = concat_unet_channels or getattr(cfg, "CONCAT_UNET_CHANNELS", [64, 128, 256])
        num_residual_layers = num_residual_layers or cfg.NUM_RESIDUAL_LAYERS
        t_embed_dim = t_embed_dim or cfg.T_EMBED_DIM
        n_integration_steps = n_integration_steps or cfg.N_INTEGRATION_STEPS
        cond_channels = cond_channels if cond_channels is not None else getattr(
            cfg, "CONCAT_COND_CHANNELS", num_channels
        )

        self.image_size = image_size
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.cond_channels = cond_channels

        # For compatibility with existing training code that may inspect this:
        self.decoder_type = "concat"

        self.decoder = FlowMatchingDecoderConcat(
            output_dim=output_dim,
            image_size=image_size,
            num_channels=num_channels,
            cond_channels=cond_channels,
            unet_channels=concat_unet_channels,
            num_residual_layers=num_residual_layers,
            time_embed_dim=t_embed_dim,
            n_integration_steps=n_integration_steps,
        )

    def forward(self, x: torch.Tensor, cond_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute flow matching loss.

        Args:
            x: target images, flattened to (B, output_dim)
            cond_images: conditioning images (B, cond_channels, H, W)
        """
        return self.decoder.get_loss(x, cond_images=cond_images)

    def sample(
        self,
        cond_images: torch.Tensor,
        device: torch.device = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate samples from the concat model.

        Args:
            cond_images: conditioning images (B, cond_channels, H, W)
            device: target device (optional, inferred from model if None)
            n_samples: number of samples per conditioning image
        """
        if device is None:
            device = next(self.parameters()).device

        cond_images = cond_images.to(device)
        samples = self.decoder.sample(cond_images=cond_images, n_samples=n_samples)
        return samples


def build_model_concat(device: str = None, **kwargs) -> SingleEncoderGalaxyFlowConcat:
    """
    Build a concat-only flow matching model for galaxy images.

    Args:
        device: Device to move model to (e.g., 'cuda', 'cpu')
        **kwargs: Override any config parameters

    Returns:
        SingleEncoderGalaxyFlowConcat model
    """
    model = SingleEncoderGalaxyFlowConcat(**kwargs)

    if device is not None:
        model = model.to(device)

    return model


