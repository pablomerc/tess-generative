"""Unconditional Flow Matching model for galaxy images.

This model generates galaxy images without any conditioning.
It uses FlowMatchingDecoder in unconditional mode.
"""

import torch
import torch.nn as nn
from galaxy_images.galaxy_flow.flowmatching_decoder import FlowMatchingDecoder
from galaxy_images.galaxy_flow import unconditional_config as cfg


class UnconditionalGalaxyFlow(nn.Module):
    """
    Unconditional Flow Matching model for galaxy images.

    This model generates galaxy images directly from noise without any conditioning.
    The images are expected to be preprocessed (cropped, clamped, rescaled, range compressed).
    """

    def __init__(
        self,
        image_size: int = None,
        output_dim: int = None,
        num_channels: int = None,
        velocity_field_type: str = None,
        use_film: bool = None,
        unet_channels: list = None,
        num_residual_layers: int = None,
        t_embed_dim: int = None,
        z_embed_dim: int = None,
        mlp_hidden_dims: list = None,
        n_integration_steps: int = None,
    ):
        super().__init__()

        # Use config defaults if not provided
        image_size = image_size or cfg.IMAGE_SIZE
        num_channels = num_channels if num_channels is not None else cfg.NUM_CHANNELS
        output_dim = output_dim or cfg.OUTPUT_DIM
        velocity_field_type = velocity_field_type or cfg.VELOCITY_FIELD_TYPE
        use_film = use_film if use_film is not None else cfg.USE_FILM
        unet_channels = unet_channels or cfg.UNET_CHANNELS
        num_residual_layers = num_residual_layers or cfg.NUM_RESIDUAL_LAYERS
        t_embed_dim = t_embed_dim or cfg.T_EMBED_DIM
        z_embed_dim = z_embed_dim or cfg.Z_EMBED_DIM
        mlp_hidden_dims = mlp_hidden_dims or cfg.MLP_HIDDEN_DIMS
        n_integration_steps = n_integration_steps or cfg.N_INTEGRATION_STEPS

        # Verify output_dim matches image_size and num_channels
        expected_output_dim = num_channels * image_size * image_size
        if output_dim != expected_output_dim:
            raise ValueError(
                f"output_dim {output_dim} must equal num_channels * image_size² = {expected_output_dim} "
                f"for UNetVelocityField. Got num_channels={num_channels}, image_size={image_size}"
            )

        # Create unconditional decoder (input_dim=0 means no conditioning)
        self.decoder = FlowMatchingDecoder(
            input_dim=0,  # No conditioning for unconditional model
            output_dim=output_dim,
            velocity_field_type=velocity_field_type,
            hidden_dims=mlp_hidden_dims,
            n_integration_steps=n_integration_steps,
            # U-Net specific parameters
            image_size=image_size,
            unet_channels=unet_channels,
            num_residual_layers=num_residual_layers,
            t_embed_dim=t_embed_dim,
            z_embed_dim=z_embed_dim,
            use_film=use_film,
            unconditional=True,  # Explicitly set unconditional mode
            num_channels=num_channels,  # Pass num_channels for multi-channel support
        )

        self.image_size = image_size
        self.num_channels = num_channels
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute flow matching loss.

        Args:
            x: Preprocessed galaxy images, shape (batch_size, output_dim)
               Images should be flattened and preprocessed (cropped, clamped, rescaled, range compressed)

        Returns:
            loss: Flow matching loss
        """
        # For unconditional model, pass z=None
        loss = self.decoder.get_loss(x, z=None)
        return loss

    def sample(self, batch_size: int, device: torch.device = None, n_samples: int = 1) -> torch.Tensor:
        """
        Generate samples from the unconditional model.

        Args:
            batch_size: Number of samples to generate
            device: Device to generate on (if None, uses model's device)
            n_samples: Number of samples per batch (for multi-sample generation)

        Returns:
            samples: Generated images, shape (batch_size, output_dim) or
                    (n_samples, batch_size, output_dim) if n_samples > 1
        """
        if device is None:
            device = next(self.parameters()).device

        # For unconditional sampling, pass z=None and specify batch_size and device
        samples = self.decoder.sample(
            z=None,
            batch_size=batch_size,
            device=device,
            n_samples=n_samples
        )

        return samples

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input images (for visualization/debugging).
        For unconditional model, this just generates new samples.

        Args:
            x: Input images, shape (batch_size, output_dim)

        Returns:
            reconstructions: Reconstructed images, shape (batch_size, output_dim)
        """
        # For unconditional model, we can't really "reconstruct" since there's no encoder
        # This just generates new samples with the same batch size
        batch_size = x.shape[0]
        device = x.device
        return self.sample(batch_size=batch_size, device=device, n_samples=1)


def build_model(device: str = None, **kwargs) -> UnconditionalGalaxyFlow:
    """
    Build an unconditional flow matching model with config defaults.

    Args:
        device: Device to move model to (e.g., 'cuda', 'cpu')
        **kwargs: Override any config parameters

    Returns:
        UnconditionalGalaxyFlow model
    """
    model = UnconditionalGalaxyFlow(**kwargs)

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    # Test model creation
    model = build_model(device='cpu')
    print(f"Model created successfully!")
    print(f"Image size: {model.image_size}")
    print(f"Output dim: {model.output_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, model.output_dim)
    loss = model(x)
    print(f"\nForward pass test:")
    print(f"Input shape: {x.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test sampling
    samples = model.sample(batch_size=4, device='cpu')
    print(f"\nSampling test:")
    print(f"Sample shape: {samples.shape}")

    # try with python -m galaxy_images.galaxy_flow.unconditional_model
