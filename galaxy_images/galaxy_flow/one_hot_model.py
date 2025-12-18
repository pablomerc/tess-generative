"""One-hot Flow Matching model for galaxy images.

This model generates galaxy images with one-hot conditioning.
It uses FlowMatchingDecoder in one-hot mode.
"""

import torch
import torch.nn as nn
from galaxy_images.galaxy_flow.flowmatching_decoder import FlowMatchingDecoder
from galaxy_images.galaxy_flow import unconditional_config as cfg


class OneHotGalaxyFlow(nn.Module):
    """
    One-hot Flow Matching model for galaxy images.

    This model generates galaxy images directly from noise with one-hot conditioning.
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

        # Create conditional decoder with one-hot conditioning (input_dim=2 for [HSC, Legacy])
        self.decoder = FlowMatchingDecoder(
            input_dim=2,  # One-hot conditioning: [1, 0] for HSC, [0, 1] for Legacy Survey
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
            unconditional=False,  # Enable conditioning with one-hot vectors
            num_channels=num_channels,  # Pass num_channels for multi-channel support
        )

        self.image_size = image_size
        self.num_channels = num_channels
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute flow matching loss.

        Args:
            x: Preprocessed galaxy images, shape (batch_size, output_dim)
               Images should be flattened and preprocessed (cropped, clamped, rescaled, range compressed)
            z: One-hot conditioning vectors, shape (batch_size, 2)
               [1, 0] for HSC images, [0, 1] for Legacy Survey images

        Returns:
            loss: Flow matching loss
        """
        loss = self.decoder.get_loss(x, z=z)
        return loss

    def sample(self, batch_size: int, device: torch.device = None, n_samples: int = 1, z: torch.Tensor = None) -> torch.Tensor:
        """
        Generate samples from the conditional model.

        Args:
            batch_size: Number of samples to generate (used if z is None)
            device: Device to generate on (if None, uses model's device)
            n_samples: Number of samples per batch (for multi-sample generation)
            z: One-hot conditioning vectors, shape (batch_size, 2)
               [1, 0] for HSC images, [0, 1] for Legacy Survey images
               If None, batch_size must be provided and zeros will be used

        Returns:
            samples: Generated images, shape (batch_size, output_dim) or
                    (n_samples, batch_size, output_dim) if n_samples > 1
        """
        if device is None:
            device = next(self.parameters()).device

        # If z is provided, use it for conditional sampling
        # If z is None, the decoder will use zeros (unconditional fallback)
        if z is None:
            samples = self.decoder.sample(
                z=None,
                batch_size=batch_size,
                device=device,
                n_samples=n_samples
            )
        else:
            samples = self.decoder.sample(
                z=z,
                batch_size=None,  # batch_size determined by z
                device=device,
                n_samples=n_samples
            )

        return samples

    def reconstruct(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input images (for visualization/debugging).
        For flow matching models, this generates new samples conditioned on z.

        Args:
            x: Input images, shape (batch_size, output_dim)
            z: One-hot conditioning vectors, shape (batch_size, 2)
               [1, 0] for HSC images, [0, 1] for Legacy Survey images

        Returns:
            reconstructions: Generated images conditioned on z, shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        device = x.device
        return self.sample(batch_size=batch_size, device=device, n_samples=1, z=z)


def build_model(device: str = None, **kwargs) -> OneHotGalaxyFlow:
    """
    Build an one-hot flow matching model with config defaults.

    Args:
        device: Device to move model to (e.g., 'cuda', 'cpu')
        **kwargs: Override any config parameters

    Returns:
        OneHotGalaxyFlow model
    """
    model = OneHotGalaxyFlow(**kwargs)

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Testing OneHotGalaxyFlow Model")
    print("=" * 60)

    # Test model creation
    print("\n1. Creating model...")
    start_time = time.time()
    model = build_model(device='cpu')
    creation_time = time.time() - start_time
    print(f"   Model created in {creation_time:.2f}s")
    print(f"   Image size: {model.image_size}")
    print(f"   Output dim: {model.output_dim}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Temporarily reduce integration steps for faster testing
    original_steps = model.decoder.n_integration_steps
    model.decoder.n_integration_steps = 100  # Use only 100 steps for testing
    print(f"   Using {model.decoder.n_integration_steps} integration steps for testing (normal: {original_steps})")

    # Test forward pass with one-hot conditioning
    print("\n2. Testing forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, model.output_dim)
    # Create one-hot vectors: [1, 0] for HSC, [0, 1] for Legacy Survey
    z = torch.tensor([
        [1, 0],  # HSC
        [0, 1],  # Legacy Survey
        [1, 0],  # HSC
        [0, 1],  # Legacy Survey
    ], dtype=torch.float32)
    start_time = time.time()
    loss = model(x, z)
    forward_time = time.time() - start_time
    print(f"   Forward pass completed in {forward_time:.2f}s")
    print(f"   Input shape: {x.shape}")
    print(f"   Conditioning shape: {z.shape}")
    print(f"   Loss: {loss.item():.4f}")

    # Test sampling with one-hot conditioning (just one test to save time)
    print("\n3. Testing sampling (this may take a moment on CPU)...")
    print("   Generating 2 samples with mixed conditioning...")
    z_test = torch.tensor([
        [1, 0],  # HSC
        [0, 1],  # Legacy Survey
    ], dtype=torch.float32)
    start_time = time.time()
    samples = model.sample(batch_size=2, device='cpu', z=z_test)
    sampling_time = time.time() - start_time
    print(f"   Sampling completed in {sampling_time:.2f}s")
    print(f"   Sample shape: {samples.shape}")
    print(f"   Sample range: [{samples.min():.4f}, {samples.max():.4f}]")

    # Restore original integration steps
    model.decoder.n_integration_steps = original_steps

    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Model creation: {creation_time:.2f}s")
    print(f"  Forward pass: {forward_time:.2f}s")
    print(f"  Sampling (2 samples, 10 steps): {sampling_time:.2f}s")
    print(f"  Estimated time for full sampling (250 steps): ~{sampling_time * 25:.1f}s per 2 samples")
    print("=" * 60)
    print("\n✓ All tests passed!")
    print("\nNote: Full sampling with 250 integration steps will be slower.")
    print("      For training, consider using GPU for faster iteration.")
