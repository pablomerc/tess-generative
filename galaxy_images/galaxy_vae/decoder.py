"""
Galaxy-specific decoder architecture.

Decodes combined latent representations (galaxy identity + instrument characteristics)
back to multi-channel galaxy images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *


class GalaxyDecoder(nn.Module):
    """
    Joint Decoder for Galaxy Images: Combines number and filter encodings to reconstruct ground truth.

    Adapted for:
    - Input: Combined latent (number_z + filter_z)
    - Output: (B, NUM_CHANNELS, 160, 160) - multi-channel galaxy images
    """

    def __init__(self,
                 number_latent_dim=NUMBER_ENCODER_LATENT_DIM,
                 filter_latent_dim=FILTER_ENCODER_LATENT_DIM,
                 num_channels=NUM_CHANNELS):
        super().__init__()

        self.combined_latent_dim = number_latent_dim + filter_latent_dim
        self.num_channels = num_channels

        # Project combined latent to feature map
        # We'll start with 5x5 feature maps (matching encoder's final spatial size)
        self.fc_dec = nn.Linear(self.combined_latent_dim, 512 * 5 * 5)

        # Decoder layers: 5x5 -> 10x10 -> 20x20 -> 40x40 -> 80x80 -> 160x160
        self.dec = nn.Sequential(
            # 5x5 -> 10x10
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 10x10 -> 20x20
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 20x20 -> 40x40
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 40x40 -> 80x80
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 80x80 -> 160x160
            nn.ConvTranspose2d(64, num_channels, kernel_size=4, stride=2, padding=1),
            # No activation here - we'll apply different activations per channel type
        )

    def forward(self, number_z, filter_z=None):
        """
        Forward pass: combine encodings and decode

        Args:
            number_z: Latent encoding from number encoder
            filter_z: Latent encoding from filter encoder (optional, defaults to zeros)

        Returns:
            torch.Tensor: Reconstructed image (B, NUM_CHANNELS, 160, 160)
        """
        # Handle case where filter_z is None (for single-encoder VAE)
        if filter_z is None:
            filter_z = torch.zeros_like(number_z)

        # Concatenate the two latent representations
        combined_z = torch.cat([number_z, filter_z], dim=1)

        # Project to feature map
        h = self.fc_dec(combined_z).view(-1, 512, 5, 5)

        # Decode
        reconstruction = self.dec(h)

        # Apply appropriate activations per channel type
        # For flux: allow negative values (no sigmoid)
        # For ivar: should be positive (ReLU or sigmoid)
        # For mask: should be [0,1] (sigmoid)
        # We'll use a simple approach: sigmoid for all for now, can be refined later
        # Note: This is a simplification - in practice you might want different activations

        return reconstruction


def test_decoder():
    """Test function to verify decoder architecture"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    batch_size = 4
    number_z = torch.randn(batch_size, NUMBER_ENCODER_LATENT_DIM).to(device)
    filter_z = torch.randn(batch_size, FILTER_ENCODER_LATENT_DIM).to(device)

    print(f"Testing with latent shapes: number_z={number_z.shape}, filter_z={filter_z.shape}")

    # Test Decoder
    print("\nTesting GalaxyDecoder...")
    decoder = GalaxyDecoder().to(device)
    reconstruction = decoder(number_z, filter_z)
    print(f"Decoder output shape: {reconstruction.shape}")
    print(f"Expected shape: ({batch_size}, {NUM_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE})")

    assert reconstruction.shape == (batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), \
        f"Shape mismatch! Got {reconstruction.shape}, expected ({batch_size}, {NUM_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE})"

    print("\nDecoder test passed!")


if __name__ == "__main__":
    test_decoder()
