"""
Galaxy-specific encoder architectures.

These encoders are adapted for:
- Multi-channel input (12-13 channels: flux, ivar, mask, optionally object_mask)
- Larger image size (160x160 vs 28x28)
- Astronomical image characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *


class GalaxyNumberEncoder(nn.Module):
    """
    Number Encoder for Galaxy Images: Encodes galaxy identity from same galaxy with different instrument.

    Adapted for:
    - Input: (B, NUM_CHANNELS, 160, 160) - multi-channel galaxy images
    - Output: latent representation of galaxy identity
    """

    def __init__(self, latent_dim=NUMBER_ENCODER_LATENT_DIM, num_channels=NUM_CHANNELS):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels

        # Encoder architecture for 160x160 images
        # Use strided convolutions to reduce spatial dimensions
        self.enc = nn.Sequential(
            # First block: 160x160 -> 80x80
            nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Second block: 80x80 -> 40x40
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Third block: 40x40 -> 20x20
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Fourth block: 20x20 -> 10x10
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Final block: 10x10 -> 5x5
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Flatten()
        )

        # Latent space projection
        # After 5x5 spatial size: 512 * 5 * 5 = 12800
        self.fc_mu = nn.Linear(512 * 5 * 5, latent_dim)
        self.fc_logvar = nn.Linear(512 * 5 * 5, latent_dim)

    def encode(self, x):
        """Encode input to latent representation"""
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass: encode and reparameterize"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class GalaxyFilterEncoder(nn.Module):
    """
    Filter Encoder for Galaxy Images: Encodes instrument characteristics from different galaxy with same instrument.

    Adapted for:
    - Input: (B, NUM_CHANNELS, 160, 160) - multi-channel galaxy images
    - Output: latent representation of instrument characteristics
    """

    def __init__(self, latent_dim=FILTER_ENCODER_LATENT_DIM, num_channels=NUM_CHANNELS):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels

        # Same architecture as NumberEncoder (they encode different aspects)
        self.enc = nn.Sequential(
            # First block: 160x160 -> 80x80
            nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Second block: 80x80 -> 40x40
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Third block: 40x40 -> 20x20
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Fourth block: 20x20 -> 10x10
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Final block: 10x10 -> 5x5
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Flatten()
        )

        # Latent space projection
        self.fc_mu = nn.Linear(512 * 5 * 5, latent_dim)
        self.fc_logvar = nn.Linear(512 * 5 * 5, latent_dim)

    def encode(self, x):
        """Encode input to latent representation"""
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass: encode and reparameterize"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


def test_encoders():
    """Test function to verify encoder architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data matching galaxy image format
    batch_size = 32
    num_channels = NUM_CHANNELS
    test_input = torch.randn(batch_size, num_channels, IMAGE_SIZE, IMAGE_SIZE).to(device)

    print(f"Testing with input shape: {test_input.shape}")

    # Test Number Encoder
    print("\nTesting GalaxyNumberEncoder...")
    number_encoder = GalaxyNumberEncoder().to(device)
    number_z, number_mu, number_logvar = number_encoder(test_input)
    print(f"Number encoder output shapes:")
    print(f"  z: {number_z.shape}")
    print(f"  mu: {number_mu.shape}")
    print(f"  logvar: {number_logvar.shape}")

    # Test Filter Encoder
    print("\nTesting GalaxyFilterEncoder...")
    filter_encoder = GalaxyFilterEncoder().to(device)
    filter_z, filter_mu, filter_logvar = filter_encoder(test_input)
    print(f"Filter encoder output shapes:")
    print(f"  z: {filter_z.shape}")
    print(f"  mu: {filter_mu.shape}")
    print(f"  logvar: {filter_logvar.shape}")

    print("\nAll encoder tests passed!")


if __name__ == "__main__":
    test_encoders()
