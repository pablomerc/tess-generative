"""
Encoder Architectures for Double Encoder Model

This module contains:
1. NumberEncoder: Encodes digit identity from same digit with different augmentation
2. FilterEncoder: Encodes augmentation style from different digit with same augmentation
Based on ConditionalConvVAE architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class NumberEncoder(nn.Module):
    """
    Number Encoder: Learns to encode digit identity from same digit with different augmentation
    Based on ConditionalConvVAE encoder architecture
    """
    def __init__(self, latent_dim=NUMBER_ENCODER_LATENT_DIM):
        super().__init__()

        # Encoder layers (based on ConditionalConvVAE)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),   # (B,64,28,28) - use 3x3 kernel with padding=1
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),  # (B,128,28,28) - use 3x3 kernel with padding=1
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space projection
        self.fc_mu = nn.Linear(128*28*28, latent_dim)
        self.fc_logvar = nn.Linear(128*28*28, latent_dim)

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


class FilterEncoder(nn.Module):
    """
    Filter Encoder: Learns to encode augmentation style from different digit with same augmentation
    Based on ConditionalConvVAE encoder architecture
    """
    def __init__(self, latent_dim=FILTER_ENCODER_LATENT_DIM):
        super().__init__()

        # Encoder layers (based on ConditionalConvVAE)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),   # (B,64,28,28) - use 3x3 kernel with padding=1
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),  # (B,128,28,28) - use 3x3 kernel with padding=1
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space projection
        self.fc_mu = nn.Linear(128*28*28, latent_dim)
        self.fc_logvar = nn.Linear(128*28*28, latent_dim)

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


class DoubleEncoder(nn.Module):
    """
    Combined model containing both encoders
    """
    def __init__(self, number_latent_dim=NUMBER_ENCODER_LATENT_DIM,
                 filter_latent_dim=FILTER_ENCODER_LATENT_DIM):
        super().__init__()

        self.number_encoder = NumberEncoder(number_latent_dim)
        self.filter_encoder = FilterEncoder(filter_latent_dim)

    def forward(self, same_digit, different_digit):
        """
        Forward pass through both encoders

        Args:
            same_digit: Same digit with different augmentation (for number encoder)
            different_digit: Different digit with same augmentation (for filter encoder)

        Returns:
            tuple: (number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar)
        """
        # Encode digit identity
        number_z, number_mu, number_logvar = self.number_encoder(same_digit)

        # Encode augmentation style
        filter_z, filter_mu, filter_logvar = self.filter_encoder(different_digit)

        return number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar


def test_encoders():
    """Test function to verify encoder architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Create test data
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28).to(device)

    # Test Number Encoder
    print("Testing Number Encoder...")
    number_encoder = NumberEncoder().to(device)
    number_z, number_mu, number_logvar = number_encoder(test_input)
    print(f"Number encoder output shapes:")
    print(f"  z: {number_z.shape}")
    print(f"  mu: {number_mu.shape}")
    print(f"  logvar: {number_logvar.shape}")

    # Test Filter Encoder
    print("\nTesting Filter Encoder...")
    filter_encoder = FilterEncoder().to(device)
    filter_z, filter_mu, filter_logvar = filter_encoder(test_input)
    print(f"Filter encoder output shapes:")
    print(f"  z: {filter_z.shape}")
    print(f"  mu: {filter_mu.shape}")
    print(f"  logvar: {filter_logvar.shape}")

    # Test Double Encoder
    print("\nTesting Double Encoder...")
    double_encoder = DoubleEncoder().to(device)
    number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = double_encoder(test_input, test_input)
    print(f"Double encoder output shapes:")
    print(f"  number_z: {number_z.shape}")
    print(f"  filter_z: {filter_z.shape}")
    print(f"  Combined latent dim: {number_z.shape[1] + filter_z.shape[1]}")

    # Test with actual triplet data
    print("\nTesting with triplet data...")
    from triplet_creation import TripletCreator
    triplet_creator = TripletCreator()
    ground_truth, different_digit, same_digit, original_labels, different_labels = triplet_creator.create_batch_triplets(2)

    ground_truth = ground_truth.to(device)
    different_digit = different_digit.to(device)
    same_digit = same_digit.to(device)

    number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = double_encoder(same_digit, different_digit)
    print(f"Triplet test output shapes:")
    print(f"  number_z: {number_z.shape}")
    print(f"  filter_z: {filter_z.shape}")

    print("\nAll encoder tests passed!")


if __name__ == "__main__":
    test_encoders()
