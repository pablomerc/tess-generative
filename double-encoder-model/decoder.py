"""
Joint Decoder for Double Encoder Model

This module contains the decoder that combines the outputs from both encoders
(number encoder and filter encoder) to reconstruct the ground truth image.
Based on ConditionalConvVAE decoder architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class JointDecoder(nn.Module):
    """
    Joint Decoder: Combines number and filter encodings to reconstruct ground truth
    Based on ConditionalConvVAE decoder architecture
    """
    def __init__(self, number_latent_dim=NUMBER_ENCODER_LATENT_DIM,
                 filter_latent_dim=FILTER_ENCODER_LATENT_DIM):
        super().__init__()

        # Combined latent dimension
        self.combined_latent_dim = number_latent_dim + filter_latent_dim

        # Decoder layers (based on ConditionalConvVAE decoder)
        self.fc_dec = nn.Linear(self.combined_latent_dim, 128*28*28)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, 1), # (B,64,28,28) - use 3x3 kernel
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, 1, 1),  # (B,1,28,28) - use 3x3 kernel
            nn.Sigmoid()
        )

    def forward(self, number_z, filter_z):
        """
        Forward pass: combine encodings and decode

        Args:
            number_z: Latent encoding from number encoder
            filter_z: Latent encoding from filter encoder

        Returns:
            torch.Tensor: Reconstructed image
        """
        # Concatenate the two latent representations
        combined_z = torch.cat([number_z, filter_z], dim=1)

        # Decode
        h = self.fc_dec(combined_z).view(-1, 128, 28, 28)
        reconstruction = self.dec(h)

        return reconstruction


class DoubleEncoderDecoder(nn.Module):
    """
    Complete model combining both encoders and the joint decoder
    """
    def __init__(self, number_latent_dim=NUMBER_ENCODER_LATENT_DIM,
                 filter_latent_dim=FILTER_ENCODER_LATENT_DIM):
        super().__init__()

        # Import encoders
        from encoder_architectures import NumberEncoder, FilterEncoder

        self.number_encoder = NumberEncoder(number_latent_dim)
        self.filter_encoder = FilterEncoder(filter_latent_dim)
        self.decoder = JointDecoder(number_latent_dim, filter_latent_dim)

    def forward(self, same_digit, different_digit):
        """
        Complete forward pass through the entire model

        Args:
            same_digit: Same digit with different augmentation (for number encoder)
            different_digit: Different digit with same augmentation (for filter encoder)

        Returns:
            tuple: (reconstruction, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar)
        """
        # Encode digit identity
        number_z, number_mu, number_logvar = self.number_encoder(same_digit)

        # Encode augmentation style
        filter_z, filter_mu, filter_logvar = self.filter_encoder(different_digit)

        # Decode combined representation
        reconstruction = self.decoder(number_z, filter_z)

        return (reconstruction, number_z, filter_z,
                number_mu, number_logvar, filter_mu, filter_logvar)

    def encode_only(self, same_digit, different_digit):
        """
        Only perform encoding (useful for analysis)
        """
        number_z, number_mu, number_logvar = self.number_encoder(same_digit)
        filter_z, filter_mu, filter_logvar = self.filter_encoder(different_digit)

        return number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar

    def decode_only(self, number_z, filter_z):
        """
        Only perform decoding (useful for generation)
        """
        return self.decoder(number_z, filter_z)


def test_decoder():
    """Test function to verify decoder architecture"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Create test data
    batch_size = 4
    number_z = torch.randn(batch_size, NUMBER_ENCODER_LATENT_DIM).to(device)
    filter_z = torch.randn(batch_size, FILTER_ENCODER_LATENT_DIM).to(device)

    # Test Joint Decoder
    print("Testing Joint Decoder...")
    decoder = JointDecoder().to(device)
    reconstruction = decoder(number_z, filter_z)
    print(f"Decoder output shape: {reconstruction.shape}")
    print(f"Expected shape: ({batch_size}, 1, 28, 28)")

    # Test complete model
    print("\nTesting Complete Double Encoder Decoder...")
    model = DoubleEncoderDecoder().to(device)

    # Create dummy input images
    same_digit = torch.randn(batch_size, 1, 28, 28).to(device)
    different_digit = torch.randn(batch_size, 1, 28, 28).to(device)

    reconstruction, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = model(same_digit, different_digit)

    print(f"Complete model output shapes:")
    print(f"  reconstruction: {reconstruction.shape}")
    print(f"  number_z: {number_z.shape}")
    print(f"  filter_z: {filter_z.shape}")
    print(f"  number_mu: {number_mu.shape}")
    print(f"  filter_mu: {filter_mu.shape}")

    # Test encode_only and decode_only
    print("\nTesting encode_only and decode_only...")
    number_z_test, filter_z_test, _, _, _, _ = model.encode_only(same_digit, different_digit)
    reconstruction_test = model.decode_only(number_z_test, filter_z_test)

    print(f"encode_only + decode_only reconstruction shape: {reconstruction_test.shape}")
    print(f"Direct forward pass reconstruction shape: {reconstruction.shape}")

    # Check if they're the same
    if torch.allclose(reconstruction, reconstruction_test, atol=1e-6):
        print("✓ encode_only + decode_only matches direct forward pass")
    else:
        print("✗ encode_only + decode_only does not match direct forward pass")

    # Test with actual triplet data
    print("\nTesting with triplet data...")
    from triplet_creation import TripletCreator
    triplet_creator = TripletCreator()
    ground_truth, different_digit, same_digit, _, _ = triplet_creator.create_batch_triplets(2)

    ground_truth = ground_truth.to(device)
    different_digit = different_digit.to(device)
    same_digit = same_digit.to(device)

    reconstruction, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = model(same_digit, different_digit)
    print(f"Triplet test output shapes:")
    print(f"  reconstruction: {reconstruction.shape}")
    print(f"  number_z: {number_z.shape}")
    print(f"  filter_z: {filter_z.shape}")

    print("\nAll decoder tests passed!")


if __name__ == "__main__":
    test_decoder()
