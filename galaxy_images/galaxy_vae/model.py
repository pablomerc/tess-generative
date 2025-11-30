"""
Complete Galaxy VAE models combining encoders and decoder.
"""

import torch
import torch.nn as nn
from .encoders import GalaxyNumberEncoder, GalaxyFilterEncoder
from .decoder import GalaxyDecoder
from .config import *
from .single_decoder import GalaxySingleDecoder

class DoubleEncoderGalaxyVAE(nn.Module):
    """
    Complete Double Encoder VAE for Galaxy Images.

    This model uses:
    1. GalaxyNumberEncoder: Encodes galaxy identity from same galaxy with different instrument
    2. GalaxyFilterEncoder: Encodes instrument characteristics from different galaxy with same instrument
    3. GalaxyDecoder: Reconstructs images from combined latents

    The training objective is to reconstruct the ground truth image from:
    - same_galaxy (different instrument) -> number encoder (galaxy identity)
    - different_galaxy (same instrument) -> filter encoder (instrument characteristics)
    """

    def __init__(self,
                 number_latent_dim=NUMBER_ENCODER_LATENT_DIM,
                 filter_latent_dim=FILTER_ENCODER_LATENT_DIM,
                 num_channels=NUM_CHANNELS):
        super().__init__()

        self.number_encoder = GalaxyNumberEncoder(number_latent_dim, num_channels)
        self.filter_encoder = GalaxyFilterEncoder(filter_latent_dim, num_channels)
        self.decoder = GalaxyDecoder(number_latent_dim, filter_latent_dim, num_channels)

    def forward(self, same_galaxy, different_galaxy):
        """
        Complete forward pass through the entire model.

        Args:
            same_galaxy: Same galaxy with different instrument (B, NUM_CHANNELS, H, W)
            different_galaxy: Different galaxy with same instrument (B, NUM_CHANNELS, H, W)

        Returns:
            tuple: (reconstruction, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar)
        """
        # Encode galaxy identity
        number_z, number_mu, number_logvar = self.number_encoder(same_galaxy)

        # Encode instrument characteristics
        filter_z, filter_mu, filter_logvar = self.filter_encoder(different_galaxy)

        # Decode combined representation
        reconstruction = self.decoder(number_z, filter_z)

        return (reconstruction, number_z, filter_z,
                number_mu, number_logvar, filter_mu, filter_logvar)

    def encode_only(self, same_galaxy, different_galaxy):
        """
        Only perform encoding (useful for analysis).
        """
        number_z, number_mu, number_logvar = self.number_encoder(same_galaxy)
        filter_z, filter_mu, filter_logvar = self.filter_encoder(different_galaxy)

        return number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar

    def decode_only(self, number_z, filter_z):
        """
        Only perform decoding (useful for generation).
        """
        return self.decoder(number_z, filter_z)

    def reconstruct(self, same_galaxy, different_galaxy):
        """
        Reconstruct ground truth from encodings (convenience method).
        """
        reconstruction, _, _, _, _, _, _ = self.forward(same_galaxy, different_galaxy)
        return reconstruction


class GalaxyVAE(nn.Module):
    """
    Simple single-encoder VAE for galaxy images (for comparison/testing).

    This is a simpler baseline that uses a single encoder-decoder pair.
    """

    def __init__(self, latent_dim=LATENT_DIM, num_channels=NUM_CHANNELS):
        super().__init__()

        # Simple encoder
        self.encoder = GalaxyNumberEncoder(latent_dim, num_channels)

        # Simple decoder (reuse GalaxyDecoder but with single latent)
        # Decoder expects combined_latent_dim = number_latent_dim + filter_latent_dim
        # So we set filter_latent_dim = 0 and pass only number_z
        self.decoder = GalaxyDecoder(latent_dim, 0, num_channels)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        # Pass None for filter_z, decoder will handle it
        reconstruction = self.decoder(z, None)
        return reconstruction, z, mu, logvar

class SingleEncoderGalaxyVAE(nn.Module):
    """
    Simple single-encoder VAE for galaxy images (for comparison/testing).
    """
    def __init__(self, latent_dim=LATENT_DIM, num_channels=NUM_CHANNELS):
        super().__init__()
        self.encoder = GalaxyNumberEncoder(latent_dim, num_channels)
        self.decoder = GalaxySingleDecoder(latent_dim, num_channels)
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z, mu, logvar

def test_models():
    """Test function to verify model architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    batch_size = 4
    test_same = torch.randn(batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    test_different = torch.randn(batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)

    print(f"Testing with input shapes: {test_same.shape}")


    # Test Single Encoder VAE
    print("\nTesting SingleEncoderGalaxyVAE...")
    model = SingleEncoderGalaxyVAE().to(device)
    reconstruction, z, mu, logvar = model(test_same)
    print(f"Model output shapes:")
    print(f"  reconstruction: {reconstruction.shape}")
    print(f"  z: {z.shape}")
    print(f"  mu: {mu.shape}")
    print(f"  logvar: {logvar.shape}")
    assert reconstruction.shape == (batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), \
        f"Reconstruction shape mismatch!"



    # Test Double Encoder VAE
    print("\nTesting DoubleEncoderGalaxyVAE...")
    model = DoubleEncoderGalaxyVAE().to(device)
    reconstruction, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
        model(test_same, test_different)


    print(f"Model output shapes:")
    print(f"  reconstruction: {reconstruction.shape}")
    print(f"  number_z: {number_z.shape}")
    print(f"  filter_z: {filter_z.shape}")

    assert reconstruction.shape == (batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), \
        f"Reconstruction shape mismatch!"


    # Test encode_only and decode_only
    print("\nTesting encode_only and decode_only...")
    # Use the z values from the forward pass to ensure consistency
    # (reparameterization uses random noise, so re-encoding would give different z values)
    reconstruction_test = model.decode_only(number_z, filter_z)

    assert torch.allclose(reconstruction, reconstruction_test, atol=1e-5), \
        "decode_only doesn't match forward pass!"

    print("\nAll model tests passed!")


if __name__ == "__main__":
    test_models()
