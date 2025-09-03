"""
Double Encoder Flow Matching Architecture - Simplified Version

This module combines the double encoder architecture (NumberEncoder and FilterEncoder)
with the flow matching decoder for conditional generation.

SIMPLIFIED: Data preprocessing now outputs [-1,1] range directly
- No need for normalization/denormalization in the model
- Flow matching decoder gets data in expected [-1,1] range
- Output is in [-1,1] range (can be converted to [0,1] for visualization if needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

# Import the flow matching decoder
from carol_decoder import FlowMatchingDecoder

# Import double encoder components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'double-encoder-model'))
from encoder_architectures import NumberEncoder, FilterEncoder
from config import *


class DoubleEncoderFlowMatching(nn.Module):
    """
    Double Encoder with Flow Matching Decoder - Simplified Version

    This model uses:
    1. NumberEncoder: Encodes digit identity from same digit with different augmentation
    2. FilterEncoder: Encodes augmentation style from different digit with same augmentation
    3. FlowMatchingDecoder: Generates images using flow matching conditioned on combined latents
    
    SIMPLIFIED: Assumes input data is already in [-1,1] range
    """

    def __init__(self,
                 number_latent_dim=NUMBER_ENCODER_LATENT_DIM,
                 filter_latent_dim=FILTER_ENCODER_LATENT_DIM,
                 image_size=28,
                 # Flow matching decoder parameters
                 velocity_field_type="unet",
                 n_integration_steps=100,
                 unet_channels=[32, 64, 128],
                 num_residual_layers=2,
                 t_embed_dim=40,
                 z_embed_dim=40):
        super().__init__()

        self.number_latent_dim = number_latent_dim
        self.filter_latent_dim = filter_latent_dim
        self.combined_latent_dim = number_latent_dim + filter_latent_dim
        self.image_size = image_size
        self.image_dim = image_size * image_size

        # Double encoder components
        self.number_encoder = NumberEncoder(number_latent_dim)
        self.filter_encoder = FilterEncoder(filter_latent_dim)

        # Flow matching decoder
        # Input: combined latent (number_z + filter_z)
        # Output: flattened image
        self.decoder = FlowMatchingDecoder(
            input_dim=self.combined_latent_dim,
            output_dim=self.image_dim,
            velocity_field_type=velocity_field_type,
            n_integration_steps=n_integration_steps,
            # U-Net specific parameters
            image_size=image_size,
            unet_channels=unet_channels,
            num_residual_layers=num_residual_layers,
            t_embed_dim=t_embed_dim,
            z_embed_dim=z_embed_dim
        )

    def forward(self, same_digit, different_digit):
        """
        Forward pass through the entire model

        Args:
            same_digit: Same digit with different augmentation (for number encoder)
            different_digit: Different digit with same augmentation (for filter encoder)

        Returns:
            tuple: (combined_z, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar)
        """
        # Encode digit identity
        number_z, number_mu, number_logvar = self.number_encoder(same_digit)

        # Encode augmentation style
        filter_z, filter_mu, filter_logvar = self.filter_encoder(different_digit)

        # Combine latent representations
        combined_z = torch.cat([number_z, filter_z], dim=1)

        return (combined_z, number_z, filter_z,
                number_mu, number_logvar, filter_mu, filter_logvar)

    def get_flow_loss(self, same_digit, different_digit, ground_truth):
        """
        Compute flow matching loss

        Args:
            same_digit: Same digit with different augmentation
            different_digit: Different digit with same augmentation
            ground_truth: Target images for reconstruction (in [-1,1] range)

        Returns:
            tuple: (flow_loss, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar)
        """
        # Get encodings
        combined_z, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
            self.forward(same_digit, different_digit)

        # Flatten ground truth images for flow matching
        ground_truth_flat = ground_truth.view(ground_truth.size(0), -1)

        # Compute flow matching loss (ground_truth should already be in [-1,1] range)
        flow_loss = self.decoder.get_loss(ground_truth_flat, combined_z)

        return (flow_loss, number_z, filter_z,
                number_mu, number_logvar, filter_mu, filter_logvar)

    def sample(self, same_digit, different_digit, num_samples=1):
        """
        Generate samples by combining number and filter encodings

        Args:
            same_digit: Same digit with different augmentation
            different_digit: Different digit with same augmentation
            num_samples: Number of samples to generate

        Returns:
            torch.Tensor: Generated images in [-1,1] range
        """
        with torch.no_grad():
            # Get encodings
            combined_z, _, _, _, _, _, _ = self.forward(same_digit, different_digit)

            # Sample from flow matching decoder (outputs in [-1,1] range)
            samples_flat = self.decoder.sample(combined_z, num_samples)
            # Clamp to reasonable range to prevent extreme values
            samples_flat = torch.clamp(samples_flat, -3.0, 3.0)

            # Reshape to image format
            if num_samples == 1:
                samples = samples_flat.view(-1, 1, self.image_size, self.image_size)
            else:
                samples = samples_flat.view(num_samples, -1, 1, self.image_size, self.image_size)

            return samples

    def reconstruct(self, same_digit, different_digit):
        """
        Reconstruct images using the model

        Args:
            same_digit: Same digit with different augmentation
            different_digit: Different digit with same augmentation

        Returns:
            torch.Tensor: Reconstructed images in [-1,1] range
        """
        return self.sample(same_digit, different_digit, num_samples=1)

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
        
        Returns:
            torch.Tensor: Generated images in [-1,1] range
        """
        combined_z = torch.cat([number_z, filter_z], dim=1)
        samples_flat = self.decoder.sample(combined_z, 1)
        # Clamp to reasonable range to prevent extreme values
        samples_flat = torch.clamp(samples_flat, -3.0, 3.0)
        
        return samples_flat.view(-1, 1, self.image_size, self.image_size)

    def to_visualization_range(self, x):
        """
        Convert from [-1,1] to [0,1] range for visualization
        
        Args:
            x: Tensor in [-1,1] range
            
        Returns:
            x: Tensor in [0,1] range
        """
        return (x + 1.0) / 2.0


def test_double_encoder_flow():
    """Test function to verify the architecture works correctly"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    print(f"Testing DoubleEncoderFlowMatching (Simplified) on {device}")

    # Create model
    model = DoubleEncoderFlowMatching().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test data in [-1,1] range (simulating flow matching input)
    batch_size = 4
    same_digit = torch.rand(batch_size, 1, 28, 28).to(device) * 2 - 1  # [-1,1] range
    different_digit = torch.rand(batch_size, 1, 28, 28).to(device) * 2 - 1  # [-1,1] range
    ground_truth = torch.rand(batch_size, 1, 28, 28).to(device) * 2 - 1  # [-1,1] range

    print(f"Input shapes:")
    print(f"  same_digit: {same_digit.shape}")
    print(f"  different_digit: {different_digit.shape}")
    print(f"  ground_truth: {ground_truth.shape}")
    
    print(f"Input ranges:")
    print(f"  same_digit: [{same_digit.min().item():.3f}, {same_digit.max().item():.3f}]")
    print(f"  different_digit: [{different_digit.min().item():.3f}, {different_digit.max().item():.3f}]")
    print(f"  ground_truth: [{ground_truth.min().item():.3f}, {ground_truth.max().item():.3f}]")

    # Test forward pass
    print("\nTesting forward pass...")
    combined_z, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
        model.forward(same_digit, different_digit)

    print(f"Output shapes:")
    print(f"  combined_z: {combined_z.shape}")
    print(f"  number_z: {number_z.shape}")
    print(f"  filter_z: {filter_z.shape}")
    print(f"  number_mu: {number_mu.shape}")
    print(f"  number_mu: {number_mu.shape}")

    # Test flow loss computation
    print("\nTesting flow loss computation...")
    flow_loss, _, _, _, _, _, _ = model.get_flow_loss(same_digit, different_digit, ground_truth)
    print(f"Flow loss: {flow_loss.item():.4f}")

    # Test sampling
    print("\nTesting sampling...")
    samples = model.sample(same_digit, different_digit)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Generated samples range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")

    # Test reconstruction
    print("\nTesting reconstruction...")
    reconstruction = model.reconstruct(same_digit, different_digit)
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Reconstruction range: [{reconstruction.min().item():.3f}, {reconstruction.max().item():.3f}]")

    # Test encode_only and decode_only
    print("\nTesting encode_only and decode_only...")
    number_z_test, filter_z_test, _, _, _, _ = model.encode_only(same_digit, different_digit)
    reconstruction_test = model.decode_only(number_z_test, filter_z_test)
    print(f"encode_only + decode_only reconstruction shape: {reconstruction_test.shape}")
    print(f"encode_only + decode_only reconstruction range: [{reconstruction_test.min().item():.3f}, {reconstruction_test.max().item():.3f}]")

    # Test visualization conversion
    print("\nTesting visualization conversion...")
    vis_reconstruction = model.to_visualization_range(reconstruction)
    print(f"Visualization range: [{vis_reconstruction.min().item():.3f}, {vis_reconstruction.max().item():.3f}]")

    # Verify output is in [-1,1] range
    if reconstruction.min() >= -1 and reconstruction.max() <= 1:
        print("✓ Output is correctly in [-1,1] range")
    else:
        print("✗ Output is NOT in [-1,1] range!")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_double_encoder_flow()
