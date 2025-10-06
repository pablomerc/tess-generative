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
import math
from typing import List, Tuple, Optional

# Import the flow matching decoder (relative import inside package)
try:
    from .carol_decoder import FlowMatchingDecoder
except ImportError:  # script execution fallback
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
                 z_embed_dim=40,
                 use_film: bool = True,
                 multi_samples: bool = False):
        super().__init__()

        self.number_latent_dim = number_latent_dim
        self.filter_latent_dim = filter_latent_dim
        self.combined_latent_dim = number_latent_dim + filter_latent_dim
        self.image_size = image_size
        self.image_dim = image_size * image_size
        self.use_film = use_film

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
            z_embed_dim=z_embed_dim,
            use_film=use_film
        )

        if multi_samples:
            use_attention = False
            use_concatenation = False
            if use_attention:

                self.num_aggregator = AttentionPooling(
                    dim=number_latent_dim,
                    n_hidden=128
                )
                self.filter_aggregator = AttentionPooling(
                    dim=filter_latent_dim,
                    n_hidden=128
                )

            elif use_concatenation:
                self.num_aggregator = Concatenator()
                self.filter_aggregator = Concatenator()
            else:
                print('DEBUG: Using DeepSets with MLP+Mean')
                self.num_aggregator = DeepSets(
                    in_dim=number_latent_dim,
                    out_dim=number_latent_dim,
                    pooling="mean")

                self.filter_aggregator=DeepSets(
                    in_dim=filter_latent_dim,
                    rho_hidden=128,
                    out_dim=filter_latent_dim,
                    pooling="mean")




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

    def multi_sample_encoding(
        self,
        same_number_augments: torch.Tensor,
        same_filter_augments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode augmentation sets, pool them, and concatenate the pooled latents.

        Args:
            same_number_augments: Tensor of shape [B, N, C, H, W] in [-1, 1]; each slice along N
                is an augmentation of the same digit. Gradients should flow through all entries.
            same_filter_augments: Tensor of shape [B, F, C, H, W] in [-1, 1]; each slice along F
                shares the filter/augmentation but comes from a different digit.

        Returns:
            tuple: (combined_z, pooled_number_z, pooled_filter_z) where each pooled latent is
            produced via sum-based DeepSets aggregation over its augmentations.

        Notes:
            This method reuses self.number_encoder and self.filter_encoder on each augmentation, then
            applies the specified permutation-invariant pooling to obtain a single latent per set.
        """

        if not hasattr(self, "num_aggregator") or not hasattr(self, "filter_aggregator"):
            raise RuntimeError(
                "multi_sample_encoding requires constructing DoubleEncoderFlowMatching with "
                "multi_samples=True."
            )

        B, N, C, H, W = same_number_augments.shape
        _, F, _, _, _ = same_filter_augments.shape

        number_flat = same_number_augments.view(B * N, C, H, W)
        filter_flat = same_filter_augments.view(B * F, C, H, W)

        number_z_flat, _, _ = self.number_encoder(number_flat)
        filter_z_flat, _, _ = self.filter_encoder(filter_flat)

        number_z = number_z_flat.view(B, N, -1)
        filter_z = filter_z_flat.view(B, F, -1)

        # TODO: Support variable augmentation counts by tracking lengths per example.
        pooled_number_z = self.num_aggregator(number_z)
        pooled_filter_z = self.filter_aggregator(filter_z)

        combined_z = torch.cat([pooled_number_z, pooled_filter_z], dim=1)
        return combined_z, pooled_number_z, pooled_filter_z

    def get_flow_loss_multi(self, same_number_augments, same_filter_augments, ground_truth):
        """
        Get flow matching loss for multi-sample encoding
        Args:
            same_number_augments: Tensor of shape [B, N, C, H, W] in [-1, 1]; each slice along N
            same_filter_augments: Tensor of shape [B, F, C, H, W] in [-1, 1]; each slice along F
            ground_truth: Tensor of shape [B, C, H, W] in [-1, 1]

        Returns:
            flow_loss: Tensor of shape [B]
        """
        combined_z, number_combined_z, filter_combined_z = self.multi_sample_encoding(same_number_augments, same_filter_augments)

        #Flatten ground truth images for flow matching
        ground_truth_flat = ground_truth.view(ground_truth.size(0), -1)

        flow_loss = self.decoder.get_loss(ground_truth_flat, combined_z)
        return flow_loss


class AttentionHead(nn.Module):
    """Single-head self-attention with a learnable CLS token."""
    def __init__(self, dim: int, n_hidden: int) -> None:
        """
        Args:
            dim: Feature width of each latent token.
            n_hidden: Size of internal projections (Q/K/V) and the pooled output.
        """

        super().__init__()

        self.W_K = nn.Linear(dim, n_hidden) # W_K weight matrix
        self.W_Q = nn.Linear(dim, n_hidden) # W_Q weight matrix
        self.W_V = nn.Linear(dim, n_hidden) # W_V weight matrix
        self.n_hidden = n_hidden
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.scale = math.sqrt(n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pools a latent sequence with attention.

        Args:
            x: Tensor shaped `[batch, num_tokens, dim]`.

        Returns:
            Pooled embedding `[batch, 1, n_hidden]`.
        """
        B,N,_ = x.shape
        device=x.device

        cls_token = self.cls_token.expand(B, -1, -1).to(device)

        x = torch.cat([cls_token, x], dim=1)
        x = x.to(device) # Ensure x is on the correct device

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        QK = torch.matmul(Q,K.transpose(-2,-1))/self.scale

        alpha = torch.nn.functional.softmax(QK, dim=-1)

        out = torch.matmul(alpha, V)

        pooled_output = out[:,:1]

        return pooled_output

class AttentionPooling(nn.Module):
    """Attention-based set pooling followed by projection."""
    def __init__(self, dim: int, n_hidden: int) -> None:
        """
        Args:
            dim: Output embedding size (matches input token width).
            n_hidden: Hidden size used inside the attention head.
        """
        super().__init__()

        self.dim=dim
        self.n_hidden=n_hidden
        self.head = AttentionHead(dim=dim, n_hidden=n_hidden)
        self.proj = nn.Linear(n_hidden,dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compresses a token set to a single latent.

        Args:
            x: Tensor `[batch, num_tokens, dim]`.

        Returns:
            Tensor `[batch, 1, dim]` containing the pooled latent.
        """

        x_pooled = self.head(x)
        x_pooled = self.proj(x_pooled)

        return x_pooled.squeeze(1) # [batch,dim]



class DeepSets(nn.Module):
    def __init__(self, in_dim, phi_hidden=128, set_emb_dim=128, rho_hidden=128, out_dim=1,
                 pooling="sum"):
        super().__init__()
        # φ: elementwise network (3 layers with ReLU)
        self.phi = nn.Sequential(
            nn.Linear(in_dim, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, set_emb_dim)  # this is φ(x) ∈ R^{set_emb_dim}
        )
        # ρ: post-pool network (3 layers with ReLU)
        self.rho = nn.Sequential(
            nn.Linear(set_emb_dim, rho_hidden), nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden), nn.ReLU(),
            nn.Linear(rho_hidden, out_dim)
        )
        assert pooling in {"sum", "mean", "max"}
        self.pooling = pooling


    def forward(self, x, lengths=None):
        """
        x: (B, N_max, d) padded batch of sets
        lengths: (B,) number of valid elements in each set (no lengths => all N are valid)
        """
        B, N_max, d = x.shape

        # Build mask for valid elements
        if lengths is None:
            mask = torch.ones(B, N_max, device=x.device, dtype=torch.bool)
        else:
            idx = torch.arange(N_max, device=x.device)[None, :].expand(B, N_max)
            mask = idx < lengths[:, None]  # True where valid

        # Apply φ elementwise
        x_flat = x.view(B * N_max, d)
        phi_flat = self.phi(x_flat)                        # (B*N_max, set_emb_dim)
        phi = phi_flat.view(B, N_max, -1)                  # (B, N_max, set_emb_dim)

        # Zero-out padded positions
        phi = phi * mask.unsqueeze(-1)                     # broadcast over emb dim

        # Pool across set elements (invariant)
        if self.pooling == "sum":
            set_emb = phi.sum(dim=1)                      # (B, set_emb_dim)
        elif self.pooling == "mean":
            denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(phi.dtype)
            set_emb = phi.sum(dim=1) / denom
        else:  # max
            # replace invalid with very negative before max
            very_neg = torch.finfo(phi.dtype).min
            phi_masked = phi.masked_fill(~mask.unsqueeze(-1), very_neg)
            set_emb = phi_masked.max(dim=1).values

        # Apply ρ
        out = self.rho(set_emb)                            # (B, out_dim)
        return out



class Concatenator(nn.Module):
    """Concatenate the per-latent vectors into a single vector per example.

    Expects an input tensor shaped [B, N_latents, latent_dim] and returns
    [B, N_latents * latent_dim]. Uses `.flatten(start_dim=1)` to safely handle
    non-contiguous inputs.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, N_latents, latent_dim]
        Returns:
            Tensor of shape [B, N_latents * latent_dim]
        """
        if x.dim() != 3:
            raise ValueError(
                f"Concatenator expects a 3D tensor [B, N_latents, latent_dim], got {tuple(x.shape)}"
            )
        return x.flatten(start_dim=1)

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


def test_multi_sample_encoding():
    """Quick smoke test for multi-sample aggregation support."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    print(f"Testing multi-sample aggregation on {device}")

    model = DoubleEncoderFlowMatching(multi_samples=True).to(device)

    B, N, F = 2, 3, 4
    same_number_augments = torch.rand(B, N, 1, 28, 28, device=device) * 2 - 1
    same_filter_augments = torch.rand(B, F, 1, 28, 28, device=device) * 2 - 1
    ground_truth = torch.rand(B, 1, 28, 28, device=device) * 2 - 1

    combined_z, pooled_number_z, pooled_filter_z = model.multi_sample_encoding(
        same_number_augments, same_filter_augments
    )

    assert combined_z.shape == (B, model.combined_latent_dim)
    assert pooled_number_z.shape == (B, model.number_latent_dim)
    assert pooled_filter_z.shape == (B, model.filter_latent_dim)

    flow_loss = model.get_flow_loss_multi(same_number_augments, same_filter_augments, ground_truth)
    assert torch.isfinite(flow_loss).all()
    print(f"Multi-sample flow loss: {flow_loss.mean().item():.4f}")


if __name__ == "__main__":
    test_double_encoder_flow()
    test_multi_sample_encoding()
