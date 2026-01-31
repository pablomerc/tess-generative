import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import timm
import time
import sys
import shutil
import datetime
import math
from pathlib import Path
from diffusers import UNet2DConditionModel, UNet2DModel
from typing import Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import geomloss
import umap

import matplotlib.pyplot as plt

class ResNetEncoder(nn.Module):
    """
    ResNet18 encoder from timm that produces spatial feature maps for conditioning.
    Uses feature extraction to get intermediate spatial features for cross-attention.
    """

    def __init__(
        self,
        in_channels: int = 4,
        cross_attention_dim: int = 256,
        pretrained: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),  # Get features from layer2, layer3, layer4
        )

        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        self.proj = nn.Conv2d(512, cross_attention_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Conditioning image (B, C, H, W)
        Returns:
            Spatial embeddings (B, seq_len, cross_attention_dim) for cross-attention
        """
        features = self.backbone(x)
        feat = features[-1]  # (B, 512, H/32, W/32)
        feat = self.proj(feat)  # (B, cross_attention_dim, H', W')

        B, D, H, W = feat.shape
        feat = feat.view(B, D, H * W).permute(0, 2, 1)
        return feat

    def intermediate_states(self, x: torch.Tensor) -> list:
        features = self.backbone(x)
        for i, element in enumerate(features):
            print(f'Shape of features element {i} is {element.shape}')
        return features


def visualize_feat0(x: torch.Tensor, save_path: Optional[str] = None) -> None:
    """
    Visualize intermediate CNN feature maps.

    Parameters
    ----------
    x : torch.Tensor
        Feature tensor of shape (B, C, H, W).
    save_path : Optional[str]
        Path to save the figure. If None, saves to 'intermediate_features.png' in current directory.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    B, C, H, W = x.shape
    image = x[0].detach().cpu()

    nrows = 4
    ncols = math.ceil(C / nrows)  # Ensure we have enough columns

    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))

    for i, ax in enumerate(axes.flat):
        if i >= C:
            ax.axis("off")
            continue

        ax.imshow(image[i], cmap="viridis")
        ax.set_title(f"ch {i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    if save_path is None:
        save_path = "intermediate_features.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()


if __name__ == "__main__":

    print('Creating model...')
    model = ResNetEncoder(
        in_channels=4,
        cross_attention_dim=16,
        pretrained=False,
    )
    print('Model created')

    # load a random image from the dataset
    image = torch.randn(64, 4, 48, 48)
    print('Image batch created')

    # forward pass
    output = model(image)
    print('Output created')

    print(output.shape)

    # Get intermediate features
    features = model.intermediate_states(image)

    # Visualize features[0] which should be (B, 128, 6, 6) based on the TODO
    print(f'\nVisualizing features[0] with shape {features[0].shape}...')
    visualize_feat0(features[0])
