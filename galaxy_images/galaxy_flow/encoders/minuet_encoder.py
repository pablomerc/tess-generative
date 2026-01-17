from daep.ImgLayers import HostImgTransceiverEncoder
import torch
import torch.nn as nn

class MinuetEncoder(nn.Module):
    """
    Encoder Architecture from https://arxiv.org/pdf/2512.04145
    Wraps HostImgTransceiverEncoder for galaxy images
    """
    def __init__(self,
                 img_size: int = 48,
                 bottleneck_length: int = 64,
                 bottleneck_dim: int = 1,
                 patch_size: int = 3,
                 in_channels: int = 4,
                 model_dim: int = 256,
                 num_heads: int = 4,
                 ff_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 selfattn: bool = False,
                 sincosin: bool = True):
        super().__init__()

        self.encoder = HostImgTransceiverEncoder(
            img_size=img_size,
            bottleneck_length=bottleneck_length,
            bottleneck_dim=bottleneck_dim,
            patch_size=patch_size,
            in_channels=in_channels,
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout,
            selfattn=selfattn,
            sincosin=sincosin
        )

    def forward(self, x):
        """
        Args:
            x: Tensor (B, C, H, W) or dict with "flux" key containing tensor (B, C, H, W)
        Returns:
            z: Tensor (B, bottleneck_length, bottleneck_dim)
        """
        # If input is a tensor, wrap it in a dict with "flux" key
        if isinstance(x, torch.Tensor):
            x = {"flux": x}
        latent = self.encoder(x)
        return latent.squeeze(-1)


if __name__ == '__main__':

    model = MinuetEncoder(
        img_size=48,
        bottleneck_length=64,
        bottleneck_dim=1,
        patch_size=3,
        in_channels=4,  # RGB images
        model_dim=256,
        num_heads=4,
        ff_dim=256,
        num_layers=4,
        dropout=0.1,
        selfattn=False,
        sincosin=True
    )
    print(model)

    # Test the model
    images = torch.randn(32, 4, 48, 48)  # [batch, channels, height, width]
    with torch.no_grad():
        latent = model(images)  # Output: [batch, bottleneck_length, bottleneck_dim] = [32, 64, 1]

    print(f'Latent size {latent.shape}')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params:,}')
