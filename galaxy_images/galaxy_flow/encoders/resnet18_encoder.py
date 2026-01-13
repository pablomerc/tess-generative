'''
ResNet-18 Encoder for galaxy images
We take the ResNet-18 Encoder architecture and modify it to
1) Take inputs with 4 channels
2) Output the latent space, rather than logits
For now it returns a latent space of size 512 (ResNet18 embedding dim)
But could be modify to a chosen latent size
'''

import torch
import torchvision.models as models
import torch.nn as nn

class GalaxyResnet(nn.Module):
    '''
    Resnet18 based galaxy encoder
    '''
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Remove classifier head to return latents
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Tensor (B,4,H,W)
        Returns:
            z: Tensor (B, 512)
        """
        return self.backbone(x)



if __name__ == '__main__':
    model = GalaxyResnet()
    # print(model)
    # Tesing the inputs of the model
    x = torch.randn(100,4,256,256)

    with torch.no_grad():
        z = model(x)

    print(f'Shape of z {z.shape}')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total model params: {total_params:,}')
