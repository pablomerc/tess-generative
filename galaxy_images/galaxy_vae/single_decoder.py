'''
Single decoder for galaxy images, will be tested together with a single encoder to make a VAE.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *

class GalaxySingleDecoder(nn.Module):
    '''
    Decoder for a single Galaxy Image encoder.

    Input: latents (B,latent_dim)
    Output: (B, num_channels, 160, 160) - multi-channel galaxy images
    '''

    def __init__(self,
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS
    ):

        super().__init__()

        self.latent_dim=latent_dim
        self.num_channels=num_channels

        # First project onto the feature map size (separate from conv layers)
        self.fc_dec = nn.Linear(self.latent_dim, 512*5*5)

        # Decoder layers: 5x5 -> 10x10 -> 20x20 -> 40x40 -> 80x80 -> 160x160
        self.dec = nn.Sequential(
            # 5x5->10x10
            nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            #10x10->20x20
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 20x20->40x40
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 40x40->80x80
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 80x80 --> 160x160
            nn.ConvTranspose2d(64, num_channels, kernel_size=4, stride=2, padding=1)
            # No activation -- each channel treated differently
        )

    def forward(self, z):
        '''
        Forward pass: decode

        Args:
            z: latent encoding (B,LATENT_DIM)
        Returns:
            torch.Tensor: Reconstructed image (B,NUM_CH,160,160)
        '''
        # Project latent to feature map and reshape to (B, 512, 5, 5)
        h = self.fc_dec(z).view(-1, 512, 5, 5)

        # Decode through transpose convolutions
        reconstruction = self.dec(h)

        # Activations
        # Note: If your data is normalized to [0,1], sigmoid is fine for all channels.
        # If flux/ivar are in their original ranges, consider:
        #   - Flux: ReLU (to ensure non-negative) or no activation if normalized
        #   - IVAR: ReLU (to ensure non-negative) or no activation if normalized
        #   - Mask: sigmoid (correct for [0,1])
        #   - Object mask: sigmoid (correct for [0,1])

        #Channels 0-3 are flux [0,inf] - using sigmoid assumes normalized to [0,1]
        #Channels 4-7 are ivar [0,inf] - using sigmoid assumes normalized to [0,1]
        #Channels 8-11 are mask [0,1] - sigmoid is correct
        #Channel 12 is object mask [0,1] - sigmoid is correct

        reconstruction[:,0:4,:,:]=F.sigmoid(reconstruction[:,0:4,:,:])
        if USE_IVAR:
            reconstruction[:,4:8,:,:]=F.relu(reconstruction[:,4:8,:,:])
        if USE_FLUX_MASK:
            reconstruction[:,8:12,:,:]=F.sigmoid(reconstruction[:,8:12,:,:])
        if USE_OBJECT_MASK:
            reconstruction[:,-1,:,:]=F.sigmoid(reconstruction[:,-1,:,:])

        return reconstruction


def test_decoder():
    """Test function to verify decoder architecture"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    batch_size = 128
    z = torch.randn(batch_size, LATENT_DIM).to(device)

    print(f"Testing with latent shapes: z={z.shape}")

    # Test Decoder
    print("\nTesting GalaxySingleDecoder...")
    decoder = GalaxySingleDecoder().to(device)
    reconstruction = decoder(z)
    print(f"Decoder output shape: {reconstruction.shape}")
    print(f"Expected shape: ({batch_size}, {NUM_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE})")

    assert reconstruction.shape == (batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), \
        f"Shape mismatch! Got {reconstruction.shape}, expected ({batch_size}, {NUM_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE})"

    print("\nDecoder test passed!")


if __name__ == "__main__":
    test_decoder()
