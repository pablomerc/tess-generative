"""
Single Encoder Flow Matching Architecture for Galaxy Images

This model uses a single encoder to encode galaxy images into a latent space.

We can use it with the same survey (HSC and predict HSC, or Legacy and predict Legacy) or different surveys (Encode HSC and predict Legacy, or Legacy and predict HSC).

"""

import torch
import torch.nn as nn
from typing import Optional

from galaxy_images.galaxy_flow.flowmatching_decoder import FlowMatchingDecoder

from galaxy_images.galaxy_flow import single_encoder_config as cfg

from galaxy_images.galaxy_flow.encoder_architectures import GalaxyEncoder

from galaxy_images.galaxy_flow.encoders.resnet18_encoder import GalaxyResnet

class SingleEncoderGalaxyFlow(nn.Module):
    '''
    Conditional Flow Matching model for galaxy images with a single encoder.
    The images are expected to be preprocessed (cropped, clamped, rescaled, range compressed).
    '''
    def __init__(
        self,
        encoder_latent_dim: int=None,
        image_size: int=None,
        output_dim: int=None,
        num_channels: int=None,
        #Unet params
        velocity_field_type: str=None,
        n_integration_steps: int = None,
        unet_channels: list=None,
        num_residual_layers: int = None,
        use_film: bool=None,
        t_embed_dim: int = None,
        z_embed_dim: int = None,
        encoder_type: str = 'resnet'
    ):
        encoder_type = encoder_type or cfg.ENCODER_TYPE
        encoder_latent_dim = encoder_latent_dim or cfg.ENCODER_LATENT_DIM
        image_size = image_size or cfg.IMAGE_SIZE
        num_channels = num_channels if num_channels is not None else cfg.NUM_CHANNELS
        output_dim = output_dim or cfg.OUTPUT_DIM
        velocity_field_type = velocity_field_type or cfg.VELOCITY_FIELD_TYPE
        n_integration_steps = n_integration_steps or cfg.N_INTEGRATION_STEPS
        unet_channels = unet_channels or cfg.UNET_CHANNELS
        num_residual_layers = num_residual_layers or cfg.NUM_RESIDUAL_LAYERS
        use_film = use_film or cfg.USE_FILM
        t_embed_dim = t_embed_dim or cfg.T_EMBED_DIM
        z_embed_dim = z_embed_dim or cfg.Z_EMBED_DIM

        super().__init__()

        if encoder_type == 'cnn':
            self.encoder = GalaxyEncoder(encoder_latent_dim)
            self._encoder_returns_tuple = True  # GalaxyEncoder returns (z, mu, logvar)
        elif encoder_type == 'resnet':
            self.encoder = GalaxyResnet(z_dim=encoder_latent_dim)
            self._encoder_returns_tuple = False  # GalaxyResnet returns z directly
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'cnn' or 'resnet'")


        self.decoder = FlowMatchingDecoder(
            input_dim=encoder_latent_dim,
            output_dim=output_dim,
            velocity_field_type=velocity_field_type,
            n_integration_steps=n_integration_steps,
            image_size=image_size,
            unet_channels=unet_channels,
            num_residual_layers=num_residual_layers,
            use_film=use_film,
            t_embed_dim=t_embed_dim,
            z_embed_dim=z_embed_dim,
            num_channels=num_channels,
        )

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        ''' Encode an image into a low dimensional embedding using the encoder'''
        encoder_output = self.encoder(img)
        # GalaxyEncoder returns (z, mu, logvar), GalaxyResnet returns z directly
        z = encoder_output[0] if self._encoder_returns_tuple else encoder_output
        return z

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass: compute the flow matching loss

        Args:
            x: preprocessed galaxy images, flattened to (B, output_dim)
               where output_dim = num_channels * image_size * image_size
               Images should be preprocessed (cropped, clamped, rescaled, range compressed)
               and normalized (?) to [-1, 1] range

            z: encoding vector (B, d)
               where d is the encoder_latent_dim
               this should be the encoding of the counterfactual image

        Returns:
            loss: Flow Matching loss
        '''
        loss = self.decoder.get_loss(x, z=z)
        return loss

    def sample(self, device: torch.device = None, n_samples: int=1, z: torch.Tensor = None) -> torch.Tensor:
        '''
        Generate samples from the conditional model

        For now - it assumes z is always provided

        '''

        if device is None:
            device = next(self.parameters()).device

        samples = self.decoder.sample(
            z=z,
            batch_size=None,  #determined by z
            device=device,
            n_samples=n_samples
        )
        return samples

def build_model(device: str = None, **kwargs) -> SingleEncoderGalaxyFlow:
    '''
    Build a single encoder galaxy flow matching model

    Args:
        device: Device to move model to (e.g., 'cuda', 'cpu')
        **kwargs: Override any config parameters

    Returns:
        SingleEncoderGalaxyFlow model
    '''
    model = SingleEncoderGalaxyFlow(**kwargs)

    if device is not None:
        model = model.to(device)

    return model
