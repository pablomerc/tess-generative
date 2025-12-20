"""
Single Encoder Flow Matching Architecture for Galaxy Images

This model uses a single encoder to encode galaxy iamges into a latent space.

We can use it with the same survey (HSC and predict HSC, or Legacy and predict Legacy) or different surveys (Encode HSC and predict Legacy, or Legacy and predict HSC).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional

from galaxy_images.galaxy_flow.flowmatching_decoder import FlowMatchingDecoder

#TODO: Make a config for this model
from galaxy_images.galaxy_flow import unconditional_config as cfg

from galaxy_images.galaxy_flow.encoder_architectures import NumberEncoder, FilterEncoder

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
    #Unet params
    velocity_field_type: str=None,
    n_integration_steps: int = None,
    unet_channels: list=None,
    num_residual_layers: int = None,
    use_film: bool=None,
    t_embed_dim: int = None,
    z_embed_dim: int = None,
)
image_size = image_size or cfg.IMAGE_SIZE
output_dim = output_dim or cfg.OUTPUT_DIM
velocity_field_type = velocity_field_type or cfg.VELOCITY_FIELD_TYPE
n_integration_steps = n_integration_steps or cfg.N_INTEGRATION_STEPS
unet_channels = unet_channels or cfg.UNET_CHANNELS
num_residual_layers = num_residual_layers or cfg.NUM_RESIDUAL_LAYERS
use_film = use_film or cfg.USE_FILM
t_embed_dim = t_embed_dim or cfg.T_EMBED_DIM
z_embed_dim = z_embed_dim or cfg.Z_EMBED_DIM

super().__init__()
self.encoder = NumberEncoder(encoder_latent_dim)
self.decoder = FlowMatchingDecoder(
    input_dim=encoder_latent_dim,
    output_dim=output_dim,
    velocity_field_type=velocity_field_type,
    n_integration_steps=n_integration_steps,
    unet_channels=unet_channels,
    num_residual_layers=num_residual_layers,
    use_film=use_film,
    t_embed_dim=t_embed_dim,
    z_embed_dim=z_embed_dim,
)
