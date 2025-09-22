import torch
from flow_v5 import config as cfg


def build_model(device: str = None):
    """Construct DoubleEncoderFlowMatching with v5 defaults from config."""
    from flow_decoder.VAE_flow_v5 import DoubleEncoderFlowMatching

    number_latent_dim = cfg.NUMBER_ENCODER_LATENT_DIM
    filter_latent_dim = cfg.FILTER_ENCODER_LATENT_DIM

    model = DoubleEncoderFlowMatching(
        number_latent_dim=number_latent_dim,
        filter_latent_dim=filter_latent_dim,
        image_size=28,
        velocity_field_type="unet",
        n_integration_steps=50,
        unet_channels=[32, 64, 128],
        num_residual_layers=2,
        t_embed_dim=40,
        z_embed_dim=40
    )

    if device is not None:
        model = model.to(device)

    return model
