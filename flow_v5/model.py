import torch
from flow_v5 import config as cfg


def build_model(device: str = None, use_film: bool = None, multi_samples: bool = False,
                use_concatenation: bool = None, num_concat_samples: int = None,
                use_attention: bool = None, attn_layers: int = None,
                attn_heads: int = None, attn_hidden: int = None):
    """Construct DoubleEncoderFlowMatching with v5 defaults from config."""
    from flow_decoder.double_encoder_flow_architecture import DoubleEncoderFlowMatching

    number_latent_dim = cfg.NUMBER_ENCODER_LATENT_DIM
    filter_latent_dim = cfg.FILTER_ENCODER_LATENT_DIM
    if use_film is None:
        use_film = getattr(cfg, "USE_FILM", True)
    if use_concatenation is None:
        use_concatenation = getattr(cfg, "USE_CONCATENATION", False)
    if num_concat_samples is None:
        num_concat_samples = getattr(cfg, "NUM_SAMPLES_CONCATENATION", 5)
    if use_attention is None:
        use_attention = getattr(cfg, "USE_ATTENTION", False)
    if attn_layers is None:
        attn_layers = getattr(cfg, "ATTENTION_NUM_LAYERS", 2)
    if attn_heads is None:
        attn_heads = getattr(cfg, "ATTENTION_NUM_HEADS", 4)
    if attn_hidden is None:
        attn_hidden = getattr(cfg, "ATTENTION_HIDDEN_DIM", 128)

    model = DoubleEncoderFlowMatching(
        number_latent_dim=number_latent_dim,
        filter_latent_dim=filter_latent_dim,
        image_size=28,
        velocity_field_type="unet",
        n_integration_steps=50,
        unet_channels=[32, 64, 128],
        num_residual_layers=2,
        t_embed_dim=40,
        z_embed_dim=40,
        use_film=use_film,
        multi_samples=multi_samples,
        use_concatenation=use_concatenation,
        num_samples_concatenation=num_concat_samples,
        use_attention=use_attention
    )

    if device is not None:
        model = model.to(device)

    return model
