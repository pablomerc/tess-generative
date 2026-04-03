import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import timm
import numpy as np
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from typing import Dict, List, Optional, Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# =============================================================================
# Hyperparameters: ResNet18 backbone for 48×48 input
# =============================================================================

# backbone level index → number of channels
BACKBONE_CHANNELS = {0: 64, 1: 64, 2: 128, 3: 256}

# backbone level index → spatial resolution for 48×48 input
BACKBONE_RES = {0: 24, 1: 12, 2: 6, 3: 3}

# backbone level index → which UNet blocks it feeds
# UNet blocks: down_0(48), down_1(24), down_2(12), down_3(6), mid(6),
#              up_0(6), up_1(12), up_2(24), up_3(48)
BACKBONE_TO_UNET = {
    0: {"down": 1, "up": 2, "res": 24},   # stem   → down_1 / up_2
    1: {"down": 2, "up": 1, "res": 12},   # layer1 → down_2 / up_1
    2: {"down": 3, "up": 0, "res": 6},    # layer2 → down_3 / up_0 / mid
}


# =============================================================================
# Experiment configurations
# =============================================================================

EXPERIMENTS = {
    "hier_full": {
        "spatial_indices": [0, 1, 2],       
        "reductions": {},                     
        "token_dim": 512,
        "global_dim": 512,
    },
    "hier_stride2": {
        "spatial_indices": [0, 1, 2],
        "reductions": {0: "stride2"},        
        "token_dim": 512,
        "global_dim": 512,
    },
    "bn_36x64": {
        "spatial_indices": [2],              
        "reductions": {},
        "token_dim": 64,
        "global_dim": 128,
    },
    "bn_36x16": { # should correspond to x14 compression of original images
        "spatial_indices": [2],
        "reductions": {},
        "token_dim": 16,
        "global_dim": 64,
    },
}


# =============================================================================
# 1. 2D Rotary Position Embeddings
# =============================================================================

class RotaryEmbedding2D(nn.Module):
    """2D RoPE for cross-attention. Splits head_dim into 4 quarters for (h,w)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0
        quarter = dim // 4
        freqs = 1.0 / (theta ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))
        self.register_buffer("freqs", freqs)
        self._cache = {}

    def _get_sincos(self, resolution: int, device):
        key = (resolution, device)
        if key not in self._cache:
            pos = torch.arange(resolution, device=device, dtype=torch.float32)
            angles = torch.outer(pos, self.freqs.to(device))
            self._cache[key] = (angles.sin(), angles.cos())
        return self._cache[key]

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, heads, N, D = x.shape
        assert N == h * w
        q = D // 4

        sin_h, cos_h = self._get_sincos(h, x.device)
        sin_w, cos_w = self._get_sincos(w, x.device)

        sh = sin_h[:, None, :].expand(h, w, q).reshape(N, q)[None, None]
        ch = cos_h[:, None, :].expand(h, w, q).reshape(N, q)[None, None]
        sw = sin_w[None, :, :].expand(h, w, q).reshape(N, q)[None, None]
        cw = cos_w[None, :, :].expand(h, w, q).reshape(N, q)[None, None]

        x1, x2, x3, x4 = x.chunk(4, dim=-1)
        return torch.cat([
            x1 * ch - x2 * sh, x1 * sh + x2 * ch,
            x3 * cw - x4 * sw, x3 * sw + x4 * cw,
        ], dim=-1)


# =============================================================================
# 2. RoPE Cross-Attention Processor
# =============================================================================

class RoPECrossAttnProcessor:
    """Drop-in attn2 processor applying 2D RoPE to Q and K."""

    def __init__(self, head_dim: int, resolution: int):
        self.rope = RotaryEmbedding2D(dim=head_dim)
        self.resolution = resolution

    def __call__(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.reshape(B, C, H * W).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        h = w = self.resolution
        query = self.rope(query, h, w)
        key = self.rope(key, h, w)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=attn.dropout, training=attn.training)

        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(batch_size, C, H, W)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


# =============================================================================
# 3. Configurable Encoder
# =============================================================================

class ConfigurableEncoder(nn.Module):
    """
    ResNet18 encoder with configurable spatial token extraction.

    Args:
        spatial_indices: Which backbone levels produce spatial tokens.
            0 = stem (24×24), 1 = layer1 (12×12), 2 = layer2 (6×6).
        reductions: Per-level spatial reduction before token projection.
            {level_idx: 'stride2'} — learned 3×3 stride-2 conv (halves resolution).
        token_dim: Projection dimension for spatial tokens (= cross_attention_dim).
        global_dim: Global embedding dimension (for AdaGN via class_labels).
            Level 3 (layer3, 3×3) is always used for global embedding.

    Returns:
        spatial_levels: list of (tokens(B,N,D), grid_h, grid_w)
        global_embedding: (B, global_dim)
        rope_flags: list[bool] — True if token grid matches UNet Q grid
    """

    def __init__(
        self,
        in_channels: int = 4,
        spatial_indices: Tuple[int, ...] = (2,),
        reductions: Optional[Dict[int, str]] = None,
        token_dim: int = 64,
        global_dim: int = 128,
        pretrained: bool = False,
    ):
        super().__init__()
        self.spatial_indices = sorted(spatial_indices)
        self.reductions = reductions or {}
        self.token_dim = token_dim
        self.global_dim = global_dim

        # Always include level 3 for global embedding
        all_indices = sorted(set(self.spatial_indices + [3]))
        self._all_indices = all_indices

        self.backbone = timm.create_model(
            "resnet18", pretrained=pretrained, features_only=True,
            out_indices=tuple(all_indices),
        )

        if in_channels != 3:
            old = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )

        # Stride-2 reduction modules
        self.reduction_modules = nn.ModuleDict()
        for idx, mode in self.reductions.items():
            assert mode == "stride2", f"Only 'stride2' reduction supported, got '{mode}'"
            ch = BACKBONE_CHANNELS[idx]
            self.reduction_modules[str(idx)] = nn.Sequential(
                nn.Conv2d(ch, ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.GELU(),
            )

        # Spatial projections: backbone channels → token_dim
        self.spatial_projs = nn.ModuleDict()
        for idx in self.spatial_indices:
            ch = BACKBONE_CHANNELS[idx]
            self.spatial_projs[str(idx)] = nn.Sequential(
                nn.LayerNorm(ch),
                nn.Linear(ch, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

        # Global projection: layer3 (256ch) → mean pool → global_dim
        self.global_proj = nn.Sequential(
            nn.LayerNorm(BACKBONE_CHANNELS[3]),
            nn.Linear(BACKBONE_CHANNELS[3], global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
        )

    def forward(self, x: torch.Tensor):
        features_list = self.backbone(x)
        feat_map = {idx: features_list[i] for i, idx in enumerate(self._all_indices)}

        spatial_levels = []
        rope_flags = []

        for idx in self.spatial_indices:
            feat = feat_map[idx]
            native_res = feat.shape[2]

            # Apply reduction if configured
            if str(idx) in self.reduction_modules:
                feat = self.reduction_modules[str(idx)](feat)

            B, C, H, W = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)       # (B, H*W, C)
            tokens = self.spatial_projs[str(idx)](tokens)   # (B, H*W, token_dim)
            spatial_levels.append((tokens, H, W))

            # RoPE valid only when token grid matches UNet Q grid
            rope_flags.append(H == native_res and W == native_res)

        # Global from layer3
        deep = feat_map[3]
        global_vec = deep.flatten(2).mean(dim=2)
        global_vec = self.global_proj(global_vec)

        return spatial_levels, global_vec, rope_flags


# =============================================================================
# 4. UNet Wrapper — routes encoder features to matched blocks
# =============================================================================

class ConditionedUNet(nn.Module):
    """
    Wraps UNet2DConditionModel, routing encoder spatial tokens to matched
    cross-attention blocks and global embedding to AdaGN.

    level_map: {block_name → level_index_or_None}
        e.g. {"down_3": 0, "mid": 0, "up_0": 0, "down_0": None, ...}
    """

    def __init__(self, unet: UNet2DConditionModel, level_map: Dict[str, Optional[int]]):
        super().__init__()
        self.unet = unet
        self.level_map = level_map

    def setup_rope_processors(self, level_grid_sizes: Dict[int, int],
                               level_rope: Dict[int, bool]):
        """Install RoPE processors where token grid matches UNet Q grid."""
        block_info = {}
        for block_name, level_idx in self.level_map.items():
            if level_idx is not None and level_idx in level_grid_sizes:
                block_info[block_name] = (
                    level_grid_sizes[level_idx],
                    level_rope.get(level_idx, True),
                )

        attn_procs = {}
        for name, proc in self.unet.attn_processors.items():
            if "attn2" not in name:
                attn_procs[name] = proc
                continue

            block_name = self._attn_to_block(name)
            if block_name in block_info:
                resolution, use_rope = block_info[block_name]
                if use_rope:
                    layer = self._get_layer(name)
                    hd = layer.inner_dim // layer.heads
                    attn_procs[name] = RoPECrossAttnProcessor(head_dim=hd, resolution=resolution)
                    print(f"  RoPE on {name} (res={resolution}, head_dim={hd})")
                else:
                    attn_procs[name] = proc
                    print(f"  Standard xattn on {name} (no RoPE, grid mismatch)")
            else:
                attn_procs[name] = proc

        self.unet.set_attn_processor(attn_procs)

    def _attn_to_block(self, name: str) -> str:
        if "down_blocks" in name:
            return f"down_{name.split('down_blocks.')[1].split('.')[0]}"
        elif "mid_block" in name:
            return "mid"
        elif "up_blocks" in name:
            return f"up_{name.split('up_blocks.')[1].split('.')[0]}"
        return ""

    def _get_layer(self, name: str) -> Attention:
        parts = name.replace(".processor", "").split(".")
        mod = self.unet
        for p in parts:
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        return mod

    def forward(self, sample, timestep, spatial_levels, class_labels=None):
        ts = timestep
        if not torch.is_tensor(ts):
            ts = torch.tensor([ts], dtype=torch.long, device=sample.device)
        elif ts.ndim == 0:
            ts = ts[None].to(sample.device)
        ts = ts.expand(sample.shape[0])

        t_emb = self.unet.time_proj(ts).to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb)
        if self.unet.class_embedding is not None and class_labels is not None:
            emb = emb + self.unet.class_embedding(class_labels).to(dtype=sample.dtype)

        sample = self.unet.conv_in(sample)

        # Down blocks
        down_res = (sample,)
        for i, block in enumerate(self.unet.down_blocks):
            lvl = self.level_map.get(f"down_{i}")
            enc = spatial_levels[lvl][0] if lvl is not None else None
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample, res = block(hidden_states=sample, temb=emb, encoder_hidden_states=enc)
            else:
                sample, res = block(hidden_states=sample, temb=emb)
            down_res += res

        # Mid block
        lvl = self.level_map.get("mid")
        enc = spatial_levels[lvl][0] if lvl is not None else None
        if hasattr(self.unet.mid_block, "has_cross_attention") and self.unet.mid_block.has_cross_attention:
            sample = self.unet.mid_block(hidden_states=sample, temb=emb, encoder_hidden_states=enc)
        else:
            sample = self.unet.mid_block(hidden_states=sample, temb=emb)

        # Up blocks
        for i, block in enumerate(self.unet.up_blocks):
            n = len(block.resnets)
            res = down_res[-n:]
            down_res = down_res[:-n]
            lvl = self.level_map.get(f"up_{i}")
            enc = spatial_levels[lvl][0] if lvl is not None else None
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample = block(hidden_states=sample, temb=emb,
                               res_hidden_states_tuple=res, encoder_hidden_states=enc)
            else:
                sample = block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res)

        if self.unet.conv_norm_out:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        return self.unet.conv_out(sample)


# =============================================================================
# 5. Build UNet + level_map from experiment config
# =============================================================================

def build_unet_and_level_map(
    config: dict,
    in_channels: int,
    image_size: int,
    model_channels: int,
    channel_mult: tuple,
    layers_per_block: int,
    attention_head_dim: int,
):
    """
    Auto-construct UNet block types and level_map from experiment config.

    Cross-attention blocks are placed only where spatial_indices map to UNet.
    Mid block gets cross-attention if any spatial level maps to 6×6.

    attention_head_dim only needs to divide block_out_channels at cross-attn
    positions (and be divisible by 4 for RoPE). It does NOT need to divide
    token_dim — cross_attention_dim is just an input to the K/V linear
    projections, not reshaped by head_dim.
    """
    spatial_indices = config["spatial_indices"]
    token_dim = config["token_dim"]
    global_dim = config["global_dim"]

    block_out_channels = tuple(model_channels * m for m in channel_mult)

    # Which UNet down/up indices need cross-attention?
    xattn_down = set()
    xattn_up = set()
    has_6x6 = False

    for idx in spatial_indices:
        info = BACKBONE_TO_UNET[idx]
        xattn_down.add(info["down"])
        xattn_up.add(info["up"])
        if info["res"] == 6:
            has_6x6 = True

    down_types = tuple(
        "CrossAttnDownBlock2D" if i in xattn_down else "DownBlock2D"
        for i in range(len(channel_mult))
    )
    up_types = tuple(
        "CrossAttnUpBlock2D" if i in xattn_up else "UpBlock2D"
        for i in range(len(channel_mult))
    )
    mid_type = "UNetMidBlock2DCrossAttn" if has_6x6 else "UNetMidBlock2D"

    # attention_head_dim must divide block_out_channels at xattn positions
    # and be divisible by 4 for RoPE. Cap at smallest xattn block width.
    xattn_block_indices = xattn_down | xattn_up
    effective_hd = min(attention_head_dim,
                       min(block_out_channels[i] for i in xattn_block_indices))
    # Ensure divisibility
    while effective_hd > 4:
        ok = (
            all(block_out_channels[i] % effective_hd == 0 for i in xattn_block_indices)
            and effective_hd % 4 == 0
        )
        if ok:
            break
        effective_hd //= 2

    unet = UNet2DConditionModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=in_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_types,
        up_block_types=up_types,
        mid_block_type=mid_type,
        cross_attention_dim=token_dim,
        attention_head_dim=effective_hd,
        resnet_time_scale_shift="scale_shift",
        class_embed_type="projection",
        projection_class_embeddings_input_dim=global_dim,
    )

    # Build level_map
    level_map = {f"down_{i}": None for i in range(len(channel_mult))}
    level_map.update({f"up_{i}": None for i in range(len(channel_mult))})
    level_map["mid"] = None

    for level_idx, backbone_idx in enumerate(spatial_indices):
        info = BACKBONE_TO_UNET[backbone_idx]
        level_map[f"down_{info['down']}"] = level_idx
        level_map[f"up_{info['up']}"] = level_idx
        if info["res"] == 6:
            level_map["mid"] = level_idx

    return unet, level_map, effective_hd


# =============================================================================
# 6. Lightning Module
# =============================================================================

class FlowMatchingModule(pl.LightningModule):
    """
    Conditional Flow Matching with configurable encoder.
    Encoder and UNet built automatically from experiment config dict.
    """

    def __init__(
        self,
        experiment_config: dict,
        in_channels: int = 4,
        cond_channels: int = 4,
        image_size: int = 48,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        layers_per_block: int = 2,
        attention_head_dim: int = 64,
        pretrained_encoder: bool = False,
        lr: float = 1e-4,
        num_sample_images: int = 6,
        num_integration_steps: int = 250,
        num_astropy_lenses: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_sample_images = num_sample_images
        self.num_integration_steps = num_integration_steps
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_astropy_lenses = num_astropy_lenses

        cfg = experiment_config

        # ---- Encoder ----
        self.encoder = ConfigurableEncoder(
            in_channels=cond_channels,
            spatial_indices=cfg["spatial_indices"],
            reductions=cfg.get("reductions", {}),
            token_dim=cfg["token_dim"],
            global_dim=cfg["global_dim"],
            pretrained=pretrained_encoder,
        )

        # ---- UNet ----
        unet, level_map, effective_hd = build_unet_and_level_map(
            config=cfg,
            in_channels=in_channels,
            image_size=image_size,
            model_channels=model_channels,
            channel_mult=channel_mult,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
        )
        self.conditioned_unet = ConditionedUNet(unet=unet, level_map=level_map)

        # ---- Probe encoder + setup RoPE ----
        print(f"\nEncoder configuration:")
        with torch.no_grad():
            dummy = torch.zeros(1, cond_channels, image_size, image_size)
            spatial_levels, global_vec, rope_flags = self.encoder(dummy)

            grid_sizes = {}
            rope_enabled = {}
            total_tokens = 0
            total_spatial_values = 0
            for i, (tokens, gh, gw) in enumerate(spatial_levels):
                n_tok = tokens.shape[1]
                total_tokens += n_tok
                total_spatial_values += n_tok * cfg["token_dim"]
                grid_sizes[i] = gh
                rope_enabled[i] = rope_flags[i]
                print(f"  Level {i}: {gh}×{gw} = {n_tok} tokens × {cfg['token_dim']}d"
                      f" = {n_tok * cfg['token_dim']} values"
                      f" | RoPE={'yes' if rope_flags[i] else 'no'}")

            total_cond = total_spatial_values + cfg["global_dim"]
            input_vals = cond_channels * image_size * image_size
            ratio = input_vals / total_cond
            tag = f"{ratio:.1f}× compression" if ratio > 1 else f"{1/ratio:.1f}× expansion"
            print(f"  Global: {cfg['global_dim']}d")
            print(f"  Total: {total_tokens} tokens + global = {total_cond} values ({tag} from {input_vals})")
            print(f"  Attention head dim: {effective_hd}")

        has_xattn = any(v is not None for v in level_map.values())
        if has_xattn:
            print("Setting up cross-attention processors...")
            self.conditioned_unet.setup_rope_processors(grid_sizes, rope_enabled)
        print("Done.\n")

    # -----------------------------------------------------------------
    # Forward / loss / sampling
    # -----------------------------------------------------------------

    def forward(self, x_t, t, cond_image):
        timesteps = t * 1000
        spatial_levels, global_embedding, _ = self.encoder(cond_image)
        return self.conditioned_unet(
            sample=x_t, timestep=timesteps,
            spatial_levels=spatial_levels,
            class_labels=global_embedding,
        )

    def compute_loss(self, batch):
        x_1, cond_image = batch
        B = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=x_1.device)
        t_exp = t[:, None, None, None]
        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        target = x_1 - x_0
        predicted = self(x_t, t, cond_image)
        return F.mse_loss(predicted, target)

    @torch.no_grad()
    def sample(self, cond_images, num_steps=None):
        num_steps = num_steps or self.num_integration_steps
        B = cond_images.shape[0]
        x = torch.randn(B, self.in_channels, self.image_size, self.image_size,
                         device=cond_images.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=x.device)
            x = x + self(x, t, cond_images) * dt
        return x

    # -----------------------------------------------------------------
    # Lightning hooks
    # -----------------------------------------------------------------

    def on_train_start(self):
        val_loaders = getattr(self.trainer, "val_dataloaders", None)
        if val_loaders is not None and len(val_loaders) > 1:
            self._lense_batch = next(iter(val_loaders[1]))
        else:
            self._lense_batch = None

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 1:
            self._val_lens_target_batch = batch[0].clone()
            self._val_lens_cond_batch = batch[1].clone()
            loss = self.compute_loss(batch)
            self.log("val/loss_lenses", loss, on_epoch=True, sync_dist=True)
            return loss
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        if batch_idx == 0:
            self._val_cond_batch = batch[1][:self.num_sample_images].clone()
            self._val_target_batch = batch[0][:self.num_sample_images].clone()
        return loss

    # -----------------------------------------------------------------
    # W&B visualization
    # -----------------------------------------------------------------

    def _norm_vis(self, img):
        img = img.clone()
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        return img

    def on_validation_epoch_end(self):
        if not self.logger or not hasattr(self, "_val_cond_batch"):
            return
        import matplotlib.pyplot as plt

        # ===== Sample grid =====
        nc = min(6, len(self._val_cond_batch))
        ns = 5
        ncols = 2 + ns + 1

        fig, ax = plt.subplots(nc, ncols, figsize=(2 * ncols, 2 * nc), squeeze=False)
        titles = ["Cond", "Target"] + [f"S{j+1}" for j in range(ns)] + ["Mean"]
        for j, t in enumerate(titles):
            ax[0, j].set_title(t, fontsize=10)

        for i in range(nc):
            c = self._val_cond_batch[i:i+1].to(self.device)
            tgt = self._val_target_batch[i:i+1].to(self.device)
            samples = self.sample(c.repeat(ns, 1, 1, 1))
            mean_s = samples.mean(0, keepdim=True)

            ax[i, 0].imshow(self._norm_vis(c[0, :3]).cpu().permute(1, 2, 0).numpy())
            ax[i, 0].axis("off")
            ax[i, 1].imshow(self._norm_vis(tgt[0, :3]).cpu().permute(1, 2, 0).numpy())
            ax[i, 1].axis("off")
            for j in range(ns):
                ax[i, 2+j].imshow(self._norm_vis(samples[j, :3]).cpu().permute(1, 2, 0).numpy())
                ax[i, 2+j].axis("off")
            ax[i, -1].imshow(self._norm_vis(mean_s[0, :3]).cpu().permute(1, 2, 0).numpy())
            ax[i, -1].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({"val/sample_grid": wandb.Image(fig),
                                     "global_step": self.global_step})
        plt.close(fig)

        # ===== Lens plots =====
        if hasattr(self, "_val_lens_target_batch") and hasattr(self, "_val_lens_cond_batch"):
            self._plot_lens_row_scaled()
            self._plot_lens_astropy()

    def _plot_lens_row_scaled(self):
        import matplotlib.pyplot as plt

        def _row_rgb(x, vmin, vmax):
            x3 = x[:3]
            lo = torch.as_tensor(vmin, device=x3.device, dtype=x3.dtype).view(3, 1, 1)
            hi = torch.as_tensor(vmax, device=x3.device, dtype=x3.dtype).view(3, 1, 1)
            return ((x3 - lo) / (hi - lo + 1e-8)).clamp(0, 1).permute(1, 2, 0)

        nl = min(6, len(self._val_lens_target_batch))
        ns = 5
        ncols = 2 + ns + 1
        fig, ax = plt.subplots(nl, ncols, figsize=(2 * ncols, 2 * nl), squeeze=False)
        titles = ["Cond", "Target"] + [f"S{j+1}" for j in range(ns)] + ["Mean"]
        for j, t in enumerate(titles):
            ax[0, j].set_title(t, fontsize=10)

        for i in range(nl):
            c = self._val_lens_cond_batch[i:i+1].to(self.device)
            tgt = self._val_lens_target_batch[i:i+1].to(self.device)
            samps = self.sample(c.repeat(ns, 1, 1, 1))
            mean_s = samps.mean(0, keepdim=True)
            vmin = tgt[0, :3].amin(dim=(1, 2))
            vmax = tgt[0, :3].amax(dim=(1, 2))

            ax[i, 0].imshow(_row_rgb(c[0], vmin, vmax).cpu().numpy()); ax[i, 0].axis("off")
            ax[i, 1].imshow(_row_rgb(tgt[0], vmin, vmax).cpu().numpy()); ax[i, 1].axis("off")
            for j in range(ns):
                ax[i, 2+j].imshow(_row_rgb(samps[j], vmin, vmax).cpu().numpy())
                ax[i, 2+j].axis("off")
            ax[i, -1].imshow(_row_rgb(mean_s[0], vmin, vmax).cpu().numpy())
            ax[i, -1].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({"val/sample_grid_row_scaled_lenses": wandb.Image(fig),
                                     "global_step": self.global_step})
        plt.close(fig)

    def _plot_lens_astropy(self):
        import matplotlib.pyplot as plt
        try:
            from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

            def _imshow(ax, d, iv, st, title=None):
                d = np.asarray(d, dtype=np.float32)
                m = np.isfinite(d)
                if not np.any(m):
                    ax.axis("off"); return
                lo, hi = iv.get_limits(d[m])
                ax.imshow(d, origin="lower",
                          norm=ImageNormalize(vmin=lo, vmax=hi, stretch=st, clip=True),
                          cmap="magma")
                if title:
                    ax.set_title(title, fontsize=9, fontweight="bold")
                ax.set_xticks([]); ax.set_yticks([])

            def _rgb(img, r, g, b, iv, st):
                nc = img.shape[0]
                r, g, b = [min(x, nc - 1) for x in [r, g, b]]
                rgb = np.stack([img[r], img[g], img[b]], axis=-1).astype(np.float32)
                out = np.zeros_like(rgb)
                for k in range(3):
                    ch = rgb[..., k]
                    m = np.isfinite(ch)
                    if not np.any(m):
                        continue
                    lo, hi = iv.get_limits(ch[m])
                    out[..., k] = ImageNormalize(vmin=lo, vmax=hi, stretch=st, clip=True)(ch)
                mx = np.nanmax(out)
                if mx > 0:
                    out /= mx
                return np.clip(out, 0, 1)

            iv = PercentileInterval(99.5)
            st = AsinhStretch()
            bands = ["g", "r", "i", "z"]

            nl = min(self.num_astropy_lenses, len(self._val_lens_target_batch))
            fig, gs = plt.subplots(3 * nl, 5, figsize=(14, 10 / 3 * nl),
                                   constrained_layout=True)
            fig.suptitle("Lens: Target | Sample | Legacy", fontsize=12, y=1.02)
            if 3 * nl == 1:
                gs = gs.reshape(1, -1)

            for L in range(nl):
                cL = self._val_lens_cond_batch[L:L+1].to(self.device)
                tL = self._val_lens_target_batch[L:L+1].to(self.device)
                sL = self.sample(cL)
                t_np = tL[0].cpu().numpy()
                s_np = sL[0].cpu().numpy()
                c_np = cL[0].cpu().numpy()
                b = L * 3

                for row, arr, lbl in [(b, t_np, "Target"),
                                       (b+1, s_np, "Sample"),
                                       (b+2, c_np, "Legacy")]:
                    for ch in range(min(4, arr.shape[0])):
                        _imshow(gs[row, ch], arr[ch], iv, st,
                                title=f"{lbl} {bands[ch]}")
                    gs[row, 4].imshow(_rgb(arr, 2, 1, 0, iv, st), origin="lower")
                    gs[row, 4].set_title(f"{lbl} RGB", fontsize=9, fontweight="bold")
                    gs[row, 4].axis("off")

            self.logger.experiment.log({
                "val/lens_triple_target_sample_legacy": wandb.Image(fig),
                "global_step": self.global_step,
            })
            plt.close(fig)
        except Exception as e:
            if self.global_rank == 0:
                print(f"[Lens astropy] Skipped: {e}")

    # -----------------------------------------------------------------
    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        sched = CosineAnnealingLR(opt, T_max=self.trainer.max_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# =============================================================================
# 7. Main
# =============================================================================

if __name__ == "__main__":
    from pytorch_lightning.loggers import WandbLogger
    from torch.utils.data import DataLoader
    from data import HSCLegacyDatasetZoom, HSCLegacyDatasetZoomLenses

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help="Experiment config name")
    parser.add_argument("--max_steps", type=int, default=300_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    cfg = EXPERIMENTS[args.experiment]

    # Compute description string
    total_spatial = sum(
        (BACKBONE_RES[idx] // (2 if cfg.get("reductions", {}).get(idx) else 1)) ** 2
        * cfg["token_dim"]
        for idx in cfg["spatial_indices"]
    )
    total_cond = total_spatial + cfg["global_dim"]
    input_vals = 4 * 48 * 48
    ratio = input_vals / total_cond
    tag = f"{ratio:.1f}× compression" if ratio > 1 else f"{1/ratio:.1f}× expansion"

    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"  {total_cond} conditioning values ({tag})")
    print(f"{'='*60}\n")

    # ---- Data ----
    data_dir = "/mnt/home/ccuesta/ceph/data_for_pablo_legacy_hsc"
    hdf5_path = os.path.join(data_dir, "preprocessed_hsc_legacy_48x48_all.h5")

    train_dataset = HSCLegacyDatasetZoom(hdf5_path=hdf5_path, idx_list=list(range(90_000)))
    val_dataset = HSCLegacyDatasetZoom(hdf5_path=hdf5_path, idx_list=list(range(90_000, 100_000)))
    lense_indices = [3199, 3298, 4368, 4556, 8357, 9503, 19076, 20869, 26247,
                     40506, 51839, 53037, 60565, 60980, 64245, 72326, 74053, 77857, 99695]
    lense_dataset = HSCLegacyDatasetZoomLenses(
        hdf5_path=hdf5_path, lense_indices=lense_indices, is96=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    lense_loader = DataLoader(lense_dataset, batch_size=args.batch_size, num_workers=4)

    # ---- Model ----
    model = FlowMatchingModule(
        experiment_config=cfg,
        lr=1e-4,
        num_sample_images=6,
        num_integration_steps=250,
        num_astropy_lenses=10,
    )

    enc_p = sum(p.numel() for p in model.encoder.parameters())
    unet_p = sum(p.numel() for p in model.conditioned_unet.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Parameter count:")
    print(f"  Encoder:  {enc_p/1e6:.1f}M")
    print(f"  UNet:     {unet_p/1e6:.1f}M")
    print(f"  Total:    {total_p/1e6:.1f}M")

    # ---- W&B ----
    wandb_logger = WandbLogger(project="flow-matching", name=args.experiment, log_model=False)
    wandb_logger.experiment.config.update({
        "experiment": args.experiment,
        **{k: str(v) for k, v in cfg.items()},
        "total_conditioning_values": total_cond,
        "compression": tag,
        "encoder_params_M": enc_p / 1e6,
        "unet_params_M": unet_p / 1e6,
        "total_params_M": total_p / 1e6,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
    })

    # ---- Train ----
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        logger=wandb_logger,
        accelerator="auto",
        devices=args.devices,
        log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
    )
    trainer.fit(model, train_loader, [val_loader, lense_loader])