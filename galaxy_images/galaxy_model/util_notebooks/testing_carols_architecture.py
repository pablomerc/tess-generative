import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import timm
from diffusers import UNet2DConditionModel, UNet2DModel
from typing import Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


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


class ConditionalFlowMatchingModule(pl.LightningModule):
    """
    Conditional Flow Matching model using optimal transport conditional paths.

    Conditions on a second image (e.g., Legacy Survey) to generate the target
    image (e.g., HSC).

    The interpolation is: x_t = (1 - t) * x_0 + t * x_1
    where x_0 ~ N(0, I) (noise), x_1 ~ target data.

    The target velocity is: v(x_t, t, c) = x_1 - x_0
    where c is the conditioning image embedding.

    Args:
        concat_conditioning: If True, concatenate conditioning image directly to
            input (no encoder, uses UNet2DModel). If False, use ResNet encoder
            with cross-attention (uses UNet2DConditionModel).
    """

    def __init__(
        self,
        #DATA PARAMS
        in_channels: int = 4, # Channels in the target domain (image)
        cond_channels: int = 4, # Encoder input channels or concatenated channels for conditioning
        image_size: int = 64, # Spatial size of the image
        #UNET PARAMS
        model_channels: int = 128, # Base channel width for the unet
        channel_mult: tuple = (1, 2, 4, 4), # channel multiplier for each block. we downsample spatially and increase the channels
        layers_per_block: int = 2, # resnet-like layers in each unet block
        attention_head_dim: int = 8, # head dimension used by attention blocks inside diffusers unet
        # Conditioning params
        cross_attention_dim: int = 256, # cross-attention mode (conditionning mode). must match the resnet encoder cross att dim and the unet encoding dim
        pretrained_encoder: bool = False, # load pretrained imagenet weights
        concat_conditioning: bool = False, # if true -> no encoder, conditioning is concatenated as extra channels to the input
        # Optimization params
        lr: float = 1e-4,
        num_sample_images: int = 8, # number of exmaples cached for first validation batch for W&B
        num_integration_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_sample_images = num_sample_images
        self.num_integration_steps = num_integration_steps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.image_size = image_size
        self.concat_conditioning = concat_conditioning

        block_out_channels = tuple(model_channels * m for m in channel_mult)

        if concat_conditioning:
            self.encoder = None

            self.velocity_model = UNet2DModel(
                sample_size=image_size,
                in_channels=in_channels + cond_channels,  # x_t + cond concatenated
                out_channels=in_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                ),
                attention_head_dim=attention_head_dim,
            )
        else:
            self.encoder = ResNetEncoder(
                in_channels=cond_channels,
                cross_attention_dim=cross_attention_dim,
                pretrained=pretrained_encoder,
            )

            self.velocity_model = UNet2DConditionModel(
                sample_size=image_size,
                in_channels=in_channels,
                out_channels=in_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=(
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=cross_attention_dim,
                attention_head_dim=attention_head_dim,
            )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t, c).

        Args:
            x_t: Noisy image at time t (B, C, H, W)
            t: Time in [0, 1] (B,)
            cond_image: Conditioning image (B, C, H, W)
        """
        timesteps = t * 1000

        if self.concat_conditioning:
            x_input = torch.cat([x_t, cond_image], dim=1)
            return self.velocity_model(x_input, timesteps).sample
        else:
            cond_embedding = self.encoder(cond_image)  # (B, seq_len, embed_dim)
            return self.velocity_model(
                x_t,
                timesteps,
                encoder_hidden_states=cond_embedding,
            ).sample

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """Compute conditional flow matching loss."""
        x_1, cond_image = batch
        batch_size = x_1.shape[0]

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=x_1.device)

        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        target_velocity = x_1 - x_0

        predicted_velocity = self(x_t, t, cond_image)

        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        if batch_idx == 0:
            self._val_cond_batch = batch[1][:self.num_sample_images].clone()
            self._val_target_batch = batch[0][:self.num_sample_images].clone()

        return loss

    @torch.no_grad()
    def sample(
        self,
        cond_images: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples conditioned on input images using Euler integration.

        Args:
            cond_images: Conditioning images (B, C, H, W)
            num_steps: Number of integration steps
        """
        num_steps = num_steps or self.num_integration_steps
        num_samples = cond_images.shape[0]
        device = cond_images.device

        x = torch.randn(
            num_samples, self.in_channels, self.image_size, self.image_size,
            device=device,
        )

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            velocity = self(x, t, cond_images)
            x = x + velocity * dt

        return x

    def _normalize_for_vis(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] for visualization."""
        img = img.clone()
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return img

    def on_validation_epoch_end(self) -> None:
        """Log sampled images as a grid to W&B.

        Creates a grid where each row corresponds to one conditioning image:
        [Cond | Target | Sample1 | Sample2 | ... | SampleN | Mean]
        """
        if not self.logger or not hasattr(self, '_val_cond_batch'):
            return

        import matplotlib.pyplot as plt

        num_cond_images = min(6, len(self._val_cond_batch))
        num_samples_per_cond = 5
        num_cols = 2 + num_samples_per_cond + 1  # cond + target + samples + mean

        fig, axes = plt.subplots(
            num_cond_images, num_cols,
            figsize=(2 * num_cols, 2 * num_cond_images),
            squeeze=False,
        )

        col_titles = ["Cond", "Target"] + [f"Sample {j+1}" for j in range(num_samples_per_cond)] + ["Mean"]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=10)

        for i in range(num_cond_images):
            cond = self._val_cond_batch[i:i+1].to(self.device)
            target = self._val_target_batch[i:i+1].to(self.device)

            cond_repeated = cond.repeat(num_samples_per_cond, 1, 1, 1)
            samples = self.sample(cond_repeated)
            mean_sample = samples.mean(dim=0, keepdim=True)

            cond_rgb = self._normalize_for_vis(cond[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(cond_rgb)
            axes[i, 0].axis('off')

            target_rgb = self._normalize_for_vis(target[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes[i, 1].imshow(target_rgb)
            axes[i, 1].axis('off')

            for j in range(num_samples_per_cond):
                sample_rgb = self._normalize_for_vis(samples[j, :3]).cpu().permute(1, 2, 0).numpy()
                axes[i, 2 + j].imshow(sample_rgb)
                axes[i, 2 + j].axis('off')

            mean_rgb = self._normalize_for_vis(mean_sample[0, :3]).cpu().permute(1, 2, 0).numpy()
            axes[i, -1].imshow(mean_rgb)
            axes[i, -1].axis('off')

        plt.tight_layout()

        self.logger.experiment.log({
            "val/sample_grid": wandb.Image(fig),
            "global_step": self.global_step,
        })

        plt.close(fig)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



if __name__ == "__main__":

    dummy_image = torch.randn(10, 4, 48, 48)
    encoder = ResNetEncoder()

    print(f'Encoder model parameters: {sum(p.numel() for p in encoder.parameters()):,}')
    # print(encoder(dummy_image).shape)
    image_size = 48
    in_channels = 4
    layers_per_block = 2
    block_out_channels = (128, 256, 512, 512)
    cross_attention_dim = 512
    attention_head_dim = 8

    decoder = UNet2DConditionModel(
                sample_size=image_size,
                in_channels=in_channels,
                out_channels=in_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=(
                    "DownBlock2D",
                    # "CrossAttnDownBlock2D",
                    # "CrossAttnDownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                ),
                mid_block_type='UNetMidBlock2D',
                up_block_types=(
                    "UpBlock2D",
                    # "CrossAttnUpBlock2D",
                    # "CrossAttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=cross_attention_dim,
                attention_head_dim=attention_head_dim,
            )
    print(f'Decoder total model parameters: {sum(p.numel() for p in decoder.parameters()):,}')
    # print(decoder)

    def count_params(module: nn.Module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print("\n=== DOWN BLOCKS ===")
    for i, block in enumerate(decoder.down_blocks):
        print(f"down_blocks[{i}] ({block.__class__.__name__}): "
            f"{count_params(block):,}")

    print("\n=== MID BLOCK ===")
    print(f"mid_block ({decoder.mid_block.__class__.__name__}): "
        f"{count_params(decoder.mid_block):,}")

    print("\n=== UP BLOCKS ===")
    for i, block in enumerate(decoder.up_blocks):
        print(f"up_blocks[{i}] ({block.__class__.__name__}): "
            f"{count_params(block):,}")


    x = torch.rand(50,5,4,48,48)

    z = encoder(x)

    print(z.shape)


    # print(torch.rand((10,4,48,48)).shape)
#(48,48) returns (10,4,256)
#(96,96) returns (10,9,256)
