import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from diffusers import UNet2DModel
from typing import Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt


class ConditionalFlowMatchingModule(pl.LightningModule):
    """
    Conditional Flow Matching model using optimal transport conditional paths.
    Conditions on 4 scalar values by concatenating scalar feature maps
    to the UNet input channels.
    """

    def __init__(
        self,
        # DATA PARAMS
        in_channels: int = 4,
        cond_channels: int = 4, # Number of input scalars
        img_height: int = 64,
        img_width: int = 64,
        # UNET PARAMS
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        # OPTIMIZATION
        lr: float = 1e-4,
        num_sample_images: int = 8,
        num_integration_steps: int = 250,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_sample_images = num_sample_images
        self.num_integration_steps = num_integration_steps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.image_height, self.image_width = self._validate_image_size(
            img_height, img_width
        )
        self.image_size = (self.image_height, self.image_width)

        block_out_channels = tuple(model_channels * m for m in channel_mult)


        self.velocity_model = UNet2DModel(
            sample_size=self.image_size,
            in_channels=in_channels + cond_channels,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,

            # Standard ResNet/Attention blocks
            down_block_types=(
                "DownBlock2D",      # ResNet only
                "AttnDownBlock2D",  # ResNet + Self-Attention
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

        #You might need to change it for:
        # down_block_types=(
        #             "DownBlock2D",
        #             "DownBlock2D",
        #             "DownBlock2D",
        #             "DownBlock2D",
        #         ),
        #         mid_block_type='UNetMidBlock2D',
        #         up_block_types=(
        #             "UpBlock2D",
        #             "UpBlock2D",
        #             "UpBlock2D",
        #             "UpBlock2D",
        #         ),
        # but im not sure

    @staticmethod
    def _validate_image_size(img_height: int, img_width: int):
        if not isinstance(img_height, int) or not isinstance(img_width, int):
            raise ValueError("img_height and img_width must be integers.")

        if img_height <= 0 or img_width <= 0:
            raise ValueError("img_height and img_width must be positive.")

        # With 4 down/up blocks, dimensions should be divisible by 2^(4-1)=8.
        if img_height % 8 != 0 or img_width % 8 != 0:
            raise ValueError(
                "img_height and img_width must both be divisible by 8 for this UNet config, "
                f"got {(img_height, img_width)}."
            )

        return img_height, img_width

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_scalars: torch.Tensor, # Shape: (B, 4)
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t, c).
        """
        # UNet2DModel expects timesteps to be scaled roughly to 0-1000 range usually
        # but pure Flow Matching often works with [0,1]. Diffusers defaults usually prefer larger ints.
        timesteps = t * 1000
        batch_size, _, height, width = x_t.shape
        cond_scalars = cond_scalars.to(dtype=x_t.dtype)
        cond_map = cond_scalars[:, :, None, None].expand(
            batch_size, self.cond_channels, height, width
        )
        model_input = torch.cat([x_t, cond_map], dim=1)

        return self.velocity_model(model_input, timesteps).sample

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        x_1, cond_scalars = batch # Expects (Image, Scalars)
        batch_size = x_1.shape[0]

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=x_1.device)

        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        target_velocity = x_1 - x_0

        # Predict
        predicted_velocity = self(x_t, t, cond_scalars)

        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Cache for visualization - always visualizae the same validation batch
        if batch_idx == 0:
            self._val_cond_batch = batch[1][:self.num_sample_images].clone()
            self._val_target_batch = batch[0][:self.num_sample_images].clone()
        return loss

    @torch.no_grad()
    def sample(self, cond_scalars: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        num_steps = num_steps or self.num_integration_steps
        num_samples = cond_scalars.shape[0]
        device = cond_scalars.device

        # Start from Noise
        x = torch.randn(
            num_samples, self.in_channels, self.image_height, self.image_width,
            device=device,
        )

        dt = 1.0 / num_steps

        # Euler Integration
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            velocity = self(x, t, cond_scalars)
            x = x + velocity * dt

        return x

    def on_validation_epoch_end(self) -> None:
        """
        Log visualization: Target Image vs Generated Samples
        Labels Y-axis with the scalar values used.
        """
        if not self.logger or not hasattr(self, "_val_cond_batch"):
            return

        num_conds = min(6, len(self._val_cond_batch))
        num_samples_per_cond = 4
        num_cols = 1 + num_samples_per_cond # Target + Samples

        def _row_scale(x, vmin, vmax):
            # Scale (C,H,W) to (H,W,C) for matplotlib
            x = x[:3].permute(1, 2, 0) # Only RGB
            # normalize to 0-1 based on target range
            return ((x - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1).cpu().numpy()

        fig, axes = plt.subplots(
            num_conds, num_cols,
            figsize=(3 * num_cols, 3 * num_conds),
            squeeze=False,
        )

        # Set column titles
        titles = ["Target"] + [f"Sample {j+1}" for j in range(num_samples_per_cond)]
        for j, t in enumerate(titles):
            axes[0, j].set_title(t)

        for i in range(num_conds):
            scalars = self._val_cond_batch[i:i+1].to(self.device)
            target = self._val_target_batch[i].to(self.device)

            # Generate
            samples = self.sample(scalars.repeat(num_samples_per_cond, 1))

            # Determine visual range from target
            # Note: Assuming image is (C,H,W) and we view as RGB
            target_rgb = target[:3].permute(1, 2, 0)
            vmin, vmax = target_rgb.min(), target_rgb.max()

            # Plot Target
            axes[i, 0].imshow(_row_scale(target, vmin, vmax))

            # Add Scalar text label
            scalar_txt = "\n".join([f"{v:.2f}" for v in scalars[0].cpu().numpy()])
            axes[i, 0].set_ylabel(f"Input:\n{scalar_txt}", rotation=0, labelpad=50, va='center')
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            # Plot Samples
            for j in range(num_samples_per_cond):
                axes[i, j+1].imshow(_row_scale(samples[j], vmin, vmax))
                axes[i, j+1].axis("off")

        plt.tight_layout()

        self.logger.experiment.log({
            "val/samples": wandb.Image(fig),
            "global_step": self.global_step,
        })
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



if __name__ == "__main__":
    from pytorch_lightning.loggers import WandbLogger
    from torch.utils.data import DataLoader, TensorDataset
    from data import HSCLegacyDataset # this is a dataset for the example, replace it with your own

    batch_size = 64

    ### Replace this with your own Dataset class
    # in the __getitem__ method it should return a tuple of (image, scalars)

    train_dataset = HSCLegacyDataset(
        hdf5_path='/mnt/home/ccuesta/ceph/legacysurvey_hsc_crossmatched/pablos_data/preprocessed_hsc_legacy_48x48.h5',
        idx_list=list(range(5000)),
    )
    val_dataset = HSCLegacyDataset(
        hdf5_path='/mnt/home/ccuesta/ceph/legacysurvey_hsc_crossmatched/pablos_data/preprocessed_hsc_legacy_48x48.h5',
        idx_list=list(range(5000, 5140)),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    model = ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        img_height=48,
        img_width=48,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        lr=1e-4,
        num_sample_images=6,
        num_integration_steps=250,
    )

    wandb_logger = WandbLogger(
        project="flow-matching",
        name="conditional-unet2d-scalars",
        log_model=False,
    )

    n_devices = 1 # number of GPUs, pytorch lightning will automatically handle multiple GPUs
    trainer = pl.Trainer(
        max_steps=300_000/n_devices, # maybe this is too large for you and you don't need to train that long
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model, train_loader, val_loader)
