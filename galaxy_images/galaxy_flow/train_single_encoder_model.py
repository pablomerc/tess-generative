'''
Training script for the single encoder flow matching model for galaxy images.

It expects data already preprocessed.

'''

import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import wandb

current_path=os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0,parent_dir)

from .single_encoder_model import build_model
from .single_encoder_model_concat import build_model_concat
from . import single_encoder_config as cfg
from .hsc_data_loader import HSC_Legacy_DataLoader_OneHot


def _is_concat_model(model):
    """Check if model is a concat model (has decoder_type attribute or is SingleEncoderGalaxyFlowConcat)"""
    if hasattr(model, 'decoder_type'):
        return model.decoder_type == "concat"
    # Check by class name as fallback
    return 'Concat' in model.__class__.__name__


def train_epoch(model, data_loader, optimizer, device, batch_size=cfg.BATCH_SIZE, normalize=False):
    """Train the model for one epoch.

    The single encoder model needs pairs of images (source and target).
    Supports both latent and concat decoder types.
    """

    model.train()
    total_loss = 0
    num_batches = 0
    is_concat = _is_concat_model(model)

    num_samples = cfg.NUM_SAMPLES_PER_EPOCH
    num_batches_epoch = max(1, num_samples // batch_size)

    batch_iter = tqdm(range(num_batches_epoch), desc='Batches', leave=False)
    for batch_idx in batch_iter:
        try:
            hsc_batch, legacy_batch, _, _ = data_loader.get_batch(batch_size=batch_size)
            hsc_batch=hsc_batch.to(device)
            legacy_batch=legacy_batch.to(device)

            if normalize:
                min_val_hsc = hsc_batch.min()
                max_val_hsc = hsc_batch.max()

                hsc_norm = (hsc_batch - min_val_hsc) / ((max_val_hsc - min_val_hsc)+1e-3)
                hsc_flat = hsc_norm.flatten(1)

                min_val_legacy = legacy_batch.min()
                max_val_legacy = legacy_batch.max()
                legacy_batch = (legacy_batch - min_val_legacy) / ((max_val_legacy - min_val_legacy)+1e-3)
            else:
                hsc_flat = hsc_batch.flatten(1)

            # Handle different decoder types
            if is_concat:
                # Concat mode: pass images directly
                loss = model(hsc_flat, cond_images=legacy_batch)
            else:
                # Latent mode: encode first
                z = model.encode(legacy_batch)
                loss = model(hsc_flat, z=z)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            if hasattr(cfg, 'MAX_GRAD_NORM') and cfg.MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update the progress bar with current stats
            batch_iter.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })

        except Exception as e:
            print(f'Error in training batch {batch_idx}: {e}')
            continue

    avg_loss = total_loss / num_batches

    return avg_loss


def train(
    model, data_loader, num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE,
    plots_dir=cfg.PLOTS_DIR, device=None, weight_decay=cfg.WEIGHT_DECAY, use_wandb=True, normalize=False
):
    '''Main training loop'''

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="galaxy-sing-encoder-flow",
            name=f"single_enc_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "batch_size": cfg.BATCH_SIZE,
                "weight_decay": weight_decay,
                "image_size": cfg.IMAGE_SIZE,
                "num_channels": cfg.NUM_CHANNELS,
                "output_dim": cfg.OUTPUT_DIM,
                "velocity_field_type": cfg.VELOCITY_FIELD_TYPE,
                "use_film": cfg.USE_FILM,
                "unet_channels": cfg.UNET_CHANNELS,
                "num_residual_layers": cfg.NUM_RESIDUAL_LAYERS,
                "n_integration_steps": cfg.N_INTEGRATION_STEPS,
                "num_samples_per_epoch": cfg.NUM_SAMPLES_PER_EPOCH,
                "save_interval": cfg.SAVE_INTERVAL,
                "visualization_interval": cfg.VISUALIZATION_INTERVAL,
                "device": str(device),
                "decoder_type": getattr(cfg, 'DECODER_TYPE', 'latent'),
            }
        )
        wandb.watch(model, log="all")
        print("Wandb initialized")

    #Create output directory
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(plots_dir, f'single_enc_flow_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    train_losses=[]
    print(f"Starting training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {run_dir}")


    start_epoch = 0
    pbar = tqdm(range(start_epoch,num_epochs),desc='Training', initial=start_epoch, total=num_epochs)

    for epoch in pbar:
        epoch_start = time.time()

        #Train 1 epoch
        train_loss = train_epoch(model, data_loader, optimizer, device, normalize=normalize)
        train_losses.append(train_loss)

        scheduler.step(train_loss)

        epoch_time = time.time() - epoch_start

        # Print loss
        print(f"\n Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs} - Train Loss: {train_loss:.5f}")

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time,
            }, step=epoch + 1)

        # Update progress bar with loss info
        pbar.set_postfix({
            'loss': f'{train_loss:.4f}',
            'time': f'{epoch_time:.1f}s'
        })

        # Visualization
        if (epoch + 1) % cfg.VISUALIZATION_INTERVAL == 0:
            try:
                visualize_samples(
                    model,
                    device,
                    data_loader,
                    num_examples=4,
                    num_samples=4,
                    save_path=os.path.join(run_dir, f'samples_epoch_{epoch+1}.png'),
                    epoch=epoch + 1,
                    use_wandb=use_wandb,
                    normalize=normalize,
                )
            except Exception as e:
                print(f"Error creating visualization: {e}")

        # Save checkpoint
        if (epoch+1) % cfg.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(run_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("Wandb run finished")


def visualize_samples(model, device, data_loader, num_examples=8, num_samples=8, save_path=None, epoch=None, use_wandb=False, normalize=False):
    '''
    Visualize generated samples for HSC based on Legacy Survey examples.
    Supports both latent and concat decoder types.
    '''

    model.eval()  # Set model to evaluation mode
    batch_size = num_examples
    is_concat = _is_concat_model(model)

    with torch.no_grad():  # Disable gradient computation to save memory
        hsc_batch, legacy_batch, _, _ = data_loader.get_batch(batch_size=batch_size)
        hsc_batch=hsc_batch.to(device)
        legacy_batch=legacy_batch.to(device)

        if normalize:
            min_val_hsc = hsc_batch.min()
            max_val_hsc = hsc_batch.max()

            hsc_norm = (hsc_batch - min_val_hsc) / ((max_val_hsc - min_val_hsc)+1e-3)
            hsc_flat = hsc_norm.flatten(1)

            min_val_legacy = legacy_batch.min()
            max_val_legacy = legacy_batch.max()
            legacy_batch = (legacy_batch - min_val_legacy) / ((max_val_legacy - min_val_legacy)+1e-3)
        else:
            hsc_norm = hsc_batch

        # Clear MPS cache before sampling to free up memory
        if device.type == 'mps':
            torch.mps.empty_cache()

        # Handle different decoder types
        if is_concat:
            # Concat mode: pass images directly
            samples_hsc = model.sample(cond_images=legacy_batch, device=device, n_samples=num_samples)
        else:
            # Latent mode: encode first
            z = model.encode(legacy_batch)
            samples_hsc = model.sample(device=device, n_samples=num_samples, z=z)

        # samples_hsc shape: (n_samples, n_examples, C*H*W) for both modes

        # Clear cache again after sampling
        if device.type == 'mps':
            torch.mps.empty_cache()

        samples_hsc = samples_hsc.view(num_samples, num_examples, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        # Convert to numpy for saving and visualization
        samples_hsc_np = samples_hsc.cpu().numpy()
        hsc_batch_np = hsc_norm.cpu().numpy()
        legacy_batch_np = legacy_batch.cpu().numpy()

    num_rows = num_examples
    num_cols = num_samples + 2  # +2 for target and condition columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2 * num_rows))

    # Ensure axes is 2D array
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)


    #TODO: Check if this is what we want to do or not
    def to_rgb(channels_np):
        """Convert 4-channel (g, r, i, z) to RGB using first 3 channels with per-channel normalization."""
        rgb = np.stack([
            channels_np[0],  # g -> R
            channels_np[1],  # r -> G
            channels_np[2],  # i -> B
        ], axis=-1)  # (H, W, 3)

        for c in range(3):
            ch = rgb[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, :, c] = 0.0
        return rgb


    for j in range(num_examples):
        rgb_gt = to_rgb(hsc_batch_np[j])
        axes[j,0].imshow(rgb_gt, vmin=0, vmax=1)
        axes[j,0].set_title(f'Target (HSC)')

        rgb_cond = to_rgb(legacy_batch_np[j])
        axes[j,1].imshow(rgb_cond, vmin=0, vmax=1)
        axes[j,1].set_title(f'Cond. (Legacy)')

        for i in range(num_samples):
            rgb_hsc = to_rgb(samples_hsc_np[i,j])
            axes[j, i+2].imshow(rgb_hsc, vmin=0, vmax=1)
            axes[j, i+2].set_title(f'Sample {i+1}')
            axes[j, i+2].axis('off')
    plt.suptitle(
        f'Generated Samples (HSC) - Epoch {epoch}'
        if epoch is not None
        else 'Generated Samples (HSC)',
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if use_wandb:
        wandb.log({"generated_samples": wandb.Image(fig)}, step=epoch if epoch is not None else None)

    plt.close(fig)

    return samples_hsc



if __name__ == "__main__":
    print('Entered main function')
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    hdf5_path = cfg.PREPROCESSED_HDF5_PATH

    data_loader = HSC_Legacy_DataLoader_OneHot(
        hdf5_path=hdf5_path,
        seed=42,
        load_to_memory=getattr(cfg, 'LOAD_TO_MEMORY', True),
        max_samples=getattr(cfg, 'MAX_SAMPLES', None),
    )

    # Initialize the model based on decoder type
    decoder_type = getattr(cfg, 'DECODER_TYPE', 'latent')
    if decoder_type == "concat":
        print("Using concat decoder model")
        model = build_model_concat(device=device)
    else:
        print("Using latent decoder model")
        model = build_model(device=device)

    print(f'Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters')

    # Optional: View the number of parameters per part of the model
    # for name, p in model.named_parameters():
    #     print(name, p.shape, p.numel(), p.requires_grad)

    # Train
    use_wandb = getattr(cfg, 'USE_WANDB', True)  # Default to True for backward compatibility
    train(model, data_loader, num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE, plots_dir=cfg.PLOTS_DIR, device=device, use_wandb=use_wandb)

    # Cleanup: close HDF5 file if using HSCDataLoader
    if isinstance(data_loader, HSC_Legacy_DataLoader_OneHot):
        data_loader.close()
