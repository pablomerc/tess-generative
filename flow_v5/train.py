from typing import Tuple
import time
from datetime import datetime, timedelta
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from flow_v5.utils import normalize_to_flow_range
from flow_v5.viz import create_recon_figure, uncertainty_figure, calculate_mean_std_of_samples, calculate_reconstruction_error


def train(model, triplet_creator, num_epochs: int, lr: float, plots_dir: str, start_epoch: int = 0, multi_samples: bool = False) -> Tuple[list, list]:
    """Training loop migrated for v5 modules.

    start_epoch: number of epochs the loaded model has already been trained for.
                 Used to keep cumulative epoch counting in logs, plots, and filenames.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    max_grad_norm = 0.1

    train_losses, test_losses = [], []
    start_time = time.time()

    device = next(model.parameters()).device
    BATCH_SIZE = wandb.config.get("batch_size", 128)

    # Create a fixed validation batch for consistent reconstruction error calculation
    print("Creating fixed validation batch for reconstruction error...")
    if multi_samples:
        try:
            from flow_v5 import config as cfg
            num_filter_augs = getattr(cfg, 'MULTI_NUM_FILTER_AUGS', 2)
            num_number_augs = getattr(cfg, 'MULTI_NUM_NUMBER_AUGS', 2)
        except Exception:
            num_filter_augs = 2
            num_number_augs = 2

        fixed_batch = triplet_creator.create_batch_multi_triplets(
            batch_size=32, dataset='test',  # Use smaller batch for reconstruction error
            num_filter_augs=num_filter_augs, num_number_augs=num_number_augs
        )
        fixed_ground_truth = normalize_to_flow_range(fixed_batch["anchor"]).to(device)
        fixed_same_number_augments = normalize_to_flow_range(fixed_batch["same_number_augments"]).to(device)
        fixed_same_filter_augments = normalize_to_flow_range(fixed_batch["same_filter_augments"]).to(device)
        # For visualization/debugging: use first augmentation from each set
        fixed_different_digit = fixed_same_filter_augments[:, 0]  # First filter augmentation
        fixed_same_digit = fixed_same_number_augments[:, 0]  # First number augmentation
    else:
        (fixed_ground_truth, fixed_different_digit, fixed_same_digit, _, _, _, _, _, _) = \
            triplet_creator.create_batch_triplets(32, dataset='test')  # Use smaller batch for reconstruction error

        fixed_ground_truth = normalize_to_flow_range(fixed_ground_truth.to(device))
        fixed_different_digit = normalize_to_flow_range(fixed_different_digit.to(device))
        fixed_same_digit = normalize_to_flow_range(fixed_same_digit.to(device))

    print(f"Fixed validation batch created - Ground truth shape: {fixed_ground_truth.shape}")

    # Namespace plots/checkpoints by dataset and run
    dataset_dir = os.path.join(plots_dir, str(getattr(triplet_creator, 'dataset_type', 'unknown')))
    os.makedirs(dataset_dir, exist_ok=True)
    # Use wandb run name if available, otherwise timestamp
    run_name = None
    if wandb.run is not None and hasattr(wandb.run, 'name') and wandb.run.name:
        run_name = wandb.run.name
    else:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(dataset_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        num_batches = 0

        total_train_samples = 60000
        num_batches_epoch = total_train_samples // BATCH_SIZE
        total_epoch = start_epoch + epoch + 1
        print(f"Epoch {total_epoch} (this run {epoch+1}/{num_epochs}): Processing {num_batches_epoch} batches ({total_train_samples} samples)")

        for batch_idx in range(num_batches_epoch):
            if multi_samples:
                # Get augmentation counts from config
                from flow_v5 import config as cfg
                num_filter_augs = getattr(cfg, 'MULTI_NUM_FILTER_AUGS', 2)
                num_number_augs = getattr(cfg, 'MULTI_NUM_NUMBER_AUGS', 2)

                batch = triplet_creator.create_batch_multi_triplets(
                    batch_size=BATCH_SIZE, dataset='train',
                    num_filter_augs=num_filter_augs, num_number_augs=num_number_augs
                )

                anchor = normalize_to_flow_range(batch["anchor"]).to(device)
                same_number_augments = normalize_to_flow_range(batch["same_number_augments"]).to(device)
                same_filter_augments = normalize_to_flow_range(batch["same_filter_augments"]).to(device)
                anchor_labels = batch["anchor_labels"].to(device)
                filter_labels = batch["filter_labels"].to(device)
            else:
                (ground_truth, different_digit, same_digit, original_labels, different_labels,
                ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                    triplet_creator.create_batch_triplets(BATCH_SIZE, dataset='train')

                ground_truth = normalize_to_flow_range(ground_truth.to(device))
                different_digit = normalize_to_flow_range(different_digit.to(device))
                same_digit = normalize_to_flow_range(same_digit.to(device))
                original_labels = original_labels.to(device)
                different_labels = different_labels.to(device)

            optimizer.zero_grad()

            if multi_samples:
                flow_loss = model.get_flow_loss_multi(
                    same_number_augments,
                    same_filter_augments,
                    anchor,
                )
            else:
                flow_loss, number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = \
                    model.get_flow_loss(same_digit, different_digit, ground_truth)

            if torch.isnan(flow_loss) or torch.isinf(flow_loss) or flow_loss > 1000:
                print(f"WARNING: Loss explosion detected at batch index {batch_idx}. Loss: {flow_loss.item()}")
                print("Skipping this batch and continuing...")
                continue

            flow_loss.backward()

            total_grad_norm = 0
            max_grad = 0
            nan_grads = 0
            inf_grads = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    total_grad_norm += grad_norm.item() ** 2
                    max_grad = max(max_grad, param.grad.data.abs().max().item())
                    if torch.isnan(param.grad).any():
                        nan_grads += 1
                    if torch.isinf(param.grad).any():
                        inf_grads += 1

            total_grad_norm = total_grad_norm ** 0.5

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}: Loss={flow_loss.item():.4f}, GradNorm={total_grad_norm:.4f}, MaxGrad={max_grad:.4f}, NaN grads: {nan_grads}, Inf grads: {inf_grads}")

            if total_grad_norm > 100 or max_grad > 50 or nan_grads > 0 or inf_grads > 0:
                print(f"WARNING: Gradient explosion at batch {batch_idx}!")
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss += flow_loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')

        model.eval()
        test_loss = 0
        valid_test_batches = 0
        with torch.no_grad():
            total_val_samples = 10000
            num_val_batches = total_val_samples // BATCH_SIZE
            print(f"Validation (after epoch {total_epoch}): Processing {num_val_batches} batches ({total_val_samples} samples)")
            for batch_idx in range(num_val_batches):
                if multi_samples:
                    # Get augmentation counts from config (same as training)
                    from flow_v5 import config as cfg
                    num_filter_augs = getattr(cfg, 'MULTI_NUM_FILTER_AUGS', 2)
                    num_number_augs = getattr(cfg, 'MULTI_NUM_NUMBER_AUGS', 2)

                    batch = triplet_creator.create_batch_multi_triplets(
                        batch_size=BATCH_SIZE, dataset='test',
                        num_filter_augs=num_filter_augs, num_number_augs=num_number_augs
                    )

                    anchor = normalize_to_flow_range(batch["anchor"]).to(device)
                    same_number_augments = normalize_to_flow_range(batch["same_number_augments"]).to(device)
                    same_filter_augments = normalize_to_flow_range(batch["same_filter_augments"]).to(device)

                    flow_loss = model.get_flow_loss_multi(
                        same_number_augments,
                        same_filter_augments,
                        anchor,
                    )
                else:
                    (ground_truth, different_digit, same_digit, original_labels, different_labels,
                     ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                        triplet_creator.create_batch_triplets(BATCH_SIZE, dataset='test')

                    ground_truth = normalize_to_flow_range(ground_truth.to(device))
                    different_digit = normalize_to_flow_range(different_digit.to(device))
                    same_digit = normalize_to_flow_range(same_digit.to(device))
                    original_labels = original_labels.to(device)
                    different_labels = different_labels.to(device)

                    flow_loss, _, _, _, _, _, _ = model.get_flow_loss(same_digit, different_digit, ground_truth)
                if not (torch.isnan(flow_loss) or torch.isinf(flow_loss) or flow_loss > 1000):
                    test_loss += flow_loss.item()
                    valid_test_batches += 1

        avg_test_loss = test_loss / valid_test_batches if valid_test_batches > 0 else float('inf')

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        scheduler.step(avg_train_loss)


        # Compute reconstruction error and log to wandb using fixed validation batch
        try:
            print(f"Computing reconstruction error for epoch {total_epoch}...")
            if multi_samples:
                reconstruction_error = calculate_reconstruction_error(
                    model, fixed_ground_truth, fixed_different_digit, fixed_same_digit,
                    is_multi=multi_samples, num_samples=10,
                    same_number_augments=fixed_same_number_augments,
                    same_filter_augments=fixed_same_filter_augments
                )
            else:
                reconstruction_error = calculate_reconstruction_error(
                    model, fixed_ground_truth, fixed_different_digit, fixed_same_digit,
                    is_multi=multi_samples, num_samples=10
                )
            wandb.log({"reconstruction_error": reconstruction_error}, step=total_epoch)
        except Exception as e:
            print(f"Error computing reconstruction error: {e}")




        # Compute uncertainty metric from sampling and log to wandb
        try:
            sample_std_mean, sample_std_std = calculate_mean_std_of_samples(
                model, triplet_creator, num_samples=64, num_examples=16
            )
        except Exception as e:
            print(f"Error computing sample std metrics: {e}")
            sample_std_mean, sample_std_std = float('nan'), float('nan')

        wandb.log({
            "epoch": total_epoch,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "learning_rate": lr,
            "sample_std_mean": sample_std_mean,
            "sample_std_std": sample_std_std,
        }, step=total_epoch)

        if (epoch + 1) % 5 == 0:
            print(f"Creating reconstruction plot for epoch {total_epoch}...")
            try:
                model.eval()
                with torch.no_grad():
                    vis_batch_size = 8
                    if multi_samples:
                        batch = triplet_creator.create_batch_multi_triplets(
                            batch_size=vis_batch_size, dataset='train'
                        )

                        ground_truth = normalize_to_flow_range(batch["anchor"]).to(device)
                        same_number_augments = normalize_to_flow_range(batch["same_number_augments"]).to(device)
                        same_filter_augments = normalize_to_flow_range(batch["same_filter_augments"]).to(device)
                        original_labels = batch["anchor_labels"].to(device)

                        combined_z, _, _ = model.multi_sample_encoding(
                            same_number_augments,
                            same_filter_augments,
                        )
                        reconstruction_flat = model.decoder.sample(combined_z, 1)
                        reconstruction = reconstruction_flat.view(ground_truth.shape)

                        # Use the first augmentation from each set for visualization inputs
                        different_digit = same_filter_augments[:, 0]
                        same_digit = same_number_augments[:, 0]
                        #TODO: Instead of just passing on of the examples, make a custom visualizatoin function and show all of them?
                    else:
                        (ground_truth, different_digit, same_digit, original_labels, different_labels,
                         ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                            triplet_creator.create_batch_triplets(vis_batch_size, dataset='train')

                        ground_truth = normalize_to_flow_range(ground_truth.to(device))
                        different_digit = normalize_to_flow_range(different_digit.to(device))
                        same_digit = normalize_to_flow_range(same_digit.to(device))
                        original_labels = original_labels.to(device)

                        reconstruction = model.reconstruct(same_digit, different_digit)

                    reconstruction_fig = create_recon_figure(
                        ground_truth, different_digit, same_digit, reconstruction,
                        original_labels, triplet_creator.class_names, total_epoch
                    )
                    wandb.log({"reconstructions": wandb.Image(reconstruction_fig)}, step=total_epoch)
                    plt.close(reconstruction_fig)
            except Exception as e:
                print(f"Error creating reconstruction plot: {e}")

        if (epoch + 1) % 5 == 0:
            print(f"Creating uncertainty analysis for epoch {total_epoch}...")
            try:
                fig = uncertainty_figure(model, triplet_creator, num_samples=50, num_examples=8)
                wandb.log({"uncertainty_analysis": wandb.Image(fig)}, step=total_epoch)
                uncertainty_filename = os.path.join(run_dir, f"uncertainty_epoch_{total_epoch:03d}.png")
                fig.savefig(uncertainty_filename, dpi=150, bbox_inches='tight')
                wandb.log({"uncertainty_analysis_file": wandb.Image(uncertainty_filename)}, step=total_epoch)
                plt.close(fig)
                print(f"Saved uncertainty plot to: {uncertainty_filename}")
            except Exception as e:
                print(f"Error creating uncertainty analysis: {e}")

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        print(f'Epoch [{total_epoch}] (this run {epoch+1}/{num_epochs}), Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Time: {epoch_time:.1f}s, Total: {str(timedelta(seconds=int(total_time)))}')

    print(f"Training completed in {str(timedelta(seconds=int(time.time() - start_time)))}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_total_epochs = start_epoch + num_epochs
    model_save_path = os.path.join(run_dir, f"double_encoder_flow_model_{getattr(triplet_creator, 'dataset_type', 'unknown')}_epoch_{final_total_epochs}_{timestamp}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'epochs_trained': final_total_epochs,
        'config': {
            'number_latent_dim': model.number_latent_dim,
            'filter_latent_dim': model.filter_latent_dim,
            'image_size': model.image_size,
        }
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")

    return train_losses, test_losses
