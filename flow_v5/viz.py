import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flow_v5.utils import normalize_to_flow_range, to_visualization_range


def debug_normalization(triplet_creator, device):
    """Check data normalization and ranges."""
    print("\n" + "="*60)
    print("DEBUGGING DATA NORMALIZATION")
    print("="*60)

    (ground_truth, different_digit, same_digit, original_image, original_label, different_label,
     ground_truth_rotation, ground_truth_scale, same_digit_rotation, same_digit_scale) = \
        triplet_creator.create_triplet(dataset='train')

    print(f"\nTensor properties:")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Ground truth dtype: {ground_truth.dtype}")
    print(f"Ground truth min: {ground_truth.min().item():.4f}")
    print(f"Ground truth max: {ground_truth.max().item():.4f}")
    print(f"Ground truth mean: {ground_truth.mean().item():.4f}")
    print(f"Ground truth std: {ground_truth.std().item():.4f}")

    print('\n Transforming to [-1, 1] range \n')
    ground_truth = ground_truth.to(device)
    ground_truth = normalize_to_flow_range(ground_truth)
    print(f"Ground truth min: {ground_truth.min().item():.4f}")
    print(f"Ground truth max: {ground_truth.max().item():.4f}")
    print(f"Ground truth mean: {ground_truth.mean().item():.4f}")
    print(f"Ground truth std: {ground_truth.std().item():.4f}")

    print("="*60)


def create_recon_figure(ground_truth, different_digit, same_digit, reconstruction,
                        original_labels, class_names, epoch):
    """Create reconstruction plot for wandb logging."""
    num_examples = ground_truth.shape[0]
    fig, axes = plt.subplots(4, num_examples, figsize=(2*num_examples + 4, 12))

    for i in range(num_examples):
        axes[0, i].imshow(to_visualization_range(ground_truth[i, 0]).cpu(), cmap='gray')
        axes[0, i].set_title(f'Ground Truth\nLabel: {class_names[original_labels[i]]}')
        axes[0, i].axis('off')

        axes[1, i].imshow(to_visualization_range(different_digit[i, 0]).cpu(), cmap='gray')
        axes[1, i].set_title(f'Different Digit\nFilter Encoder Input')
        axes[1, i].axis('off')

        axes[2, i].imshow(to_visualization_range(same_digit[i, 0]).cpu(), cmap='gray')
        axes[2, i].set_title(f'Same Digit\nNumber Encoder Input')
        axes[2, i].axis('off')

        axes[3, i].imshow(to_visualization_range(reconstruction[i, 0]).cpu(), cmap='gray')
        axes[3, i].set_title(f'Reconstruction\nEpoch {epoch}')
        axes[3, i].axis('off')

    plt.tight_layout()
    return fig


def uncertainty_figure(model, triplet_creator, num_samples=50, num_examples=8):
    """Analyze uncertainty by sampling multiple reconstructions per example."""
    model.eval()

    (ground_truth, different_digit, same_digit, original_labels, different_labels,
     ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
        triplet_creator.create_batch_triplets(num_examples, dataset='test')

    device = next(model.parameters()).device
    ground_truth = normalize_to_flow_range(ground_truth.to(device))
    different_digit = normalize_to_flow_range(different_digit.to(device))
    same_digit = normalize_to_flow_range(same_digit.to(device))

    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx in range(num_examples):
            gt = ground_truth[idx:idx+1]
            sd = same_digit[idx:idx+1]
            dd = different_digit[idx:idx+1]

            combined_z, _, _, _, _, _, _ = model.forward(sd, dd)

            samples = []
            for _ in range(num_samples):
                sample_flat = model.decoder.sample(combined_z, 1)
                sample_img = sample_flat.view(1, 1, model.image_size, model.image_size)
                samples.append(sample_img.cpu().numpy())

            samples = np.stack(samples, axis=0)
            mean_img = samples.mean(axis=0)[0, 0]
            std_img = samples.std(axis=0)[0, 0]

            original_img = gt[0, 0].cpu().numpy()
            diff_img = np.abs(original_img - mean_img)

            im1 = axes[idx, 1].imshow(to_visualization_range(torch.from_numpy(mean_img)).numpy(), cmap='gray', vmin=0, vmax=1)
            axes[idx, 1].set_title(f'Mean of {num_samples} samples')
            axes[idx, 1].axis('off')

            im2 = axes[idx, 2].imshow(std_img, cmap='hot', vmin=0, vmax=np.max(std_img) if np.max(std_img) > 0 else 1)
            axes[idx, 2].set_title('Uncertainty (Std Dev)')
            axes[idx, 2].axis('off')

            im3 = axes[idx, 3].imshow(diff_img, cmap='Reds', vmin=0, vmax=np.max(diff_img) if np.max(diff_img) > 0 else 1)
            axes[idx, 3].set_title('|Original - Mean|')
            axes[idx, 3].axis('off')

            im0 = axes[idx, 0].imshow(to_visualization_range(torch.from_numpy(original_img)).numpy(), cmap='gray', vmin=0, vmax=1)
            axes[idx, 0].set_title('Ground Truth')
            axes[idx, 0].axis('off')

            plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)
            plt.colorbar(im3, ax=axes[idx, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.suptitle(f'Flow Matching Uncertainty Analysis ({num_samples} samples per example)', fontsize=16, y=1.02)
    return fig
