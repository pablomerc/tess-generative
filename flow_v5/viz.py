import torch
import numpy as np
import random
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


def uncertainty_figure(model, triplet_creator, num_samples=50, num_examples=8, fixed_encoding=False, data_seed: int | None = None, sample_seed: int | None = None):
    """Analyze uncertainty by sampling multiple reconstructions per example."""
    model.eval()

    device = next(model.parameters()).device

    # Fix data selection RNGs only if explicitly requested
    if data_seed is not None:
        random.seed(int(data_seed))
        np.random.seed(int(data_seed))
        torch.manual_seed(int(data_seed))

    is_multi = getattr(model, 'multi_samples', False)

    if is_multi:
        try:
            from flow_v5 import config as cfg
            num_filter_augs = getattr(cfg, 'MULTI_NUM_FILTER_AUGS', 2)
            num_number_augs = getattr(cfg, 'MULTI_NUM_NUMBER_AUGS', 2)
        except Exception:
            num_filter_augs = 2
            num_number_augs = 2

        print('num_filter_augs, num_number_augs')
        print(num_filter_augs, num_number_augs)

        batch = triplet_creator.create_batch_multi_triplets(
            batch_size=num_examples, dataset='test',
            num_filter_augs=num_filter_augs, num_number_augs=num_number_augs
        )

        anchor = normalize_to_flow_range(batch["anchor"]).to(device)
        same_number_augments = normalize_to_flow_range(batch["same_number_augments"]).to(device)
        same_filter_augments = normalize_to_flow_range(batch["same_filter_augments"]).to(device)
    else:
        (ground_truth, different_digit, same_digit, original_labels, different_labels,
         ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
            triplet_creator.create_batch_triplets(num_examples, dataset='test')

        ground_truth = normalize_to_flow_range(ground_truth.to(device))
        different_digit = normalize_to_flow_range(different_digit.to(device))
        same_digit = normalize_to_flow_range(same_digit.to(device))

    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx in range(num_examples):
            if is_multi:
                gt = anchor[idx:idx+1]
                # Display the first augmentation from each set for context
                sd_vis = same_number_augments[idx, 0:1]  # [1, C, H, W]
                dd_vis = same_filter_augments[idx, 0:1]  # [1, C, H, W]

                # Build combined latent from augmentation sets
                combined_z, _, _ = model.multi_sample_encoding(
                    same_number_augments[idx:idx+1],
                    same_filter_augments[idx:idx+1]
                )

                # Optional fixed RNG for sampling
                gen = None
                if sample_seed is not None:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(int(sample_seed))

                # Batched sampling: returns [N, B=1, output_dim]
                samples_flat = model.decoder.sample(combined_z, num_samples, generator=gen)
                samples = samples_flat.view(num_samples, 1, 1, model.image_size, model.image_size)
            else:
                gt = ground_truth[idx:idx+1]
                sd = same_digit[idx:idx+1]
                dd = different_digit[idx:idx+1]

                if fixed_encoding:
                    combined_z, _, _, _, _, _, _ = model.forward(sd, dd)
                    gen = None
                    if sample_seed is not None:
                        gen = torch.Generator(device=device)
                        gen.manual_seed(int(sample_seed))
                    samples_flat = model.decoder.sample(combined_z, num_samples, generator=gen)
                    samples = samples_flat.view(num_samples, 1, 1, model.image_size, model.image_size)
                else:
                    # When letting the model re-encode each time internally, we cannot pass a generator
                    samples = model.sample(sd, dd, num_samples=num_samples)  # [N,1,1,H,W]

            mean_img = samples.mean(dim=0)[0, 0].cpu().numpy()
            std_img = samples.std(dim=0)[0, 0].cpu().numpy()

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


def uncertainty_figure_2(model, ground_truth, different_digit, same_digit, is_multi: bool = False, num_samples: int = 64, same_number_augments=None, same_filter_augments=None):
    """
    Analyze uncertainty by sampling multiple reconstructions per example using pre-computed data.

    This function is similar to uncertainty_figure but takes pre-computed ground truth and input data
    instead of generating it from a triplet_creator. Useful for analyzing specific examples or
    when you want to control the exact inputs used for uncertainty analysis.

    Args:
        model: The trained flow matching model
        ground_truth: Target images for reconstruction. Shape: [B, C, H, W] in [-1, 1] range
        different_digit: Different digit with same augmentation (for filter encoder).
                        Shape: [B, C, H, W] in [-1, 1] range
        same_digit: Same digit with different augmentation (for number encoder).
                   Shape: [B, C, H, W] in [-1, 1] range
        is_multi: Whether to use multi-sample encoding mode (default: False)
        num_samples: Number of samples to generate per example (default: 64)
        same_number_augments: For multi-sample mode, tensor of shape [B, N, C, H, W] where N is
                             the number of augmentations of the same digit. Each slice along N
                             is an augmentation of the same digit. Required when is_multi=True.
        same_filter_augments: For multi-sample mode, tensor of shape [B, F, C, H, W] where F is
                             the number of different digits with same augmentation. Each slice
                             along F shares the filter/augmentation but comes from a different
                             digit. Required when is_multi=True.

    Returns:
        matplotlib.figure.Figure: Figure containing uncertainty analysis plots with:
            - Column 0: Ground truth images [B, C, H, W]
            - Column 1: Mean of num_samples reconstructions [B, C, H, W]
            - Column 2: Standard deviation across samples [B, C, H, W] (uncertainty map)
            - Column 3: Absolute difference |Ground Truth - Mean| [B, C, H, W]

    Note:
        In multi-sample mode, same_digit and different_digit are the first augmentations
        from the sets (for visualization), while same_number_augments and same_filter_augments
        are the full augmentation sets used for encoding.
    """

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if is_multi:
            if same_number_augments is None or same_filter_augments is None:
                raise ValueError("For multi-sample mode, same_number_augments and same_filter_augments must be provided")
            # Use the full augmentation sets for encoding
            combined_z, _, _ = model.multi_sample_encoding(
                same_number_augments,  # Full set of number augmentations
                same_filter_augments   # Full set of filter augmentations
            )
            samples_flat = model.decoder.sample(combined_z, num_samples)
            samples = samples_flat.view(num_samples, -1, 1, model.image_size, model.image_size) # [num_samples, batch_size, 1, 28, 28]
            print(f"Samples shape: {samples.shape}")
            print(f"Range of samples: {samples.min().item()}, {samples.max().item()}")
            print(f"Range of ground truth: {ground_truth.min().item()}, {ground_truth.max().item()}")
        else:
            # In single-sample mode, same_digit and different_digit are used directly
            samples = model.sample(same_digit, different_digit, num_samples=num_samples) # [num_samples, batch_size, 1, 28, 28]
            print(f"Samples shape: {samples.shape}")
            print(f"Range of samples: {samples.min().item()}, {samples.max().item()}")
            print(f"Range of ground truth: {ground_truth.min().item()}, {ground_truth.max().item()}")

        mean_img = samples.mean(dim=0) # [batch_size, 1, 28, 28]
        std_img = samples.std(dim=0) # [batch_size, 1, 28, 28]
        diff_img = torch.abs(ground_truth - mean_img) # [batch_size, 1, 28, 28]
        print(f"Mean img shape: {mean_img.shape}")
        print(f"Std img shape: {std_img.shape}")
        print(f"Diff img shape: {diff_img.shape}")
        print(f"Range of mean img: {mean_img.min().item()}, {mean_img.max().item()}")
        print(f"Range of std img: {std_img.min().item()}, {std_img.max().item()}")
        print(f"Range of diff img: {diff_img.min().item()}, {diff_img.max().item()}")

        # Create the figure
        batch_size = ground_truth.shape[0]
        fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)

        # Plot each example
        for idx in range(batch_size):
            # Convert to numpy for plotting
            original_img = to_visualization_range(ground_truth[idx, 0]).cpu().numpy()
            mean_img_np = to_visualization_range(mean_img[idx, 0]).cpu().numpy()
            std_img_np = to_visualization_range(std_img[idx, 0].cpu().numpy())
            diff_img_np = to_visualization_range(diff_img[idx, 0]).cpu().numpy()

            # Ground truth
            im0 = axes[idx, 0].imshow(original_img, cmap='gray')
            axes[idx, 0].set_title('Ground Truth')
            axes[idx, 0].axis('off')

            # Mean reconstruction
            im1 = axes[idx, 1].imshow(mean_img_np, cmap='gray', vmin=0, vmax=1)
            axes[idx, 1].set_title(f'Mean of {num_samples} samples')
            axes[idx, 1].axis('off')

            # Uncertainty (std)
            im2 = axes[idx, 2].imshow(std_img_np, cmap='hot', vmin=0, vmax=max(np.max(std_img_np),0.5))
            axes[idx, 2].set_title('Uncertainty (Std Dev)')
            axes[idx, 2].axis('off')

            # Difference
            im3 = axes[idx, 3].imshow(diff_img_np, cmap='Reds', vmin=0, vmax=np.max(diff_img_np) if np.max(diff_img_np) > 0 else 1)
            axes[idx, 3].set_title('|Original - Mean|')
            axes[idx, 3].axis('off')

            # Add colorbars
            plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)
            plt.colorbar(im3, ax=axes[idx, 3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.suptitle(f'Flow Matching Uncertainty Analysis ({num_samples} samples per example)', fontsize=16, y=1.02)
        return fig



def calculate_mean_std_of_samples(model, triplet_creator, num_samples=64, num_examples=16, data_seed: int | None = None, sample_seed: int | None = None):
    """Calculate the mean and std of samples from the model.
    Inputs:
    - model: the model to use
    - triplet_creator: the triplet creator to use
    - num_samples: the number of samples to generate
    - num_examples: the number of examples to process
    Outputs:
    - mean_std_mean: the mean of the mean stds
    - mean_std_std: the std of the mean stds
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        is_multi = getattr(model, 'multi_samples', False)

        # Fix data selection RNGs only if explicitly requested
        if data_seed is not None:
            random.seed(int(data_seed))
            np.random.seed(int(data_seed))
            torch.manual_seed(int(data_seed))

        if is_multi:
            if data_seed is not None:
                cpu_gen = torch.Generator(device='cpu')
                cpu_gen.manual_seed(int(data_seed))
            try:
                from flow_v5 import config as cfg
                num_filter_augs = getattr(cfg, 'MULTI_NUM_FILTER_AUGS', 2)
                num_number_augs = getattr(cfg, 'MULTI_NUM_NUMBER_AUGS', 2)
            except Exception:
                num_filter_augs = 2
                num_number_augs = 2

            batch = triplet_creator.create_batch_multi_triplets(
                batch_size=num_examples, dataset='test',
                num_filter_augs=num_filter_augs, num_number_augs=num_number_augs
            )

            anchor_batch = normalize_to_flow_range(batch["anchor"]).to(device)
            same_number_augments_batch = normalize_to_flow_range(batch["same_number_augments"]).to(device)
            same_filter_augments_batch = normalize_to_flow_range(batch["same_filter_augments"]).to(device)
        else:
            (
            ground_truth_batch, different_digit_batch, same_digit_batch,
            original_labels, different_labels, ground_truth_rotations,
            ground_truth_scales, same_digit_rotations, same_digit_scales
            ) = triplet_creator.create_batch_triplets(batch_size=num_examples, dataset='test')

            ground_truth_batch = normalize_to_flow_range(ground_truth_batch.to(device))
            different_digit_batch = normalize_to_flow_range(different_digit_batch.to(device))
            same_digit_batch = normalize_to_flow_range(same_digit_batch.to(device))
        mean_std_list = []

        for idx in range(num_examples):
            if is_multi:
                # Build a single combined latent from the augmentation sets for this example
                combined_z, _, _ = model.multi_sample_encoding(
                    same_number_augments_batch[idx:idx+1],
                    same_filter_augments_batch[idx:idx+1]
                )
                # Draw multiple samples from the fixed latent
                gen = None
                if sample_seed is not None:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(int(sample_seed))
                samples_flat = model.decoder.sample(combined_z, num_samples, generator=gen)
                samples = samples_flat.view(num_samples, 1, 1, model.image_size, model.image_size)
            else:
                sd = same_digit_batch[idx:idx+1]
                dd = different_digit_batch[idx:idx+1]
                # For determinism under fixed encoding, we can mirror the approach used above
                # by computing the latent once and passing a generator. Otherwise fall back.
                if sample_seed is not None:
                    combined_z, _, _, _, _, _, _ = model.forward(sd, dd)
                    gen = torch.Generator(device=device)
                    gen.manual_seed(int(sample_seed))
                    samples_list = []
                    for _ in range(num_samples):
                        sample_flat = model.decoder.sample(combined_z, 1, generator=gen)
                        sample_img = sample_flat.view(1, 1, model.image_size, model.image_size)
                        samples_list.append(sample_img)
                    samples = torch.stack(samples_list, dim=0)
                else:
                    samples = model.sample(sd, dd, num_samples=num_samples)
            mean_std = samples.std(dim=0).mean()
            mean_std_list.append(mean_std.item())
    return float(np.mean(mean_std_list)), float(np.std(mean_std_list))


def calculate_reconstruction_error(model, ground_truth, different_digit, same_digit, is_multi: bool = False, num_samples: int = 10, same_number_augments=None, same_filter_augments=None):
    """Calculate the reconstruction error of the model.

    Note: In multi-sample mode, same_digit and different_digit are the first augmentations
    from the sets (for visualization), while same_number_augments and same_filter_augments
    are the full augmentation sets used for encoding.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if is_multi:
            if same_number_augments is None or same_filter_augments is None:
                raise ValueError("For multi-sample mode, same_number_augments and same_filter_augments must be provided")
            # Use the full augmentation sets for encoding
            combined_z, _, _ = model.multi_sample_encoding(
                same_number_augments,  # Full set of number augmentations
                same_filter_augments   # Full set of filter augmentations
            )
            samples_flat = model.decoder.sample(combined_z, num_samples)
            samples = samples_flat.view(num_samples, -1, 1, model.image_size, model.image_size)
        else:
            # In single-sample mode, same_digit and different_digit are used directly
            samples = model.sample(same_digit, different_digit, num_samples=num_samples)

        # print('Start of debugging')
        # # Debugging the range with percentiles
        # def print_range_stats(tensor, name):
        #     min_val = tensor.min().item()
        #     max_val = tensor.max().item()
        #     p10 = torch.quantile(tensor, 0.1).item()
        #     p90 = torch.quantile(tensor, 0.9).item()
        #     print(f"{name} range: [{min_val:.3f}, {max_val:.3f}], p10: {p10:.3f}, p90: {p90:.3f}")

        # print_range_stats(samples, "Samples")
        # print_range_stats(ground_truth, "Ground truth")
        # print_range_stats(different_digit, "Different digit")
        # print_range_stats(same_digit, "Same digit")

        # print('Shape of samples: ', samples.shape)
        # print('Shape of ground truth: ', ground_truth.shape)

        # PyTorch will broadcast: [num_samples, batch_size, 1, 28, 28] - [batch_size, 1, 28, 28]
        # Result: [num_samples, batch_size, 1, 28, 28]
        differences = samples - ground_truth.unsqueeze(0)  # Explicitly add sample dimension
        # print('Shape of differences: ', differences.shape)

        # Calculate reconstruction error for each sample (MSE)
        reconstruction_errors = torch.mean(differences ** 2, dim=(2, 3, 4))  # [num_samples, batch_size]
        # print(f"Reconstruction errors per sample (MSE): {reconstruction_errors.mean(dim=1).tolist()}")

        # Overall mean reconstruction error across all samples
        overall_error = reconstruction_errors.mean()
        print(f"Overall reconstruction error (MSE): {overall_error.item():.6f}")
        print(f'(not used for training - just tracking as a metric)')

        return overall_error.item()


def plot_cls_attention(collected_attns_num, collected_attns_filter, sample_idx=0, layer_idx=None, head_idx=None,
                      token_labels=None, figsize=(16, 6)):
    """
    Visualize what the CLS token attends to across different tokens for both number and filter encoders.

    Args:
        collected_attns_num: Number encoder attention weights with shape [B, num_layers, num_heads, T+1, T+1]
        collected_attns_filter: Filter encoder attention weights with shape [B, num_layers, num_heads, T+1, T+1]
        sample_idx: Which sample in the batch to visualize (default: 0)
        layer_idx: Which layer to show (None = average across all layers)
        head_idx: Which head to show (None = average across all heads)
        token_labels: Custom labels for tokens (default: ['CLS', 'Token 0', ...])
        figsize: Figure size tuple
    """

    if collected_attns_num.shape[0] < 16:
        raise ValueError("collected_attns_num must have at least 16 examples")

    # Process number encoder attention
    attention_num = collected_attns_num.flatten(1, 2)  # [B, num_layers * num_heads, T+1, T+1]
    attention_num = attention_num[:16, :, 0, 1:].mean(1)  # [B, T] - CLS attention to other tokens

    mean_attention_num = attention_num.mean(0)
    std_attention_num = attention_num.std(0)

    attention_num = torch.cat([attention_num, mean_attention_num.unsqueeze(0), std_attention_num.unsqueeze(0)], dim=0)

    # Move to CPU for matplotlib
    attention_num = attention_num.cpu()

    # Process filter encoder attention
    attention_filter = collected_attns_filter.flatten(1, 2)  # [B, num_layers * num_heads, T+1, T+1]
    attention_filter = attention_filter[:16, :, 0, 1:].mean(1)  # [B, T] - CLS attention to other tokens

    mean_attention_filter = attention_filter.mean(0)
    std_attention_filter = attention_filter.std(0)

    attention_filter = torch.cat([attention_filter, mean_attention_filter.unsqueeze(0), std_attention_filter.unsqueeze(0)], dim=0)

    # Move to CPU for matplotlib
    attention_filter = attention_filter.cpu()

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Calculate color scale limits
    vmin_num = min(attention_num.min().item(), 0)
    vmax_num = max(attention_num.max().item(), 0.5)
    vmin_filter = min(attention_filter.min().item(), 0)
    vmax_filter = max(attention_filter.max().item(), 0/5)
    vmin = min(vmin_num, vmin_filter)
    vmax = max(vmax_num, vmax_filter)

    # Number encoder attention
    im1 = ax1.imshow(attention_num, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Token Index')
    ax1.set_ylabel('Batch Examples + Stats')
    ax1.set_title('Number Encoder: CLS Attention to Augmentations')
    ax1.set_yticks(range(attention_num.shape[0]))
    ax1.set_yticklabels([f'Example {i}' for i in range(16)] + ['Mean', 'Std'])

    # Add text annotations for number encoder
    for i in range(attention_num.shape[0]):
        for j in range(attention_num.shape[1]):
            value = attention_num[i, j].item()
            text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
            ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=text_color, fontsize=8, fontweight='bold')

    plt.colorbar(im1, ax=ax1, label='Attention Weight')

    # Filter encoder attention
    im2 = ax2.imshow(attention_filter, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_xlabel('Token Index')
    ax2.set_ylabel('Batch Examples + Stats')
    ax2.set_title('Filter Encoder: CLS Attention to Different Digits')
    ax2.set_yticks(range(attention_filter.shape[0]))
    ax2.set_yticklabels([f'Example {i}' for i in range(16)] + ['Mean', 'Std'])

    # Add text annotations for filter encoder
    for i in range(attention_filter.shape[0]):
        for j in range(attention_filter.shape[1]):
            value = attention_filter[i, j].item()
            text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
            ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=text_color, fontsize=8, fontweight='bold')

    plt.colorbar(im2, ax=ax2, label='Attention Weight')

    plt.tight_layout()
    return fig


def test_plot_cls_attention():
    collected_attns_num = torch.randn(16, 2, 4, 6, 6)*0.5
    collected_attns_filter = torch.randn(16, 2, 4, 6, 6)*0.5
    fig = plot_cls_attention(collected_attns_num, collected_attns_filter)
    fig.savefig('cls_attention.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved cls attention plot to cls_attention.png')

if __name__ == '__main__':
    test_plot_cls_attention()
