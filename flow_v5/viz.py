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


def uncertainty_figure(model, triplet_creator, num_samples=50, num_examples=8, fixed_encoding=False, data_seed: int | None = 1337, sample_seed: int | None = 7331):
    """Analyze uncertainty by sampling multiple reconstructions per example."""
    model.eval()

    device = next(model.parameters()).device

    # Fix data selection RNGs by default to ensure the same triplets are drawn
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

                samples_list = []
                for _ in range(num_samples):
                    sample_flat = model.decoder.sample(combined_z, 1, generator=gen)
                    sample_img = sample_flat.view(1, 1, model.image_size, model.image_size)
                    samples_list.append(sample_img)
                samples = torch.stack(samples_list, dim=0)  # [N,1,1,H,W]
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
                    samples_list = []
                    for _ in range(num_samples):
                        sample_flat = model.decoder.sample(combined_z, 1, generator=gen)
                        sample_img = sample_flat.view(1, 1, model.image_size, model.image_size)
                        samples_list.append(sample_img)
                    samples = torch.stack(samples_list, dim=0)  # [N,1,1,H,W]
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


def calculate_mean_std_of_samples(model, triplet_creator, num_samples=64, num_examples=16, data_seed: int | None = 1337, sample_seed: int | None = 7331):
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

        # Fix data selection RNGs by default
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
                samples_list = []
                for _ in range(num_samples):
                    sample_flat = model.decoder.sample(combined_z, 1, generator=gen)
                    sample_img = sample_flat.view(1, 1, model.image_size, model.image_size)
                    samples_list.append(sample_img)
                samples = torch.stack(samples_list, dim=0)  # [N,1,1,H,W]
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
