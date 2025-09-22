'''
This code will load a pretrained model.
Then take a batch of 10 triplets.
Run the encoders once to extract latents.
Then sample the model 11 times to generate 11 unique reconstructions.
For each of the 10 examples in the batch save a plot that is

3x4=12 images with the top left one being the ground truth and all
the others being reconstructions.

This should yield 10 plots that will be saved in flow_models/generation_study/<pretrained_model_name>/<todays_date>

Optionally, it will also run the generation 100 times.
Then take the 100 generated examples and do a UMAP dimensionality reduction.
Generate a UMAP plot for each example in the batch and save it in the same directory
'''

import os
import math
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    import umap.umap_ as umap
except Exception:
    umap = None
from flow_v5.data import make_triplet_creator
from flow_v5.model import build_model
from flow_v5.utils import normalize_to_flow_range, to_visualization_range

# path_pretrain = "../flow_decoder/reconstruction_plots_v5_mnist/double_encoder_flow_model_mnist_200.pth"
# path_pretrain = "flow_decoder/reconstruction_plots_v5_mnist/double_encoder_flow_model_mnist_200.pth"
path_pretrain='flow_models/mnist/double-encoder-flow-mnist-v5-20250922_105826/double_encoder_flow_model_mnist_epoch_50_20250922_144024.pth'
# generate_umaps = False

n_examples = 10
n_samples = 11
n_umap = 100
output_root='flow_models'
random_seed=42
dataset_type='mnist'
train_or_test='test'
fixed_encoding = True
test_determinism = True
plot_samples = False
plot_umap = False


# random.seed(random_seed); np.random.seed(random_seed); torch.manual_seed(random_seed)
# if torch.cuda.is_available():
    # torch.cuda.manual_seed_all(random_seed)

### Prepare triplet functions
triplet_creator = make_triplet_creator(dataset_type=dataset_type)
triplet_creator.get_dataset_info()

### Choose available device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

### Initialize model
model=build_model(device=device)

### Load pretrained model
if os.path.exists(path_pretrain):
    try:
        print(f"Loading pretrained weights from: {path_pretrain}")
        checkpoint = torch.load(path_pretrain, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    except Exception as e:
        print(f"WARNING: Failed to load pretrained weights from {path_pretrain}: {e}")
        print("Proceeding with training from scratch.")
else:
    print(f"WARNING: Pretrained checkpoint not found at: {path_pretrain}")
    print("Proceeding with training from scratch.")

#Turn model into evaluation mode
model.eval()

### Create a batch of 10 triplets



print(f'Creating {n_examples} triplets')
print(f'and extracting latents for the triplets')

model.eval()
with torch.no_grad():
    (
    ground_truth_batch, different_digit_batch, same_digit_batch,
    original_labels, different_labels, ground_truth_rotations,
    ground_truth_scales, same_digit_rotations, same_digit_scales
    ) = triplet_creator.create_batch_triplets(batch_size=n_examples, dataset=train_or_test)

    ground_truth_batch = normalize_to_flow_range(ground_truth_batch.to(device))
    different_digit_batch = normalize_to_flow_range(different_digit_batch.to(device))
    same_digit_batch = normalize_to_flow_range(same_digit_batch.to(device))
    # original_labels = original_labels.to(device)

    #Iterate over the number of examples to be generated


    mean_std_list = []
    for idx in range(n_examples):
        #Take the idx-th example
        gt = ground_truth_batch[idx:idx+1]
        sd = same_digit_batch[idx:idx+1]
        dd = different_digit_batch[idx:idx+1]

        #Optionally, work with the same encoding for all the samples or recalculate the encoding every time
        combined_z_fixed,_,_,_,_,_,_ = model.forward(sd, dd)
        previous_combined_z = combined_z_fixed

        if test_determinism:
            # Check encoder determinism for this example (absolute and relative diffs)
            z1_n, z1_f, *_ = model.encode_only(sd, dd)
            z2_n, z2_f, *_ = model.encode_only(sd, dd)

            diff_num = z1_n - z2_n
            diff_flt = z1_f - z2_f

            max_diff_num = diff_num.abs().max().item()
            max_diff_flt = diff_flt.abs().max().item()

            l2_diff_num = diff_num.norm(p=2).item()
            l2_diff_flt = diff_flt.norm(p=2).item()

            base_num = max(z1_n.norm(p=2).item(), 1e-8)
            base_flt = max(z1_f.norm(p=2).item(), 1e-8)

            rel_l2_num = l2_diff_num / base_num
            rel_l2_flt = l2_diff_flt / base_flt

            print(
                f"Example {idx}: max|Δ number_z|={max_diff_num:.6f}, max|Δ filter_z|={max_diff_flt:.6f}, "
                f"relL2(number)={rel_l2_num:.6e}, relL2(filter)={rel_l2_flt:.6e}"
            )

        if not fixed_encoding:
        # Generating n samples
            samples=model.sample(sd,dd,num_samples=n_samples)
        else:
            combined_z, *_ = model.forward(sd, dd)  # [1, ...] latent(s)
            samples_list = []
            for _ in range(n_samples):
                s = model.decoder.sample(combined_z, 1)  # torch.Tensor [1,1,H,W] (adjust if different)
                s = s.view(1, 1, model.image_size, model.image_size)
                samples_list.append(s)
            samples = torch.cat(samples_list, dim=0)  # torch.Tensor [N,1,1,H,W]

        print('Generated samples with shape', samples.shape)

        # Look at the mean of the per-pixel standard deviation of the samples
        # Shape is (11,1,1,28,28)
        mean_std=samples.std(dim=0).mean()
        print(f'Mean of the (per pixel) standard deviation of the samples: {mean_std}')
        mean_std_list.append(mean_std.item())


        if plot_samples:
            #Plotting the samples
            total = 1 + n_samples
            rows, cols = math.ceil(total / 4), 4
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            axes = np.atleast_2d(axes)
            fig.suptitle(f"Mean per-pixel std of samples: {mean_std.item():.5f}", fontsize=12)




            # Plot GT
            gt_img = to_visualization_range(gt.detach().cpu().numpy()).squeeze()
            axes[0, 0].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
            axes[0, 0].set_title('GT')
            axes[0, 0].axis('off')

            # Plot samples
            for i in range(n_samples):
                r = (i + 1) // cols
                c = (i + 1) % cols
                samp = to_visualization_range(samples[i].detach().cpu().numpy()).squeeze()
                axes[r, c].imshow(samp, cmap='gray', vmin=0, vmax=1)
                axes[r, c].set_title(f'sample {i+1}')
                axes[r, c].axis('off')

            # # Hide any unused axes
            # for k in range(total, rows * cols):
            #     r, c = divmod(k, cols)
            #     axes[r, c].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()



        if plot_umap:
            if umap is None:
                print("UMAP not available. Please install 'umap-learn'. Skipping UMAP plot.")
            else:
                # Flatten samples: (N,1,1,28,28) -> (N, 28*28)
                flat = samples.detach().cpu().reshape(n_samples, -1).numpy()
                reducer = umap.UMAP(n_components=2)
                emb2d = reducer.fit_transform(flat)

                plt.figure(figsize=(6, 5))
                plt.scatter(emb2d[:, 0], emb2d[:, 1], s=10, color='black', alpha=0.8)
                plt.title(f"UMAP of {n_samples} samples (example {idx})")
                plt.xlabel('UMAP-1')
                plt.ylabel('UMAP-2')
                plt.grid(True)
                plt.tight_layout()

                # Save to figures/umap_generatedsamples/
                save_dir = os.path.join('figures', 'umap_generatedsamples')
                os.makedirs(save_dir, exist_ok=True)
                fname = f"umap_example_{idx:03d}_n{n_samples}.png"
                plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
                # plt.show()

    print(f'Mean of the mean of the per-pixel standard deviation of the samples: {np.mean(mean_std_list)}')
    print(f'Std of the mean of the per-pixel standard deviation of the samples: {np.std(mean_std_list)}')

    # Save the mean_std_list to a human-readable text file (append mode)
    with open('mean_std_list.txt', 'a') as f:
        f.write(f"# run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("values: " + ", ".join(f"{v:.5f}" for v in mean_std_list) + "\n")
        f.write(f"mean={np.mean(mean_std_list):.5f}, std={np.std(mean_std_list):.5f}\n\n")


        # for _ in range(n_samples):
        #     if fixed_encoding
        #         assert 'TODO: Implement fixed encoding'
        #         combined_z=combined_z_fixed
        #     else:
