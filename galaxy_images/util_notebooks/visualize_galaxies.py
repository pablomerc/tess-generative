import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import glob
from datasets import load_dataset
from astropy.visualization import make_lupton_rgb

# --- Configuration ---
dataset_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data'
pattern = os.path.join(dataset_path, "train-00000-of-00396.parquet")
all_files = sorted(glob.glob(pattern))

dataset = load_dataset(
    "parquet",
    data_files=all_files,
    split="train"
)




N_EXAMPLES = 250
output_dir = "plots_check"
os.makedirs(output_dir, exist_ok=True)


def center_crop(image, crop_size=96):
    _, height, width = image.shape
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    return image[
         :, start_y : start_y + crop_size, start_x : start_x + crop_size
    ]


def min_max_normalize(img):
    """Standard 0-1 normalization for simple visualization."""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def apply_training_row_scale(img_tensor, vmin, vmax):
    """
    Replicates the _row_scale_rgb logic from your training code.
    Scales tensor based on provided vmin/vmax (derived from target).
    """
    # img_tensor shape: (3, H, W)
    # vmin/vmax shape: (3,)

    # View for broadcasting (3, 1, 1)
    vmin_t = vmin.view(3, 1, 1)
    vmax_t = vmax.view(3, 1, 1)

    # Normalize
    y = (img_tensor - vmin_t) / (vmax_t - vmin_t + 1e-8)

    # Clamp to [0, 1] as done in the training loop
    y = y.clamp(0, 1)

    # Permute to (H, W, 3) for matplotlib
    return y.permute(1, 2, 0).numpy()

for i in range(N_EXAMPLES):
    # 1. Fetch record
    example_record = dataset[i]

    # 2. Extract Flux (Channels 1:4) -> Shape (3, H, W)
    # Keeping them as Tensors initially for the Training Logic math
    hsc_tensor = torch.from_numpy(np.array(example_record['hsc_image']['flux'])).float()[0:3]
    legacy_tensor = torch.from_numpy(np.array(example_record['legacysurvey_image']['flux'])).float()[0:3]
    hsc_tensor = center_crop(hsc_tensor)
    legacy_tensor = center_crop(legacy_tensor)
    # # ==========================================
    # # 1) SIMPLE MIN-MAX (Individual)
    # # ==========================================
    # hsc_simple = hsc_tensor.permute(1, 2, 0).numpy()
    # legacy_simple = legacy_tensor.permute(1, 2, 0).numpy()

    # fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
    # axs1[0].imshow(min_max_normalize(hsc_simple))
    # axs1[0].set_title(f"Ex {i}: HSC (Simple MinMax)")
    # axs1[0].axis('off')
    # axs1[1].imshow(min_max_normalize(legacy_simple))
    # axs1[1].set_title(f"Ex {i}: Legacy (Simple MinMax)")
    # axs1[1].axis('off')

    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f"comparison_{i}_simple.png"))
    # plt.close(fig1)

    # # ==========================================
    # # 2) ASTROPY (Lupton RGB)
    # # ==========================================
    # hsc_np = hsc_tensor.numpy()
    # legacy_np = legacy_tensor.numpy()

    # # Mapping channels 2->R, 1->G, 0->B (i, r, g)
    # try:
    #     hsc_astro = make_lupton_rgb(hsc_np[2], hsc_np[1], hsc_np[0], stretch=0.5, Q=10)
    #     legacy_astro = make_lupton_rgb(legacy_np[2], legacy_np[1], legacy_np[0], stretch=0.5, Q=10)
    # except ValueError:
    #     hsc_astro = np.zeros((hsc_np.shape[1], hsc_np.shape[2], 3))
    #     legacy_astro = np.zeros((legacy_np.shape[1], legacy_np.shape[2], 3))

    # fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    # axs2[0].imshow(hsc_astro)
    # axs2[0].set_title(f"Ex {i}: HSC (Astropy)")
    # axs2[0].axis('off')
    # axs2[1].imshow(legacy_astro)
    # axs2[1].set_title(f"Ex {i}: Legacy (Astropy)")
    # axs2[1].axis('off')

    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f"comparison_{i}_astropy.png"))
    # plt.close(fig2)

    # ==========================================
    # 3) TRAINING CODE LOGIC (Row-Scaled)
    # ==========================================
    # Logic extracted from on_validation_epoch_end in your provided code:
    # 1. Compute per-channel vmin/vmax from the TARGET (HSC)
    # 2. Use those exact bounds to scale the conditioning image (Legacy)

    # Calculate bounds from HSC (Target)
    # dim=(1,2) reduces H,W leaving (3,) channels
    vmin = hsc_tensor.amin(dim=(1, 2))
    vmax = hsc_tensor.amax(dim=(1, 2))

    # Apply to HSC
    hsc_train_vis = apply_training_row_scale(hsc_tensor, vmin, vmax)

    vmax_legacy = legacy_tensor.amax(dim=(1, 2))
    vmin_legacy = legacy_tensor.amin(dim=(1, 2))
    # Apply to Legacy using HSC's bounds (simulating how the model sees/conditions)
    legacy_train_vis = apply_training_row_scale(legacy_tensor, vmin_legacy, vmax_legacy)

    fig3, axs3 = plt.subplots(1, 2, figsize=(10, 5))

    axs3[0].imshow(hsc_train_vis)
    axs3[0].set_title(f"Ex {i}: HSC (Train Logic)")
    axs3[0].axis('off')

    axs3[1].imshow(legacy_train_vis)
    axs3[1].set_title(f"Ex {i}: Legacy (Train Logic)")
    axs3[1].axis('off')

    # Add a note about the scaling
    fig3.suptitle("Normalized using HSC min/max per channel (Row-Scaled)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_{i}_training_row.png"))
    plt.close(fig3)

    print(f"Saved all versions for example {i} to {output_dir}")
