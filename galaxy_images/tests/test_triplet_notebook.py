"""
Example notebook code for testing TripletCreator and visualizing triplets.

Copy this into a Jupyter notebook cell to test the triplet creation.
"""

# Cell 1: Import and setup
import matplotlib.pyplot as plt
import numpy as np
import torch
from galaxy_images.galaxy_triplets import TripletCreator

# Cell 2: Create TripletCreator (using just 1 file for quick testing)
creator = TripletCreator(
    dataset_path="/mnt/scratch/legacysurvey_hsc_crossmatched/data",
    files_to_use=1,  # Use just 1 parquet file for quick testing
    seed=42
)

print(f"Dataset loaded: {creator.num_rows} rows")

# Cell 3: Create a single triplet
triplet = creator.create_triplet()

print("Triplet created!")
print(f"Ground truth tensor shape: {triplet['ground_truth']['tensor'].shape}")
print(f"Ground truth instrument: {triplet['ground_truth']['instrument_onehot']}")
print(f"Cross instrument: {triplet['cross_instrument']['instrument_onehot']}")

# Cell 4: Helper function to visualize RGB from tensor
def tensor_to_rgb(tensor, channels=[0, 1, 2], percentile_clip=99.5):
    """
    Extract RGB channels from tensor and normalize for visualization.

    Args:
        tensor: Tensor of shape (C, H, W)
        channels: Which channels to use for R, G, B (default: first 3 flux channels)
        percentile_clip: Percentile to clip at for better contrast

    Returns:
        RGB array of shape (H, W, 3) in range [0, 1]
    """
    # Extract the specified channels
    rgb = tensor[channels].cpu().numpy()  # (3, H, W)

    # Transpose to (H, W, 3)
    rgb = np.transpose(rgb, (1, 2, 0))

    # Clip outliers using percentile
    for i in range(3):
        p_low = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        p_high = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], p_low, p_high)

    # Normalize each channel to [0, 1]
    for i in range(3):
        ch = rgb[:, :, i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            rgb[:, :, i] = (ch - ch_min) / (ch_max - ch_min)
        else:
            rgb[:, :, i] = 0.0

    return rgb

# Cell 5: Visualize the triplet
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Ground truth (anchor image)
gt_tensor = triplet['ground_truth']['tensor']
gt_rgb = tensor_to_rgb(gt_tensor, channels=[0, 1, 2])  # First 3 channels are flux (g, r, i)
axes[0].imshow(gt_rgb)
axes[0].set_title('Ground Truth (Anchor)\nInstrument: ' +
                  ('Legacy Survey' if triplet['ground_truth']['instrument_onehot'][0] > 0.5
                   else 'HSC'))
axes[0].axis('off')

# Different galaxy (same instrument)
diff_tensor = triplet['different_galaxy']['tensor']
diff_rgb = tensor_to_rgb(diff_tensor, channels=[0, 1, 2])
axes[1].imshow(diff_rgb)
axes[1].set_title('Different Galaxy\n(Same Instrument)')
axes[1].axis('off')

# Cross instrument (same galaxy, different instrument)
cross_tensor = triplet['cross_instrument']['tensor']
cross_rgb = tensor_to_rgb(cross_tensor, channels=[0, 1, 2])
axes[2].imshow(cross_rgb)
axes[2].set_title('Cross Instrument\n(Same Galaxy)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Cell 6: Print some metadata
print("\nMetadata:")
print(f"Anchor index: {triplet['anchor_index']}")
print(f"Different galaxy index: {triplet['different_index']}")
print(f"\nGround truth PSF FWHM: {triplet['ground_truth']['psf_fwhm']}")
print(f"Ground truth scale: {triplet['ground_truth']['scale']}")
