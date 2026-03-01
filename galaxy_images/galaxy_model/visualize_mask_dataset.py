"""
Quick visualization script for HSCLegacyTripletDatasetMask
Shows the first 2 examples from the dataset.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from data import HSCLegacyTripletDatasetMask

# Load dataset
dataset = HSCLegacyTripletDatasetMask(
    hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_laptop.h5',
    idx_list=list(range(10)),  # Just load first 10 for testing
    is96=True,
)

# Get first 2 examples
example1 = dataset[0]
example2 = dataset[1]

def denormalize_image(img, norm_key, norm_dict):
    """Denormalize image for visualization"""
    mean, std = norm_dict[norm_key]
    return img * std + mean

def visualize_example(example, idx, norm_dict, ax_row):
    """Visualize a single example across a row of axes"""
    anchor_image, same_galaxy, same_instrument, metadata = example

    # Denormalize images for visualization
    if metadata['anchor_survey'] == 'hsc':
        anchor_denorm = denormalize_image(anchor_image, 'hsc96', norm_dict)
        same_galaxy_denorm = denormalize_image(same_galaxy, 'legacy96_zoom', norm_dict)
        same_instrument_denorm = denormalize_image(same_instrument, 'hsc96', norm_dict)
    else:  # legacy
        anchor_denorm = denormalize_image(anchor_image, 'legacy96_zoom', norm_dict)
        same_galaxy_denorm = denormalize_image(same_galaxy, 'hsc96', norm_dict)
        same_instrument_denorm = denormalize_image(same_instrument, 'legacy96_zoom', norm_dict)

    # Convert to RGB (first 3 channels) and normalize to [0, 1] for display
    def to_rgb(tensor):
        rgb = tensor[:3].clone()
        rgb = rgb - rgb.min()
        if rgb.max() > 0:
            rgb = rgb / rgb.max()
        return rgb.permute(1, 2, 0).cpu().numpy()

    # Plot anchor image
    ax_row[0].imshow(to_rgb(anchor_denorm))
    ax_row[0].set_title(f"Anchor ({metadata['anchor_survey']})")
    ax_row[0].axis('off')

    # Plot same galaxy
    ax_row[1].imshow(to_rgb(same_galaxy_denorm))
    ax_row[1].set_title("Same Galaxy\n(other instrument)")
    ax_row[1].axis('off')

    # Plot same instrument (masked)
    ax_row[2].imshow(to_rgb(same_instrument_denorm))
    ax_row[2].set_title("Same Instrument\n(center masked)")
    ax_row[2].axis('off')

    # Add text with metadata
    info_text = f"Idx: {metadata['idx']}\nAnchor: {metadata['anchor_survey']}"
    ax_row[3].text(0.5, 0.5, info_text, ha='center', va='center',
                   fontsize=10, transform=ax_row[3].transAxes)
    ax_row[3].axis('off')

# Create figure
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Import norm_dict
from data import NORM_DICT

# Visualize examples
visualize_example(example1, 0, NORM_DICT, axes[0])
visualize_example(example2, 1, NORM_DICT, axes[1])

# Add column labels
col_labels = ['Anchor Image', 'Same Galaxy', 'Same Instrument\n(Masked)', 'Metadata']
for col_idx, label in enumerate(col_labels):
    axes[0, col_idx].text(0.5, 1.15, label, ha='center', va='bottom',
                         transform=axes[0, col_idx].transAxes, fontsize=12, weight='bold')

plt.suptitle('HSCLegacyTripletDatasetMask - First 2 Examples', fontsize=14, y=0.98)
plt.tight_layout()

# Save figure
output_path = Path('/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures')
output_path.mkdir(parents=True, exist_ok=True)
save_path = output_path / 'mask_dataset_visualization.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {save_path}")

# plt.show()
