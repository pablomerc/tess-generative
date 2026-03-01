import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# --- Configuration ---
DATASET_PATH = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data'
OUTPUT_DIR = 'triplet_figures'
NUM_EXAMPLES = 250
USE_OBJECT_MASK = False
ANCHOR_SURVEY = 'hsc'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Path Setup ---
# Add parent directory to path to find galaxy_images module
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from galaxy_images.galaxy_triplets import TripletCreator
except ImportError:
    # Fallback for relative import
    sys.path.append('..')
    from galaxy_images.galaxy_triplets import TripletCreator

# --- Helper Functions ---

def tensor_to_rgb(tensor, channels=[0, 1, 2], percentile_clip=99.5):
    """
    Extract RGB channels from tensor and normalize for visualization.
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

def get_instrument_name(instrument_onehot):
    # Assuming index 0 is Legacy, 1 is HSC (based on previous context)
    return 'Legacy Survey' if instrument_onehot[0] > 0.5 else 'HSC'

# --- Main Execution ---

def parse_args():
    p = argparse.ArgumentParser(description="Generate triplet figures.")
    p.add_argument("--num-examples", type=int, default=NUM_EXAMPLES, help="Number of triplets to generate")
    p.add_argument("--anchor-index", type=int, default=None, help="Fix HSC anchor to this dataset index (generates 1 triplet)")
    return p.parse_args()


def main():
    args = parse_args()
    num_examples = args.num_examples
    anchor_index = args.anchor_index
    if anchor_index is not None:
        num_examples = 1
        print(f"Using fixed anchor index: {anchor_index} (generating 1 triplet)")

    print(f"Initializing TripletCreator from: {DATASET_PATH}")

    # Initialize Creator (files_to_use=1 for quicker loading/testing as requested)
    creator = TripletCreator(
        dataset_path=DATASET_PATH,
        files_to_use=1,
        seed=None
    )

    print(f"Dataset loaded: {creator.num_rows} rows available.")
    print(f"Generating {num_examples} figures in '{OUTPUT_DIR}'...")

    for i in tqdm(range(num_examples), desc="Saving Figures"):
        try:
            # 1. Create Triplet (optionally fix anchor to a specific index)
            triplet = creator.create_triplet(
                idx=anchor_index,
                anchor_survey=ANCHOR_SURVEY,
                use_object_mask=USE_OBJECT_MASK
            )

            # 2. Setup Plot (1 Row, 3 Columns)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # --- Column 1: Ground Truth (Anchor) ---
            gt_tensor = triplet['ground_truth']['tensor']
            gt_instr = get_instrument_name(triplet['ground_truth']['instrument_onehot'])
            gt_rgb = tensor_to_rgb(gt_tensor)

            axes[0].imshow(gt_rgb)
            axes[0].set_title(f"Anchor (GT)\n{gt_instr}", fontsize=10)
            axes[0].axis('off')

            # --- Column 2: Different Galaxy ---
            diff_tensor = triplet['different_galaxy']['tensor']
            diff_instr = get_instrument_name(triplet['different_galaxy']['instrument_onehot'])
            diff_rgb = tensor_to_rgb(diff_tensor)

            axes[1].imshow(diff_rgb)
            axes[1].set_title(f"Different Galaxy\n{diff_instr}", fontsize=10)
            axes[1].axis('off')

            # --- Column 3: Cross Instrument ---
            cross_tensor = triplet['cross_instrument']['tensor']
            cross_instr = get_instrument_name(triplet['cross_instrument']['instrument_onehot'])
            cross_rgb = tensor_to_rgb(cross_tensor)

            axes[2].imshow(cross_rgb)
            axes[2].set_title(f"Cross Instrument (Same Galaxy)\n{cross_instr}", fontsize=10)
            axes[2].axis('off')

            # 3. Save and Clean up
            plt.suptitle(f'Triplet #{i} (Indices: {triplet["anchor_index"]} / {triplet["different_index"]})')
            plt.tight_layout()

            save_path = os.path.join(OUTPUT_DIR, f"triplet_{i:03d}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=100)

            # CRITICAL: Close the figure to free memory
            plt.close(fig)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            plt.close('all') # Safety cleanup

    print(f"\nDone! {num_examples} figures saved to {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
