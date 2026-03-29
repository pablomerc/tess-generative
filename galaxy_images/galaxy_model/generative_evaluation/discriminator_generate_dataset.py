"""Step 1: Generate a dataset of real vs generated image pairs.

Loads the flow-matching model, runs ODE integration on the neighbours
dataset, and saves paired (real, generated) numpy arrays as .npz files
— one per survey (HSC and Legacy).

Usage:
  python discriminator_generate_dataset.py --n-samples 1000
  python discriminator_generate_dataset.py --n-samples 10000

Output files (in disc_data_{tag}/):
  - disc_dataset_hsc.npz     with keys: real (N, 4, 48, 48), gen (N, 4, 48, 48)
  - disc_dataset_legacy.npz  with keys: real (N, 4, 48, 48), gen (N, 4, 48, 48)

These are consumed by discriminator_train.py (step 2).
"""

import sys
from pathlib import Path
import argparse

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import time
import torch
torch.backends.cuda.preferred_blas_library("hipblas")
import numpy as np
from torch.utils.data import DataLoader, Subset

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.neighbors import NeighborsDataset, collate_neighbors

# ======================== CONFIG ========================

MODEL_CHECKPOINT = '/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt'
HDF5_PATH = '/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5'

BATCH_SIZE = 16
NUM_INTEGRATION_STEPS = 250
IMAGE_SIZE = 48


def _make_tag(n):
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of samples per survey (default: 1000)')
    args = parser.parse_args()

    n_samples = args.n_samples
    tag = _make_tag(n_samples)
    data_dir = _script_dir / f'disc_data_{tag}'
    data_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"N_SAMPLES_PER_SURVEY: {n_samples}  (tag: {tag})")

    print("Loading flow-matching model …")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        MODEL_CHECKPOINT, map_location='cpu')
    model.eval()
    model.to(device)
    print("Model loaded.")

    print(f"Opening dataset from {HDF5_PATH} …")
    dataset = NeighborsDataset(hdf5_path=HDF5_PATH, crop_size=IMAGE_SIZE, max_neighbors=15)
    print(f"Dataset size: {len(dataset)}")

    n_use = min(len(dataset), n_samples * 3)
    subset = Subset(dataset, list(range(n_use)))
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=collate_neighbors)

    results = {'hsc': {'real': [], 'gen': []},
               'legacy': {'real': [], 'gen': []}}

    t0 = time.time()
    print(f"\nGenerating reconstructions ({NUM_INTEGRATION_STEPS} integration steps) …")
    print(f"Target: {n_samples} samples per survey")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            targets, samegals, neighbors, masks, metadata = batch
            targets = targets.to(device)
            samegals = samegals.to(device)
            neighbors = neighbors.to(device)
            masks = masks.to(device)

            reconstructions = model.sample(
                samegals, neighbors, masks=masks, num_steps=NUM_INTEGRATION_STEPS)

            for i in range(targets.shape[0]):
                survey = metadata[i]['anchor_survey']
                if len(results[survey]['real']) >= n_samples:
                    continue
                results[survey]['real'].append(targets[i].cpu().numpy())
                results[survey]['gen'].append(reconstructions[i].cpu().numpy())

            n_hsc = len(results['hsc']['real'])
            n_leg = len(results['legacy']['real'])

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                elapsed = time.time() - t0
                print(f"  Batch {batch_idx + 1:5d}  |  HSC: {n_hsc:5d}  Legacy: {n_leg:5d}  |  {elapsed:.0f}s")

            if n_hsc >= n_samples and n_leg >= n_samples:
                break

    elapsed = time.time() - t0
    print(f"\nGeneration finished in {elapsed:.1f}s")

    for survey in ['hsc', 'legacy']:
        real = np.stack(results[survey]['real'][:n_samples])
        gen = np.stack(results[survey]['gen'][:n_samples])
        out_path = data_dir / f'disc_dataset_{survey}.npz'
        np.savez_compressed(out_path, real=real, gen=gen)
        print(f"Saved {out_path.name}: real {real.shape}, gen {gen.shape}  "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\nAll datasets saved to {data_dir}")


if __name__ == '__main__':
    main()
