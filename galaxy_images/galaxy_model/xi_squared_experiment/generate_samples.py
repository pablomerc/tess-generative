"""
Generate posterior samples from the flow matching model and save alongside targets.

For each of the first N galaxies in a precomputed shard, this script generates
M independent posterior samples using Euler integration of the flow matching ODE.

TORCH.COMPILE STRATEGY
----------------------
`model.sample()` drives a Python for-loop over integration timesteps (Euler method).
Python loops cannot be compiled away by torch.compile, but each iteration calls
`model.forward()`, which in turn calls `model.velocity_model` (a UNet).  We compile
*velocity_model* — a pure nn.Module with no Python control flow — so every
UNet call within the loop uses the cached compiled graph.

The very first forward pass after compile() triggers graph tracing + kernel
compilation (expect ~30-60 s on V100).  All subsequent passes are fast.

SAMPLE BATCHING
---------------
To generate M samples for galaxy g, we tile the conditioning tensors M times
along the batch dimension and make a single model.sample() call.  This is more
GPU-efficient than M separate calls because the encoder computations are
amortized; each of the 50 Euler steps processes a batch of M noise images in
parallel rather than sequentially.

Usage
-----
    python generate_samples.py \\
        --checkpoint <ckpt.ckpt> \\
        --shard     <shard.h5> \\
        --n-galaxies 100 \\
        --m-samples  20 \\
        --output    samples.h5 \\
        --steps      50
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Allow importing from galaxy_model/ regardless of working directory
GALAXY_MODEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GALAXY_MODEL_DIR))

from double_train_fm_neighbors import ConditionalFlowMatchingModule  # noqa: E402


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> ConditionalFlowMatchingModule:
    print(f"Loading checkpoint: {checkpoint_path}")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)
    return model


def apply_compile(model: ConditionalFlowMatchingModule) -> None:
    """
    Replace model.velocity_model with its torch.compile'd counterpart.

    We target velocity_model (the UNet) rather than model.forward because:
    - velocity_model is a pure nn.Module with no Python branching → ideal compile target.
    - model.forward() calls encoders + velocity_model; compilation of velocity_model
      captures the expensive UNet computation.
    - model.sample() loops in Python, so compiling sample() directly would not help;
      compiling the body (velocity_model) is where the speedup lives.
    """
    print("Applying torch.compile to model.velocity_model …")
    model.velocity_model = torch.compile(model.velocity_model)
    print("  Graph capture will happen on the first forward pass.")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_shard(shard_path: str):
    """
    Load a precomputed NeighborsPrecomputedDataset shard entirely into RAM.

    HDF5 layout expected:
        targets          (N, 4, 48, 48)  – anchor galaxy (HSC or Legacy)
        samegals         (N, 4, 48, 48)  – same galaxy, opposite survey
        sameins          (N, k, 4, 48, 48) – same-instrument neighbors
        neighbor_masks   (N, k)          – True = real neighbor, False = padding
        meta_survey      (N,)            – b'hsc' or b'legacy' (anchor survey)
    """
    print(f"Loading shard into RAM: {shard_path}")
    t0 = time.time()
    with h5py.File(shard_path, "r") as f:
        targets  = torch.from_numpy(f["targets"][:])         # (N, 4, 48, 48)
        samegals = torch.from_numpy(f["samegals"][:])        # (N, 4, 48, 48)
        sameins  = torch.from_numpy(f["sameins"][:])         # (N, k, 4, 48, 48)
        masks    = torch.from_numpy(f["neighbor_masks"][:])  # (N, k)
        surveys  = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["meta_survey"][:]
        ]
    print(f"  Loaded {len(targets)} examples in {time.time() - t0:.1f}s")
    return targets, samegals, sameins, masks, surveys


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def generate_samples(
    model: ConditionalFlowMatchingModule,
    targets: torch.Tensor,
    samegals: torch.Tensor,
    sameins: torch.Tensor,
    masks: torch.Tensor,
    surveys: list,
    n_galaxies: int,
    m_samples: int,
    num_steps: int,
    device: torch.device,
):
    """
    Generate m_samples posterior samples for each of the first n_galaxies galaxies.

    Returns
    -------
    all_samples  : np.ndarray  (n_galaxies, m_samples, 4, 48, 48)
    all_targets  : np.ndarray  (n_galaxies, 4, 48, 48)
    all_surveys  : list[str]   length n_galaxies, each 'hsc' or 'legacy'
    all_samegals : np.ndarray  (n_galaxies, 4, 48, 48)  saved for visualisation
    """
    n_galaxies = min(n_galaxies, len(targets))
    all_samples = np.zeros((n_galaxies, m_samples, 4, 48, 48), dtype=np.float32)

    print(
        f"\nGenerating {m_samples} samples × {n_galaxies} galaxies "
        f"using {num_steps} Euler steps …"
    )
    t_total = time.time()
    first_galaxy = True

    for g in range(n_galaxies):
        t0 = time.time()

        # Tile conditioning M times: (1, …) → (M, …)
        # Each of the M noise draws is independent (fresh randn inside model.sample),
        # giving M independent posterior samples for the same conditioning.
        cond_samegal = samegals[g].unsqueeze(0).expand(m_samples, -1, -1, -1).to(device)
        cond_sameins = sameins[g].unsqueeze(0).expand(m_samples, -1, -1, -1, -1).to(device)
        cond_masks   = masks[g].unsqueeze(0).expand(m_samples, -1).to(device)

        if first_galaxy:
            print(f"  [Galaxy 0] First pass — torch.compile graph capture may take ~30-60 s …")

        samples_g = model.sample(
            cond_samegal,
            cond_sameins,
            cond_masks,
            num_steps=num_steps,
        )  # → (M, 4, 48, 48)

        elapsed = time.time() - t0
        if first_galaxy:
            print(f"  [Galaxy 0] Done in {elapsed:.1f}s (includes graph capture)")
            first_galaxy = False
        else:
            remaining = elapsed * (n_galaxies - g - 1)
            print(f"  Galaxy {g:3d}/{n_galaxies}  {elapsed:.1f}s/gal  ETA {remaining:.0f}s")

        all_samples[g] = samples_g.cpu().numpy()

    print(f"\nTotal generation time: {time.time() - t_total:.1f}s")
    return (
        all_samples,
        targets[:n_galaxies].numpy(),
        surveys[:n_galaxies],
        samegals[:n_galaxies].numpy(),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_output(output_path: str, samples, targets, surveys, samegals, args):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("samples",  data=samples,  compression="gzip", compression_opts=4)
        f.create_dataset("targets",  data=targets,  compression="gzip", compression_opts=4)
        f.create_dataset("samegals", data=samegals, compression="gzip", compression_opts=4)

        # Variable-length strings for survey labels
        dt = h5py.string_dtype()
        ds = f.create_dataset("meta_survey", (len(surveys),), dtype=dt)
        for i, s in enumerate(surveys):
            ds[i] = s

        # Embed config for reproducibility
        f.attrs["checkpoint"] = args.checkpoint
        f.attrs["shard"]      = args.shard
        f.attrs["n_galaxies"] = len(targets)
        f.attrs["m_samples"]  = int(samples.shape[1])
        f.attrs["num_steps"]  = args.steps
        f.attrs["compiled"]   = not args.no_compile

    print("  Saved.")
    print(f"  samples  : {samples.shape}")
    print(f"  targets  : {targets.shape}")
    print(f"  surveys  : {len(surveys)} entries ({surveys.count('hsc')} hsc, "
          f"{surveys.count('legacy')} legacy)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate flow matching posterior samples for Xi-squared calibration test"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--shard", required=True,
                        help="Path to precomputed HDF5 shard")
    parser.add_argument("--n-galaxies", type=int, default=100,
                        help="Number of galaxies to process (default: 100)")
    parser.add_argument("--m-samples", type=int, default=20,
                        help="Posterior samples per galaxy (default: 20)")
    parser.add_argument("--output", required=True,
                        help="Output HDF5 path")
    parser.add_argument("--steps", type=int, default=50,
                        help="Euler integration steps (default: 50; full quality: 250)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (for debugging or older PyTorch)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if args.steps < 250:
        print(f"NOTE: Using {args.steps} integration steps (not full 250). "
              "Results are approximate; mention this in any reported figures.")

    model = load_model(args.checkpoint, device)

    if not args.no_compile:
        apply_compile(model)
    else:
        print("torch.compile disabled via --no-compile")

    targets, samegals, sameins, masks, surveys = load_shard(args.shard)

    samples, targets_out, surveys_out, samegals_out = generate_samples(
        model, targets, samegals, sameins, masks, surveys,
        n_galaxies=args.n_galaxies,
        m_samples=args.m_samples,
        num_steps=args.steps,
        device=device,
    )

    save_output(args.output, samples, targets_out, surveys_out, samegals_out, args)


if __name__ == "__main__":
    main()
