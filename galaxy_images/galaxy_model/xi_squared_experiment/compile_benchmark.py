"""
torch.compile speedup benchmark for flow matching inference.

Measures the per-sample latency of model.sample() with and without
torch.compile, and separately logs the one-time compilation overhead.

HOW torch.compile WORKS (brief)
--------------------------------
torch.compile() traces the computation graph of a function/module on the
first call, then uses TorchInductor to emit optimised Triton/CUDA kernels.
After the first (slow) call:
  - kernel code is cached on disk
  - subsequent calls skip Python overhead and use fused kernels

We compile model.velocity_model (the UNet) because:
  1. It's a pure nn.Module — no Python branching — ideal for graph capture.
  2. model.sample() loops in Python over Euler steps; that loop cannot be
     compiled away, but each loop body calls velocity_model which CAN be.
  3. The UNet dominates runtime (~99%) so this is where speedup matters.

The benchmark uses a single galaxy's conditioning, repeated for each trial,
to eliminate data-loading noise.

Usage
-----
    python compile_benchmark.py \\
        --checkpoint <ckpt.ckpt> \\
        --shard      <shard.h5>  \\
        --n-trials   10          \\
        --steps      50
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

GALAXY_MODEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GALAXY_MODEL_DIR))

from double_train_fm_neighbors import ConditionalFlowMatchingModule  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
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


def load_one_galaxy(shard_path: str, galaxy_idx: int = 0):
    """Load a single galaxy's tensors from the shard (no need to load all into RAM)."""
    with h5py.File(shard_path, "r") as f:
        samegal = torch.from_numpy(f["samegals"][galaxy_idx])     # (4, 48, 48)
        sameins = torch.from_numpy(f["sameins"][galaxy_idx])      # (k, 4, 48, 48)
        mask    = torch.from_numpy(f["neighbor_masks"][galaxy_idx])  # (k,)
        survey  = f["meta_survey"][galaxy_idx]
        if isinstance(survey, bytes):
            survey = survey.decode()
    return samegal, sameins, mask, survey


def prep_batch(samegal, sameins, mask, device):
    """Prepare a batch of size 1 (one galaxy, one sample)."""
    return (
        samegal.unsqueeze(0).to(device),       # (1, 4, 48, 48)
        sameins.unsqueeze(0).to(device),       # (1, k, 4, 48, 48)
        mask.unsqueeze(0).to(device),          # (1, k)
    )


def time_sample(model, cond_samegal, cond_sameins, cond_masks, num_steps, n_trials, device):
    """
    Run model.sample() n_trials times and return list of elapsed times (seconds).
    Uses CUDA synchronisation so GPU timing is accurate.
    """
    times = []
    for _ in range(n_trials):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        model.sample(cond_samegal, cond_sameins, cond_masks, num_steps=num_steps)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    return times


def print_table(label, times):
    arr = np.array(times)
    print(f"  {label:<25}  mean={arr.mean():.3f}s  "
          f"std={arr.std():.3f}s  min={arr.min():.3f}s  max={arr.max():.3f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark torch.compile speedup for flow matching inference"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--n-warmup", type=int, default=2,
                        help="Warmup runs before timing (no-compile only)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Timed runs per condition (default: 10)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Euler integration steps (default: 50)")
    parser.add_argument("--galaxy-idx", type=int, default=0,
                        help="Which galaxy from the shard to use (default: 0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device       : {device}")
    print(f"Euler steps  : {args.steps}")
    print(f"Warmup runs  : {args.n_warmup}")
    print(f"Timed trials : {args.n_trials}")
    print()

    # ------------------------------------------------------------------
    # 1. Benchmark WITHOUT compile
    # ------------------------------------------------------------------
    print("=" * 55)
    print("PASS 1: No torch.compile")
    print("=" * 55)
    model_plain = load_model(args.checkpoint, device)
    samegal, sameins, mask, survey = load_one_galaxy(args.shard, args.galaxy_idx)
    cond_samegal, cond_sameins, cond_masks = prep_batch(samegal, sameins, mask, device)

    print(f"  Galaxy anchor survey: {survey}")
    print(f"  Warming up ({args.n_warmup} runs) …")
    _ = time_sample(model_plain, cond_samegal, cond_sameins, cond_masks,
                    args.steps, args.n_warmup, device)

    print(f"  Timing {args.n_trials} runs …")
    times_plain = time_sample(model_plain, cond_samegal, cond_sameins, cond_masks,
                              args.steps, args.n_trials, device)
    print_table("No compile", times_plain)
    del model_plain

    # ------------------------------------------------------------------
    # 2. Benchmark WITH compile
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("PASS 2: With torch.compile(model.velocity_model)")
    print("=" * 55)
    model_compiled = load_model(args.checkpoint, device)

    print("  Applying torch.compile …")
    model_compiled.velocity_model = torch.compile(model_compiled.velocity_model)

    # First call = graph capture + kernel compilation (the "compile cost")
    print("  First call — capturing graph (this is the one-time compile cost) …")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_compile_start = time.perf_counter()
    model_compiled.sample(cond_samegal, cond_sameins, cond_masks, num_steps=args.steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_compile_end = time.perf_counter()
    compile_time = t_compile_end - t_compile_start
    print(f"  Compilation + first sample time: {compile_time:.2f}s")

    print(f"  Timing {args.n_trials} post-compile runs …")
    times_compiled = time_sample(model_compiled, cond_samegal, cond_sameins, cond_masks,
                                 args.steps, args.n_trials, device)
    print_table("Compiled", times_compiled)
    del model_compiled

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    mean_plain    = float(np.mean(times_plain))
    mean_compiled = float(np.mean(times_compiled))
    speedup = mean_plain / mean_compiled if mean_compiled > 0 else float("inf")
    n_samples_to_break_even = compile_time / max(mean_plain - mean_compiled, 1e-9)

    print()
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  No-compile latency (mean)   : {mean_plain:.3f} s/sample")
    print(f"  Compiled latency  (mean)    : {mean_compiled:.3f} s/sample")
    print(f"  Speedup                     : {speedup:.2f}x")
    print(f"  One-time compile overhead   : {compile_time:.1f} s")
    print(f"  Break-even at              ~: {n_samples_to_break_even:.0f} samples")
    print(f"  (i.e. compile pays off after ~{n_samples_to_break_even:.0f} generated samples)")
    print()
    print(f"NOTE: Benchmark used {args.steps} Euler steps. "
          "Speedup ratio is expected to be similar at 250 steps.")


if __name__ == "__main__":
    main()
