"""
Xi-squared posterior calibration test.

THEORY
------
A well-calibrated generative model satisfies:

    If we draw M posterior samples { x̂¹, …, x̂ᴹ } conditioned on some input,
    the empirical predictive distribution should contain the true image x*
    as a typical draw.  Formally, the per-pixel z-scores

        z[g, c, h, w] = (x*[g,c,h,w] - μ[g,c,h,w]) / σ[g,c,h,w]

    should follow N(0, 1) if the posterior is perfectly calibrated.

    μ and σ are computed from the M samples for each (galaxy, channel, pixel).

    We pool z-scores across all galaxies, channels, and pixels into one large
    array and test whether it looks like N(0, 1).

    NOTE: This test is run with {num_steps} Euler integration steps rather than
    the full 250, to keep generation time tractable.  Fewer steps = slightly
    noisier samples; results are indicative rather than definitive at this step count.

OUTPUTS (per anchor split: 'hsc' and 'legacy')
----------------------------------------------
  calibration_report_{split}.txt   – text report (mean, std, KS test, masked fraction)
  calibration_{split}.png          – figure with histogram + Q-Q plot side by side

Usage
-----
    python calibration_test.py --samples samples.h5 --output-dir ./results
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(path: str):
    """
    Load output from generate_samples.py.

    Returns
    -------
    samples   : (N, M, 4, 48, 48)
    targets   : (N, 4, 48, 48)
    surveys   : list[str]  – 'hsc' or 'legacy' per galaxy
    meta      : dict  – checkpoint, num_steps, etc.
    """
    with h5py.File(path, "r") as f:
        samples = f["samples"][:]     # (N, M, 4, 48, 48)
        targets = f["targets"][:]     # (N, 4, 48, 48)
        surveys = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["meta_survey"][:]
        ]
        meta = dict(f.attrs)
    return samples, targets, surveys, meta


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------

def compute_zscores(samples: np.ndarray, targets: np.ndarray, eps: float = 1e-6):
    """
    Compute per-pixel z-scores.

    Parameters
    ----------
    samples : (N, M, 4, 48, 48)
    targets : (N, 4, 48, 48)
    eps     : pixels with σ < eps are masked (overconfident / constant posterior)

    Returns
    -------
    z_flat   : 1-D array of valid z-scores
    frac_masked : fraction of pixels masked due to near-zero σ
    """
    mu  = samples.mean(axis=1)   # (N, 4, 48, 48)
    sig = samples.std(axis=1)    # (N, 4, 48, 48)

    valid_mask = sig >= eps      # (N, 4, 48, 48)
    frac_masked = 1.0 - valid_mask.mean()

    z = np.where(valid_mask, (targets - mu) / np.where(valid_mask, sig, 1.0), np.nan)
    z_flat = z[valid_mask]       # 1-D
    return z_flat, float(frac_masked)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_tests(z_flat: np.ndarray):
    """
    Test whether z_flat ~ N(0,1).

    Returns a dict with:
        mean, std
        ks_stat, ks_pvalue  – Kolmogorov-Smirnov test vs N(0,1)
        n_samples
    """
    mean = float(z_flat.mean())
    std  = float(z_flat.std())
    n    = len(z_flat)

    # KS test against N(0,1).  For very large n this is sensitive to tiny
    # deviations; treat p-value as a calibration metric, not a binary pass/fail.
    # Sub-sample to ≤500k points so KS stays tractable.
    rng = np.random.default_rng(42)
    z_ks = z_flat if n <= 500_000 else rng.choice(z_flat, 500_000, replace=False)
    ks_stat, ks_pval = stats.kstest(z_ks, "norm")

    return {
        "n_samples": n,
        "mean": mean,
        "std": std,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pval),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(z_flat: np.ndarray, stats_dict: dict, split: str, num_steps: int, m_samples: int):
    """
    Two-panel figure:
      Left  – histogram of z-scores overlaid with N(0,1) PDF
      Right – Q-Q plot of z-scores vs N(0,1) quantiles
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Histogram ---
    ax = axes[0]
    # Subsample for display to keep the histogram fast
    rng = np.random.default_rng(0)
    z_disp = z_flat if len(z_flat) <= 200_000 else rng.choice(z_flat, 200_000, replace=False)
    ax.hist(z_disp, bins=100, density=True, alpha=0.6, color="steelblue", label="z-scores")
    x = np.linspace(-5, 5, 500)
    ax.plot(x, stats.norm.pdf(x), "r-", lw=2, label=r"$\mathcal{N}(0,1)$")
    ax.set_xlim(-5, 5)
    ax.set_xlabel("z-score")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Z-score distribution — anchor: {split}\n"
        f"mean={stats_dict['mean']:.3f}  std={stats_dict['std']:.3f}  "
        f"KS p={stats_dict['ks_pvalue']:.2e}"
    )
    ax.legend()

    # --- Q-Q plot ---
    ax = axes[1]
    # Use a fixed subsample for the Q-Q plot (expensive if n is huge)
    n_qq = min(5_000, len(z_flat))
    z_qq = rng.choice(z_flat, n_qq, replace=False)
    z_qq_sorted = np.sort(z_qq)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, n_qq))
    ax.scatter(theoretical, z_qq_sorted, s=1, alpha=0.3, color="steelblue")
    lim = max(abs(theoretical).max(), abs(z_qq_sorted).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r-", lw=2, label="y = x (perfect calibration)")
    ax.set_xlabel(r"Theoretical $\mathcal{N}(0,1)$ quantiles")
    ax.set_ylabel("Empirical z-score quantiles")
    ax.set_title(f"Q-Q plot — anchor: {split}")
    ax.legend(fontsize=8)

    if num_steps < 250:
        note = (
            f"NOTE: Generated with {num_steps} Euler steps (not full 250) and "
            f"{m_samples} samples/galaxy.\n"
            "Results are indicative; re-run with 250 steps for publication-quality calibration."
        )
    else:
        note = (
            f"Generated with {num_steps} Euler steps (publication quality) and "
            f"{m_samples} samples/galaxy."
        )
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, style="italic", color="gray")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(path: Path, split: str, stats_dict: dict, frac_masked: float,
                 n_galaxies: int, meta: dict):
    lines = [
        "=" * 60,
        f"Xi-squared Calibration Report — anchor survey: {split.upper()}",
        "=" * 60,
        "",
        "Generation config",
        f"  Checkpoint  : {meta.get('checkpoint', 'unknown')}",
        f"  Shard       : {meta.get('shard', 'unknown')}",
        f"  Euler steps : {meta.get('num_steps', '?')}  "
        f"(NOTE: full quality uses 250 steps)",
        f"  Samples/gal : {meta.get('m_samples', '?')}",
        f"  Galaxies    : {n_galaxies}",
        "",
        "Z-score statistics (ideal: mean=0, std=1)",
        f"  N z-scores  : {stats_dict['n_samples']:,}",
        f"  Mean        : {stats_dict['mean']:+.4f}  (ideal 0)",
        f"  Std         : {stats_dict['std']:.4f}  (ideal 1)",
        f"  Masked frac : {frac_masked:.4f}  (pixels with σ < 1e-6)",
        "",
        "Kolmogorov-Smirnov test vs N(0,1)",
        f"  KS statistic: {stats_dict['ks_stat']:.4f}",
        f"  KS p-value  : {stats_dict['ks_pvalue']:.4e}",
        "  Interpretation: p >> 0.05 → consistent with N(0,1); "
        "p ≈ 0 → significant mis-calibration",
        "  (For N > 500k, KS is sensitive to tiny deviations; "
        "use std and histogram as primary guides)",
        "",
        "=" * 60,
    ]
    path.write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Per-split analysis
# ---------------------------------------------------------------------------

def analyse_split(samples, targets, n_galaxies, split_name, output_dir, meta):
    """Run the full calibration analysis for one anchor-survey split."""
    print(f"\n{'='*50}")
    print(f"  Split: {split_name.upper()}  ({n_galaxies} galaxies)")
    print(f"{'='*50}")

    z_flat, frac_masked = compute_zscores(samples, targets)
    print(f"  Z-scores computed: {len(z_flat):,}  (masked fraction: {frac_masked:.4f})")

    st = run_tests(z_flat)

    num_steps = int(meta.get("num_steps", 50))
    m_samples = int(meta.get("m_samples", samples.shape[1]))

    fig = make_figure(z_flat, st, split_name, num_steps, m_samples)
    fig_path = output_dir / f"calibration_{split_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    report_path = output_dir / f"calibration_report_{split_name}.txt"
    write_report(report_path, split_name, st, frac_masked, n_galaxies, meta)
    print(f"  Report saved: {report_path}")

    # Also save raw stats as JSON for easy programmatic access
    json_path = output_dir / f"calibration_stats_{split_name}.json"
    payload = {"split": split_name, "n_galaxies": n_galaxies,
               "frac_masked": frac_masked, **st,
               "meta": {k: str(v) for k, v in meta.items()}}
    json_path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Xi-squared posterior calibration test (split by anchor survey)"
    )
    parser.add_argument("--samples", required=True,
                        help="HDF5 output from generate_samples.py")
    parser.add_argument("--output-dir", default="./calibration_results",
                        help="Directory for figures and reports (default: ./calibration_results)")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Mask pixels with σ < eps (default: 1e-6)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, targets, surveys, meta = load_samples(args.samples)
    print(f"Loaded: {samples.shape[0]} galaxies, {samples.shape[1]} samples each")
    print(f"Surveys: {surveys.count('hsc')} hsc, {surveys.count('legacy')} legacy")
    print(f"Euler steps used for generation: {meta.get('num_steps', '?')}")

    surveys_arr = np.array(surveys)

    # --- Run analysis for each anchor-survey split ---
    for split in ("hsc", "legacy"):
        idx = np.where(surveys_arr == split)[0]
        if len(idx) == 0:
            print(f"  No galaxies with anchor_survey='{split}', skipping.")
            continue
        analyse_split(
            samples[idx],
            targets[idx],
            n_galaxies=len(idx),
            split_name=split,
            output_dir=output_dir,
            meta=meta,
        )

    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
