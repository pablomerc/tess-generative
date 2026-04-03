"""
Extract z-score calibration data from the HDF5 samples file and save
to a lightweight .npz so the plotting script can iterate quickly without
re-reading the ~360 MB samples file.

Usage
-----
    python extract_calibration_data.py \
        --samples outputs_512g_250steps/full_samples_512g_250steps.h5 \
        --output  outputs_512g_250steps/calibration_data.npz
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import stats


def compute_zscores(samples: np.ndarray, targets: np.ndarray, eps: float = 1e-6):
    mu = samples.mean(axis=1)
    sig = samples.std(axis=1)
    valid = sig >= eps
    frac_masked = 1.0 - valid.mean()
    z = np.where(valid, (targets - mu) / np.where(valid, sig, 1.0), np.nan)
    return z[valid], float(frac_masked)


def ks_test(z_flat, max_n=500_000):
    rng = np.random.default_rng(42)
    z_sub = z_flat if len(z_flat) <= max_n else rng.choice(z_flat, max_n, replace=False)
    stat, pval = stats.kstest(z_sub, "norm")
    return float(stat), float(pval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    with h5py.File(args.samples, "r") as f:
        samples = f["samples"][:]
        targets = f["targets"][:]
        surveys = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["meta_survey"][:]
        ]
        meta = dict(f.attrs)

    surveys_arr = np.array(surveys)
    save_dict = {}

    for split in ("hsc", "legacy"):
        idx = np.where(surveys_arr == split)[0]
        if len(idx) == 0:
            continue
        z_flat, frac_masked = compute_zscores(samples[idx], targets[idx], args.eps)
        ks_stat, ks_pval = ks_test(z_flat)
        save_dict[f"z_{split}"] = z_flat.astype(np.float32)
        save_dict[f"stats_{split}"] = np.array([
            len(idx),           # n_galaxies
            len(z_flat),        # n_zscores
            z_flat.mean(),      # mean
            z_flat.std(),       # std
            frac_masked,        # frac_masked
            ks_stat,            # ks_stat
            ks_pval,            # ks_pvalue
        ])
        print(f"{split}: {len(idx)} galaxies, {len(z_flat):,} z-scores, "
              f"mean={z_flat.mean():.4f}, std={z_flat.std():.4f}")

    save_dict["num_steps"] = np.array([int(meta.get("num_steps", 0))])
    save_dict["m_samples"] = np.array([int(meta.get("m_samples", 0))])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **save_dict)
    print(f"\nSaved calibration data to {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
