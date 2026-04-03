"""
Combined calibration plot for HSC and Legacy surveys.

Reads the lightweight .npz produced by extract_calibration_data.py and
creates a single figure with both surveys overlaid.

Usage
-----
    python makecalibrationplot.py \
        --data outputs_512g_250steps/calibration_data.npz \
        --output outputs_512g_250steps/calibration_combined.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
})
import numpy as np
from scipy import stats

SPLIT_STYLE = {
    "hsc":    {"color": "#2176AE", "label": "HSC"},
    "legacy": {"color": "#E05A1A", "label": "DECaLS Legacy"},
}

N_HIST_SUBSAMPLE = 300_000
N_QQ = 5_000


def unpack_stats(arr):
    return {
        "n_galaxies":  int(arr[0]),
        "n_zscores":   int(arr[1]),
        "mean":        arr[2],
        "std":         arr[3],
        "frac_masked": arr[4],
        "ks_stat":     arr[5],
        "ks_pvalue":   arr[6],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help=".npz from extract_calibration_data.py")
    parser.add_argument("--output", default=None, help="Output figure path (default: next to data)")
    args = parser.parse_args()

    data = np.load(args.data)
    num_steps = int(data["num_steps"][0])
    m_samples = int(data["m_samples"][0])

    splits = {}
    for name in ("hsc", "legacy"):
        key_z = f"z_{name}"
        key_s = f"stats_{name}"
        if key_z not in data:
            continue
        splits[name] = {
            "z": data[key_z],
            "stats": unpack_stats(data[key_s]),
        }

    if not splits:
        raise RuntimeError("No split data found in the .npz file")

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    total_galaxies = sum(d["stats"]["n_galaxies"] for d in splits.values())

    # ── Left panel: overlaid histograms ──────────────────────────────
    ax = axes[0]
    for name, d in splits.items():
        z = d["z"]
        z_disp = z if len(z) <= N_HIST_SUBSAMPLE else rng.choice(z, N_HIST_SUBSAMPLE, replace=False)
        st = d["stats"]
        lbl = (f"{SPLIT_STYLE[name]['label']}  "
               f"($\\mu$={st['mean']:.3f}, $\\sigma$={st['std']:.3f})")
        ax.hist(z_disp, bins=120, density=True, alpha=0.5,
                color=SPLIT_STYLE[name]["color"], label=lbl)

    x = np.linspace(-5, 5, 500)
    ax.plot(x, stats.norm.pdf(x), "k--", lw=1.8, label=r"$\mathcal{N}(0,\,1)$")
    ax.set_xlim(-5, 5)
    ax.set_xlabel("z-score")
    ax.set_ylabel("Density")
    ax.set_title("Pixel z-score distributions")
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── Right panel: overlaid Q-Q plots ──────────────────────────────
    ax = axes[1]
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, N_QQ))
    for name, d in splits.items():
        z = d["z"]
        st = d["stats"]
        z_qq = rng.choice(z, min(N_QQ, len(z)), replace=False)
        z_qq_sorted = np.sort(z_qq)
        lbl = (f"{SPLIT_STYLE[name]['label']} ({st['n_galaxies']} galaxies, "
               f"$\\mu$={st['mean']:.3f}, $\\sigma$={st['std']:.3f})")
        ax.scatter(theoretical[:len(z_qq_sorted)], z_qq_sorted,
                   s=2, alpha=0.4, color=SPLIT_STYLE[name]["color"],
                   label=lbl)

    lim = 4.5
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="Perfect calibration")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Theoretical $\mathcal{N}(0,\,1)$ quantiles")
    ax.set_ylabel("Empirical z-score quantiles")
    ax.set_title("Q-Q plot")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3, lw=0.6)

    fig.text(0.5, -0.02,
             f"{total_galaxies} galaxies, {m_samples} posterior samples each, "
             f"{num_steps}-step Euler integration",
             ha="center", fontsize=11, style="italic", color="gray")

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    if args.output:
        out = Path(args.output)
    else:
        out = Path(args.data).parent / "calibration_combined.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined calibration figure to {out}")


if __name__ == "__main__":
    main()
