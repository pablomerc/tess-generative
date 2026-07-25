#!/usr/bin/env python3
"""Figures for the shuffle-conditioning ablation.

Produces:
  - qualitative grid: own | C0 | C1 | C2 | donor  (per survey)
  - σ-tracking scatter (σ_gen vs own/donor under C0 vs C2)
  - high-k / radial PSD overlay per condition
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from galaxy_images.galaxy_model.shuffle_ablation.metrics import mean_radial_psd

ABL_DIR = Path(__file__).resolve().parent


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _rgb_from_bands(img: np.ndarray, bands: Sequence[int] = (2, 1, 0)) -> np.ndarray:
    """(C,H,W) → (H,W,3) display RGB from i,r,g (or similar); percentile stretch."""
    chans = []
    for b in bands:
        x = img[b].astype(np.float64)
        lo, hi = np.percentile(x, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
        y = np.clip((x - lo) / (hi - lo), 0, 1)
        chans.append(y)
    return np.stack(chans, axis=-1)


def make_qualitative_grid(
    targets: np.ndarray,
    gens: Dict[str, np.ndarray],
    pi: np.ndarray,
    surveys: np.ndarray,
    *,
    out_path: Path,
    n_per_survey: int = 4,
    m: int = 0,
    seed: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        2 * n_per_survey,
        5,
        figsize=(10, 2.0 * 2 * n_per_survey),
        squeeze=False,
    )
    col_titles = ["own target", "C0 intact", "C1 shuffle-phy", "C2 shuffle-ins", "donor target"]

    row = 0
    for survey in ("hsc", "legacy"):
        idxs = np.where(surveys == survey)[0]
        pick = rng.choice(idxs, size=min(n_per_survey, len(idxs)), replace=False)
        for i in pick:
            panels = [
                targets[i],
                gens["C0"][i, m],
                gens["C1"][i, m],
                gens["C2"][i, m],
                targets[pi[i]],
            ]
            for col, img in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(_rgb_from_bands(img), origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{survey} #{i}", fontsize=8)
            row += 1

    # Hide unused rows if any.
    for r in range(row, axes.shape[0]):
        for c in range(axes.shape[1]):
            axes[r, c].axis("off")

    fig.suptitle("Shuffle ablation — shared noise, m=0", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out_path}")


def make_sigma_scatter(
    rows: List[Dict[str, str]],
    *,
    out_path: Path,
    sigma_key: str = "sigma_mad",
    m: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    # Collect per condition
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for cond in ("C0", "C2"):
        gens, owns, donors = [], [], []
        for r in rows:
            if r["condition"] != cond or int(r["m"]) != m:
                continue
            gens.append(float(r[sigma_key]))
            owns.append(float(r[f"{sigma_key}_own"]))
            donors.append(float(r[f"{sigma_key}_donor"]))
        data[cond] = {
            "gen": np.asarray(gens),
            "own": np.asarray(owns),
            "donor": np.asarray(donors),
        }

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), sharex=True, sharey=True)
    for ax, cond in zip(axes, ("C0", "C2")):
        d = data[cond]
        ax.scatter(d["own"], d["gen"], s=12, alpha=0.55, label="vs own", c="#1f4e79")
        ax.scatter(d["donor"], d["gen"], s=12, alpha=0.55, label="vs donor", c="#c45c26")
        # Identity line
        lo = float(min(d["own"].min(), d["donor"].min(), d["gen"].min()))
        hi = float(max(d["own"].max(), d["donor"].max(), d["gen"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        r_own = float(np.corrcoef(d["gen"], d["own"])[0, 1]) if len(d["gen"]) > 2 else float("nan")
        r_don = float(np.corrcoef(d["gen"], d["donor"])[0, 1]) if len(d["gen"]) > 2 else float("nan")
        ax.set_title(f"{cond}: r_own={r_own:.2f}, r_donor={r_don:.2f}")
        ax.set_xlabel(f"anchor {sigma_key}")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel(f"generated {sigma_key}")
    fig.suptitle(f"σ-tracking (m={m})", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out_path}")


def make_psd_overlay(
    targets: np.ndarray,
    gens: Dict[str, np.ndarray],
    *,
    out_path: Path,
    m: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    freqs_t, psd_t = mean_radial_psd(targets)
    ax.plot(freqs_t, psd_t, label="real targets", color="k", lw=1.5)
    colors = {"C0": "#1f4e79", "C1": "#2a9d8f", "C2": "#c45c26", "C3": "#6c757d"}
    for cond in ("C0", "C1", "C2", "C3"):
        if cond not in gens:
            continue
        freqs, psd = mean_radial_psd(gens[cond][:, m])
        ax.plot(freqs, psd, label=cond, color=colors.get(cond), lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("power")
    ax.set_title(f"Mean radial PSD (m={m})")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out_path}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=ABL_DIR / "results")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--n-per-survey", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma-key", type=str, default="sigma_adjdiff")
    args = p.parse_args(argv)

    out_dir = args.out_dir or (args.results_dir / "figures")
    csv_path = args.results_dir / "shuffle_per_anchor.csv"
    targets_path = args.results_dir / "targets.npz"

    rows = _read_csv(csv_path) if csv_path.exists() else []
    if rows:
        make_sigma_scatter(
            rows,
            out_path=out_dir / f"sigma_scatter_{args.sigma_key}.png",
            sigma_key=args.sigma_key,
            m=args.m,
        )

    if not targets_path.exists():
        print(f"[figures] {targets_path} missing — skipping grid/PSD (CSV scatter only)")
        return

    tdata = np.load(targets_path, allow_pickle=True)
    targets = tdata["x_1"]
    pi = tdata["pi"]
    surveys = tdata["anchor_surveys"]

    # gens arrays are indexed by position, not by absolute m — a split run may hold
    # e.g. m=4..7, where position 0 is m=4. Translate the requested m.
    m_pos = args.m
    if "m_values" in tdata.files:
        m_values = [int(v) for v in tdata["m_values"]]
        if args.m not in m_values:
            print(
                f"[figures] m={args.m} not in this run ({m_values}); "
                f"using m={m_values[0]} instead"
            )
            m_pos = 0
        else:
            m_pos = m_values.index(args.m)

    gens: Dict[str, np.ndarray] = {}
    for cond in ("C0", "C1", "C2", "C3"):
        path = args.results_dir / f"gens_{cond}.npz"
        if path.exists():
            gens[cond] = np.load(path)["gens"]

    if {"C0", "C1", "C2"}.issubset(gens):
        make_qualitative_grid(
            targets,
            gens,
            pi,
            surveys,
            out_path=out_dir / "qualitative_grid.png",
            n_per_survey=args.n_per_survey,
            m=m_pos,
            seed=args.seed,
        )
    if gens:
        make_psd_overlay(
            targets,
            gens,
            out_path=out_dir / "psd_overlay.png",
            m=m_pos,
        )


if __name__ == "__main__":
    main()
