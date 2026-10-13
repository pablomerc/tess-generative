#!/usr/bin/env python3
"""Four-series downstream-R² comparison: Ours / Contrastive / JEPA hsc-vit / JEPA cf.

Per-target grouped bars in three panels (physics, HSC instrument, Legacy
instrument), with per-series family means annotated. Colors follow the validated
categorical order (fixed by entity, never cycled); series identity is also in the
legend and the mean annotations, so color is not the only channel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENG = Path(__file__).resolve().parents[1] / "downstream_evaluation/engaging/outputs/predictions"
OUT = Path(__file__).resolve().parent / "results"

SERIES = [  # (label, csv, hex) — fixed palette order by entity
    ("Ours (counterfactual-generative)", "predict_base__aion.csv", "#2a78d6"),
    ("Contrastive (spatial-conv1x1)", "predict_contrastive-spatial-conv1x1__aion.csv", "#eb6834"),
    ("JEPA HSC-ViT", "predict_jepa-hsc-vit__aion.csv", "#1baf7a"),
    ("JEPA CF", "predict_jepa-cf__aion.csv", "#eda100"),
]
FAMILIES = [
    ("physics_provabgs", "combined_e1", "Physics properties (physics latent, e1)"),
    ("instrument_hsc", "combined_e2", "HSC instrument properties (instrument latent, e2)"),
    ("instrument_legacy", "combined_e2", "Legacy instrument properties (instrument latent, e2)"),
]

SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


def main():
    data = {lbl: pd.read_csv(ENG / f) for lbl, f, _ in SERIES}

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor=SURFACE)
    for ax, (fam, var, title) in zip(axes, FAMILIES):
        # target order: by the first series' scores, descending
        ref = data[SERIES[0][0]]
        ref = ref[(ref.task_family == fam) & (ref.latent_variant == var) & (ref.score_metric == "r2")]
        targets = list(ref.sort_values("score", ascending=False)["target"])

        n_s, width = len(SERIES), 0.19
        x = np.arange(len(targets))
        means = []
        for si, (lbl, _f, color) in enumerate(SERIES):
            d = data[lbl]
            d = d[(d.task_family == fam) & (d.latent_variant == var) & (d.score_metric == "r2")]
            vals = d.set_index("target").reindex(targets)["score"].to_numpy()
            offs = (si - (n_s - 1) / 2) * (width + 0.015)  # 2px-equivalent gap between bars
            ax.bar(x + offs, np.clip(vals, 0, None), width=width, color=color,
                   edgecolor=SURFACE, linewidth=0.5, label=lbl if fam == FAMILIES[0][0] else None)
            means.append((lbl, color, np.nanmean(vals)))

        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("R²", color=INK2, fontsize=11)
        ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("provabgs_", "").replace("legacy_", "").replace("hsc_", "")
                            for t in targets], rotation=45, ha="right", fontsize=8.5, color=INK2)
        ax.tick_params(colors=MUTED, length=0)
        # family means — visible labels (relief rule for low-contrast slots)
        mean_txt = "family means:  " + "   ".join(f"{l.split(' (')[0]}: {m:.3f}" for l, _c, m in means)
        ax.text(0.995, 0.96, mean_txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=INK2)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=4,
                   frameon=False, fontsize=10.5)
    fig.suptitle("Downstream R² — n=5469 overlap, AION MLP probe (seed 0)",
                 color=INK, fontsize=15, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.972, "final checkpoints; single-encoder JEPA arms probe one shared space for both families",
             color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "r2_four_series.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    sys.exit(main())
