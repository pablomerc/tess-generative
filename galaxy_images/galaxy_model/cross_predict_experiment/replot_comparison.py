"""Re-generate comparison bar charts with three bars:
  1. Ours (Instrument Latents from HSC+Legacy)
  2. Ours (Physics Latents from HSC+Legacy)
  3. Cross Predict ResNet
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
model_dir = script_dir.parent

BASELINE_CSV = model_dir / "downstream_evaluation" / "final" / "predict_all_zdim16_nogeom_neighbors_table.csv"

def make_plot(direction):
    r2_csv = script_dir / f"r2_results_{direction}.csv"
    df_cross = pd.read_csv(r2_csv)
    df_baseline = pd.read_csv(BASELINE_CSV)

    if direction == "hsc_to_legacy":
        group_filter = "instrument (legacy)"
        title = "HSC Image → Legacy Instrument Properties"
    else:
        group_filter = "instrument (hsc)"
        title = "Legacy Image → HSC Instrument Properties"

    baseline_instr = {}
    baseline_phys = {}
    for _, row in df_baseline.iterrows():
        if row["group"] == group_filter:
            baseline_instr[row["target"]] = row["Instrument latents"]
            baseline_phys[row["target"]] = row["Physics latents"]

    targets, r2_instr, r2_phys, r2_cross = [], [], [], []
    for _, row in df_cross.iterrows():
        name = row["target"]
        if name in baseline_instr:
            targets.append(name)
            r2_instr.append(baseline_instr[name])
            r2_phys.append(baseline_phys[name])
            r2_cross.append(row["r2_cross_predict"])

    n = len(targets)
    if n == 0:
        print(f"No matching targets for {direction}")
        return

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(11, n * 1.0), 6))

    ax.bar(x - width, r2_instr, width,
           label="Ours (Instrument Latents from HSC+Legacy)", color="#2E86AB")
    ax.bar(x, r2_phys, width,
           label="Ours (Physics Latents from HSC+Legacy)", color="#58B368")
    ax.bar(x + width, r2_cross, width,
           label=f"Cross Predict ResNet ({'HSC→Legacy' if direction == 'hsc_to_legacy' else 'Legacy→HSC'})",
           color="#CC546D")

    ax.set_ylabel(r"R$^2$ Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    short_labels = [
        t.replace("legacy_", "").replace("hsc_", "").replace("_value", "")
        for t in targets
    ]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()

    out_path = script_dir / f"comparison_{direction}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_plot("hsc_to_legacy")
    make_plot("legacy_to_hsc")
