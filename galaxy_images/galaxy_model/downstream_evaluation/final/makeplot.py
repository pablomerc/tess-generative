"""
Read predict_all.py CSV and plot R² bar chart (HSC+Legacy case).

Three groups of targets:
  1) Physics
  2) Instrument (legacy)
  3) Instrument (HSC)

For each target, 4 bars: Physics latents, Instrument latents, Untrained, Mean.
(Physics/Instrument bars are filled only where we have that head; otherwise N/A.)

Also saves a table CSV with all targets + two extra rows for legacy-only case
(Physics (legacy only), Instrument (legacy only)) for future use.

Run from galaxy_model/ or downstream_evaluation/:
  python downstream_evaluation/final/makeplot.py --suffix zdim16_nogeom_neighbors
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent

# Task -> which bar to fill for "real" R²
PHYSICS_TASKS = {"physics_mmu", "physics_provabgs"}
INSTRUMENT_TASKS = {"instrument_mmu", "instrument_neighbors_legacy", "instrument_neighbors_hsc"}

# Group labels and which tasks belong to each (for ordering in the plot)
GROUP_PHYSICS = "physics"
GROUP_INSTRUMENT_LEGACY = "instrument (legacy)"
GROUP_INSTRUMENT_HSC = "instrument (hsc)"

TASK_TO_GROUP = {
    "physics_mmu": GROUP_PHYSICS,
    "physics_provabgs": GROUP_PHYSICS,
    "instrument_mmu": GROUP_INSTRUMENT_LEGACY,
    "instrument_neighbors_legacy": GROUP_INSTRUMENT_LEGACY,
    "instrument_neighbors_hsc": GROUP_INSTRUMENT_HSC,
}

BAR_LABELS = ["Physics latents", "Instrument latents", "Untrained", "Random"]
BAR_KEYS = ["r2_physics", "r2_instrument", "r2_untrained", "r2_random"]
COLORS = ["#2E86AB", "#A23B72", "#5DA271", "#E07A5F"]
N_A_VAL = np.nan  # show as gap or zero; we'll use 0 in bar with alpha for "N/A"


def load_and_augment(csv_path):
    """Load predict_all CSV and add group. Expects columns r2_physics, r2_instrument, r2_untrained, r2_mean."""
    df = pd.read_csv(csv_path)
    df["group"] = df["task"].map(TASK_TO_GROUP)
    # New CSV has r2_physics, r2_instrument per row; fallback for old CSV with r2_real only
    if "r2_physics" not in df.columns and "r2_real" in df.columns:
        df["r2_physics"] = np.where(df["task"].isin(PHYSICS_TASKS), df["r2_real"], np.nan)
        df["r2_instrument"] = np.where(df["task"].isin(INSTRUMENT_TASKS), df["r2_real"], np.nan)
    return df


def build_plot_order(df):
    """Return list of (group_label, list of target names) in desired order for the plot."""
    order = []
    for group in [GROUP_PHYSICS, GROUP_INSTRUMENT_LEGACY, GROUP_INSTRUMENT_HSC]:
        subset = df[df["group"] == group]
        targets = subset["target"].tolist()
        if targets:
            order.append((group, targets))
    return order


def make_bar_chart(df, plot_order, out_path, suffix):
    """One subplot per group; within each, one row per target, 4 bars per target."""
    n_groups = len(plot_order)
    if n_groups == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 4 * n_groups), sharex=False)
    if n_groups == 1:
        axes = [axes]

    for ax, (group_label, targets) in zip(axes, plot_order):
        subset = df[(df["group"] == group_label) & (df["target"].isin(targets))]
        subset = subset.set_index("target").loc[targets].reset_index()  # keep order

        n_t = len(targets)
        x = np.arange(n_t)
        width = 0.2
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

        for i, (key, label) in enumerate(zip(BAR_KEYS, BAR_LABELS)):
            vals = subset[key].values.copy()
            has_any = np.any(np.isfinite(vals))
            vals = np.where(np.isfinite(vals), vals, 0.0)
            # Always draw all bars; use lower alpha for N/A (no data in this group)
            alpha = 0.9 if has_any else 0.25
            ax.bar(x + offsets[i], vals, width, label=label, color=COLORS[i], alpha=alpha, edgecolor="gray", linewidth=0.5)

        ax.set_ylabel("R²")
        ax.set_title(group_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.5)

    fig.suptitle(f"Downstream R² — HSC+Legacy embeddings ({suffix})", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def save_table(df, plot_order, out_path, suffix):
    """Save CSV table: one row per target with 4 R² columns; then two rows for legacy-only."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for group_label, targets in plot_order:
        subset = df[(df["group"] == group_label) & (df["target"].isin(targets))]
        subset = subset.set_index("target").loc[targets].reset_index()
        for _, r in subset.iterrows():
            rows.append({
                "group": group_label,
                "target": r["target"],
                "Physics latents": r["r2_physics"] if pd.notna(r["r2_physics"]) else "",
                "Instrument latents": r["r2_instrument"] if pd.notna(r["r2_instrument"]) else "",
                "Untrained": r["r2_untrained"] if pd.notna(r["r2_untrained"]) else "",
                "Mean": r["r2_mean"] if pd.notna(r["r2_mean"]) else "",
            })
    # Two extra rows for legacy-only case (placeholder until you have those runs)
    rows.append({
        "group": "legacy only",
        "target": "Physics (legacy only)",
        "Physics latents": "",
        "Instrument latents": "",
        "Untrained": "",
        "Mean": "",
    })
    rows.append({
        "group": "legacy only",
        "target": "Instrument (legacy only)",
        "Physics latents": "",
        "Instrument latents": "",
        "Untrained": "",
        "Mean": "",
    })

    fieldnames = ["group", "target", "Physics latents", "Instrument latents", "Untrained", "Mean"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Table saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot predict_all CSV (HSC+Legacy case)")
    p.add_argument("--suffix", default="zdim16_nogeom_neighbors", help="Suffix used in predict_all output")
    p.add_argument("--output-dir", type=Path, default=_here, help="Directory with CSV and where to write plot + table")
    args = p.parse_args()

    csv_path = args.output_dir / f"predict_all_{args.suffix}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run predict_all first. Missing: {csv_path}")

    df = load_and_augment(csv_path)
    plot_order = build_plot_order(df)
    if not plot_order:
        print("No groups found in CSV.")
        return

    plot_path = args.output_dir / f"predict_all_{args.suffix}_plot.png"
    make_bar_chart(df, plot_order, plot_path, args.suffix)

    table_path = args.output_dir / f"predict_all_{args.suffix}_table.csv"
    save_table(df, plot_order, table_path, args.suffix)


if __name__ == "__main__":
    main()
