#!/usr/bin/env python3
"""Coadd-depth exploration for the two LSST DP2 fields in our dataset.

Reads only lsst_meta.parquet (no image I/O — login-node friendly) and makes:
  figures/nvisits_maps.png
    row 1: RA-Dec heatmaps of median n_input_i per field (independent color
           scales — the fields differ by ~10x)
    row 2: n_input_i histograms per field + per-band median bar chart
Prints a per-field, per-band summary table (mean/median n_input).

n_input_<band> = number of individual exposures stacked into the deep coadd
at that object's position in that band — the per-object "depth" of the pixel
data. See DATASET_NOTES.md next to this script for interpretation.

  PYTHONPATH=. python galaxy_images/galaxy_model/lsst/data_exploration/nvisits_maps.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

STORE = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = ["u", "g", "r", "i", "z", "y"]
FIELDS = ("COSMOS", "DEEP2-3")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = pd.read_parquet(os.path.join(STORE, "lsst_meta.parquet"))

    print("median (mean) n_input per band:")
    rows = {}
    for f in FIELDS:
        s = meta[meta.field == f]
        rows[f] = {}
        line = [f"{f:8s} n={len(s):5d} "]
        for b in BANDS:
            v = s[f"n_input_{b}"].dropna()
            rows[f][b] = (float(v.median()) if len(v) else np.nan,
                          float(v.mean()) if len(v) else np.nan)
            line.append(f"{b}:{rows[f][b][0]:.0f} ({rows[f][b][1]:.1f})"
                        if len(v) else f"{b}: --")
        print("  " + "  ".join(line))

    fig = plt.figure(figsize=(14, 13.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.3, 1])

    # ---- all-sky context panel (Mollweide, astro convention: RA left) ----
    def to_xy(ra_deg, dec_deg):
        lon = np.remainder(np.atleast_1d(np.asarray(ra_deg, dtype=float)) + 360,
                           360)
        lon[lon > 180] -= 360
        return np.radians(-lon), np.radians(np.atleast_1d(dec_deg))  # RA left

    ax = fig.add_subplot(gs[0, :], projection="mollweide")
    cache = os.path.join(HERE, "figures", "dp2_coverage_radec.npz")
    if os.path.exists(cache):                      # full DP2 footprint context
        cov = np.load(cache)
        x, y = to_xy(cov["ra"], cov["dec"])
        ax.scatter(x, y, s=0.3, c="#bdd7e7", lw=0,
                   label="DP2 deep-coadd coverage (~3000 deg²)")
    ecl_ra = np.linspace(0, 360, 400)              # ecliptic, for orientation
    ecl_dec = 23.44 * np.sin(np.radians(ecl_ra))
    xe, ye = to_xy(ecl_ra, ecl_dec)
    o = np.argsort(xe)
    ax.plot(xe[o], ye[o], "k--", lw=0.8, alpha=0.5, label="ecliptic")
    for f, color in zip(FIELDS, ("#1f77b4", "#d62728")):
        s = meta[meta.field == f]
        x, y = to_xy(s.ra.mean(), s.dec.mean())
        x0, y0 = float(x[0]), float(y[0])
        ax.scatter([x0], [y0], s=140, marker="*", c=color, ec="k", zorder=5)
        ax.annotate(f, (x0, y0), xytext=(10, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=color)
    ax.set_xticks(np.radians(np.arange(-150, 151, 60)))
    ax.set_xticklabels([f"{(-t) % 360:.0f}°" for t in np.arange(-150, 151, 60)],
                       fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Our two fields on the sky (in the DP2 footprint)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, markerscale=8)

    for ci, f in enumerate(FIELDS):
        s = meta[meta.field == f]
        stat, xe, ye, _ = binned_statistic_2d(
            s.ra, s.dec, s.n_input_i, statistic="median", bins=55)
        ax = fig.add_subplot(gs[1, ci])
        im = ax.pcolormesh(xe, ye, stat.T, cmap="viridis")
        ax.set(xlabel="RA [deg]", ylabel="Dec [deg]" if ci == 0 else None,
               title=f"{f}: median i-band exposures per coadd pixel region")
        ax.invert_xaxis()
        fig.colorbar(im, ax=ax, label="n_input_i")

    ax = fig.add_subplot(gs[2, 0])
    for f, color in zip(FIELDS, ("#1f77b4", "#d62728")):
        v = meta.loc[meta.field == f, "n_input_i"].dropna()
        ax.hist(v, bins=60, alpha=0.6, label=f"{f} (median {v.median():.0f})",
                color=color, log=True)
    ax.set(xlabel="n_input_i (exposures in the i-band coadd)", ylabel="objects",
           title="i-band coadd depth distribution")
    ax.legend()

    ax = fig.add_subplot(gs[2, 1])
    x = np.arange(len(BANDS))
    for off, (f, color) in zip((-0.2, 0.2), zip(FIELDS, ("#1f77b4", "#d62728"))):
        med = [rows[f][b][0] for b in BANDS]
        ax.bar(x + off, med, width=0.4, label=f, color=color)
    ax.set_xticks(x, BANDS)
    ax.set(ylabel="median n_input", title="median exposures per band "
           "(DEEP2-3 u = none in DP2)")
    ax.set_yscale("log")
    ax.legend()

    fig.suptitle("DP2 coadd depth at our objects: COSMOS (Deep Drilling Field)"
                 " vs DEEP2-3 (wide-survey area)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "nvisits_maps.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
