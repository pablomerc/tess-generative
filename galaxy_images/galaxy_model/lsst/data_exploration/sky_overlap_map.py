#!/usr/bin/env python3
"""The definitive overlap map: ALL 101,725 raw_mmu galaxies vs DP2.

Top: all-sky Mollweide with every object of the original HSC/Legacy
crossmatched dataset (3 patches), the DP2 deep-coadd footprint, and — for
history — the approximate DP1 field positions (DP1 overlapped none of our
patches; DP2 added COSMOS DDF + the wide band).
Bottom: one zoom per patch showing objects over the DP2 coverage, which makes
the per-field verdict unambiguous (XMM-LSS: no coadd coverage under it).

Positions come from cross_matching/outputs/dataset_positions.parquet
(all objects incl. XMM-LSS; recovered via the MMU HATS id join).

  PYTHONPATH=. python galaxy_images/galaxy_model/lsst/data_exploration/sky_overlap_map.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS = ("/work1/jeroenaudenaert/pablomer/cross_matching/outputs/"
             "dataset_positions.parquet")
COVER = os.path.join(HERE, "figures", "dp2_coverage_radec.npz")

PATCHES = [  # name, in-DP2 verdict, zoom box (ra0, ra1, dec0, dec1)
    ("XMM-LSS", "NOT in DP2 (visited, no coadds)", (32.5, 39.0, -7.0, -2.2)),
    ("COSMOS", "in DP2 (Deep Drilling Field)", (147.5, 153.0, 0.0, 4.5)),
    ("DEEP2-3", "in DP2 (wide-survey band)", (349.0, 355.5, -2.7, 2.0)),
]
DP1_FIELDS = {  # approximate centers, for the history panel
    "47 Tuc": (6.0, -72.1), "Fornax dSph": (40.0, -34.4),
    "ECDFS": (53.1, -28.1), "EDFS": (59.1, -48.7),
    "Rubin SV 38 7": (38.0, 7.0), "Rubin SV 95 -25": (95.0, -25.0),
    "Seagull": (106.2, -10.5),
}


def to_xy(ra_deg, dec_deg):
    lon = np.remainder(np.atleast_1d(np.asarray(ra_deg, float)) + 360, 360)
    lon[lon > 180] -= 360
    return np.radians(-lon), np.radians(np.atleast_1d(dec_deg))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = pd.read_parquet(POSITIONS, columns=["ra", "dec"]).dropna()
    cov = np.load(COVER)
    print(f"{len(pos)} dataset objects; {len(cov['ra'])} DP2 coverage points")

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])

    ax = fig.add_subplot(gs[0, :], projection="mollweide")
    x, y = to_xy(cov["ra"], cov["dec"])
    ax.scatter(x, y, s=0.3, c="#bdd7e7", lw=0,
               label="LSST DP2 deep-coadd footprint")
    x, y = to_xy(pos.ra.to_numpy(), pos.dec.to_numpy())
    ax.scatter(x, y, s=0.5, c="crimson", lw=0,
               label=f"our HSC/Legacy dataset ({len(pos):,} objects)")
    for name, verdict, (r0, r1, d0, d1) in PATCHES:
        xm, ym = to_xy((r0 + r1) / 2, d1)
        ax.annotate(f"{name}\n({verdict})", (float(xm[0]), float(ym[0])),
                    xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=8, fontweight="bold", color="crimson")
    for name, (r, d) in DP1_FIELDS.items():
        xm, ym = to_xy(r, d)
        ax.scatter([float(xm[0])], [float(ym[0])], s=45, marker="s",
                   fc="none", ec="gray", lw=1)
    ax.scatter([], [], s=45, marker="s", fc="none", ec="gray",
               label="DP1 fields (approx) — none overlap us")
    ax.set_xticks(np.radians(np.arange(-150, 151, 60)))
    ax.set_xticklabels([f"{(-t) % 360:.0f}°" for t in np.arange(-150, 151, 60)],
                       fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, markerscale=8)
    ax.set_title("All 101,725 raw_mmu galaxies (3 patches) vs the LSST DP2 "
                 "footprint", fontsize=12)

    for ci, (name, verdict, (r0, r1, d0, d1)) in enumerate(PATCHES):
        ax = fig.add_subplot(gs[1, ci])
        m = (cov["ra"] > r0) & (cov["ra"] < r1) & \
            (cov["dec"] > d0) & (cov["dec"] < d1)
        ax.scatter(cov["ra"][m], cov["dec"][m], s=14, marker="s", c="#bdd7e7",
                   lw=0, label="DP2 coadds")
        s = pos[(pos.ra > r0) & (pos.ra < r1) &
                (pos.dec > d0) & (pos.dec < d1)]
        ax.scatter(s.ra, s.dec, s=0.4, c="crimson", lw=0)
        ax.set(xlim=(r1, r0), ylim=(d0, d1), xlabel="RA [deg]",
               title=f"{name}: {len(s):,} objects\n{verdict}")
        if ci == 0:
            ax.set_ylabel("Dec [deg]")
            ax.legend(loc="lower left", fontsize=7, markerscale=2)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "sky_overlap_map.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
