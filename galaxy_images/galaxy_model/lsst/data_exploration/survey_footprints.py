#!/usr/bin/env python3
"""All-sky overlay of the three surveys' available data.

  - Legacy Surveys DR10 (public): ~20,000 deg^2 — approximated as the
    extragalactic sky (|galactic b| > 18 deg) between Dec -65 and +84.
  - HSC-SSP PDR3 (public): wide strips + deep/ultradeep fields, drawn as
    approximate rectangles from the PDR3 release maps.
  - LSST DP2 (data-rights): digitized deep-coadd footprint with the official
    field names (dp2.lsst.io observations tables).
  - our raw_mmu dataset: the three patches, for context.

All non-DP2 outlines are APPROXIMATE (good to ~a degree) — meant for
orientation, not area calculations.

  PYTHONPATH=. python galaxy_images/galaxy_model/lsst/data_exploration/survey_footprints.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COVER = os.path.join(HERE, "figures", "dp2_coverage_radec.npz")
POSITIONS = ("/work1/jeroenaudenaert/pablomer/cross_matching/outputs/"
             "dataset_positions.parquet")

DP2_FIELDS = {  # dp2.lsst.io/overview/observations.html tables 1-2
    "DDF ELAIS S1": (9.5, -44.0), "DDF XMM LSS": (35.6, -4.8),
    "DDF ECDFS": (53.0, -28.1), "DDF EDFS": (61.2, -48.5),
    "DDF COSMOS": (150.1, 2.1), "Abell 2764": (5.5, -49.0),
    "DESI SV3 R1": (180.5, -0.3), "M49": (186.3, 6.9),
    "Prawn": (253.5, -41.0), "Trifid-Lagoon": (271.7, -23.9),
    "New Horizons": (289.4, -20.2), "SV 212 -7": (211.7, -7.0),
    "SV 216 -17": (216.1, -16.7), "SV 225 -40": (225.0, -39.5),
    "SV 280 -48": (280.1, -48.0), "SV 300 -41": (300.3, -41.0),
    "SV 320 -15": (320.2, -15.1),
}

# HSC-SSP PDR3 public footprint, approximate rectangles (ra0, ra1, dec0, dec1)
HSC_WIDE = [
    (330.0, 40.0, -7.0, 6.5),      # autumn equatorial strip (crosses RA 0)
    (127.5, 227.0, -2.5, 5.5),     # spring equatorial strip
    (195.0, 255.0, 41.5, 45.0),    # HECTOMAP
    (212.0, 216.5, 51.0, 53.5),    # AEGIS
]
HSC_DEEP = [
    (33.7, 37.5, -6.0, -3.4),      # XMM-LSS / SXDS
    (148.7, 151.5, 0.8, 3.4),      # (E-)COSMOS
    (240.1, 245.4, 52.7, 56.5),    # ELAIS-N1
    (350.0, 354.0, -1.6, 1.2),     # DEEP2-3
]


def to_xy(ra_deg, dec_deg):
    lon = np.remainder(np.atleast_1d(np.asarray(ra_deg, float)) + 360, 360)
    lon[lon > 180] -= 360
    return np.radians(-lon), np.radians(np.atleast_1d(dec_deg))


def galactic_b(ra_deg, dec_deg):
    """Galactic latitude in deg (NGP at RA 192.859, Dec 27.128)."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    ngp_ra, ngp_dec = np.radians(192.859), np.radians(27.128)
    sinb = (np.sin(dec) * np.sin(ngp_dec) +
            np.cos(dec) * np.cos(ngp_dec) * np.cos(ra - ngp_ra))
    return np.degrees(np.arcsin(np.clip(sinb, -1, 1)))


def rect_paths(r0, r1, d0, d1, n=60):
    """Rect edge path(s) in map coords; splits at the RA=180 seam."""
    if r1 < r0:                                     # crosses RA 0 (map center)
        segs = [(r0, 360.0), (0.0, r1)] if r0 > 180 else [(r0, r1)]
        segs = [(r0, 359.999), (0.001, r1)]
    elif r0 < 180 < r1:                             # crosses the seam
        segs = [(r0, 179.999), (180.001, r1)]
    else:
        segs = [(r0, r1)]
    out = []
    for a, b in segs:
        ra = np.concatenate([np.linspace(a, b, n), [b] * n,
                             np.linspace(b, a, n), [a] * n])
        dec = np.concatenate([[d0] * n, np.linspace(d0, d1, n),
                              [d1] * n, np.linspace(d1, d0, n)])
        out.append(to_xy(ra, dec))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8.5))
    ax = fig.add_subplot(111, projection="mollweide")

    # Legacy Surveys DR10 (approx), shaded by sub-survey / instrument:
    #   DECaLS+DES ("des-" prefix in raw_mmu!): DECam on the 4m Blanco (Chile),
    #       Dec <~ +32.4, g/r/(i)/z
    #   BASS (Bok 2.3m, g/r) + MzLS (Mayall 4m, z): north galactic cap only,
    #       Dec >~ +32.4
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    lon_e = np.linspace(-np.pi, np.pi, 721)
    lat_e = np.linspace(-np.pi / 2, np.pi / 2, 361)
    LON, LAT = np.meshgrid((lon_e[:-1] + lon_e[1:]) / 2,
                           (lat_e[:-1] + lat_e[1:]) / 2)
    RA = np.remainder(-np.degrees(LON), 360)
    DEC = np.degrees(LAT)
    b = galactic_b(RA, DEC)
    cat = np.full(RA.shape, np.nan)
    cat[(DEC > -65) & (DEC <= 32.4) & (np.abs(b) > 18)] = 0     # DECam
    cat[(DEC > 32.4) & (DEC < 84) & (b > 0)] = 1                # BASS+MzLS
    ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(cat),
                  cmap=ListedColormap(["#c9e8c5", "#e4dcf0"]),
                  vmin=0, vmax=1, zorder=0.5, shading="flat", rasterized=True)
    legacy_handles = [
        Patch(fc="#c9e8c5", label="Legacy DR10 south — DECam @ Blanco 4m, "
                                  "Chile (public, approx)"),
        Patch(fc="#e4dcf0", label="Legacy DR10 north — BASS (Bok 2.3m) + "
                                  "MzLS (Mayall 4m), Kitt Peak (approx)"),
    ]

    # DP2 digitized footprint
    cov = np.load(COVER)
    x, y = to_xy(cov["ra"], cov["dec"])
    ax.scatter(x, y, s=0.4, c="#9ecae1", lw=0, zorder=2,
               label="LSST DP2 deep coadds (data rights, ~3000 deg²)")

    # HSC PDR3 rectangles
    first = True
    for rect in HSC_WIDE:
        for xp, yp in rect_paths(*rect):
            ax.fill(xp, yp, color="#fdae6b", alpha=0.55, lw=0, zorder=3,
                    label="HSC-SSP PDR3 wide (public, approx)"
                    if first else None)
            first = False
    first = True
    for rect in HSC_DEEP:
        for xp, yp in rect_paths(*rect):
            ax.fill(xp, yp, color="#d94801", alpha=0.85, lw=0, zorder=4,
                    label="HSC-SSP PDR3 deep/ultradeep (public, approx)"
                    if first else None)
            first = False

    # our dataset patches
    pos = pd.read_parquet(POSITIONS, columns=["ra", "dec"]).dropna()
    x, y = to_xy(pos.ra.to_numpy(), pos.dec.to_numpy())
    ax.scatter(x, y, s=0.4, c="crimson", lw=0, zorder=5,
               label="our raw_mmu dataset (101,725 galaxies)")

    # DP2 field labels, official-figure style
    for name, (r, d) in DP2_FIELDS.items():
        xm, ym = to_xy(r, d)
        ax.scatter([float(xm[0])], [float(ym[0])], s=10, c="#08306b",
                   marker="D", zorder=6)
        ax.annotate(name, (float(xm[0]), float(ym[0])), xytext=(3, 4),
                    textcoords="offset points", fontsize=6.5,
                    color="#08306b", zorder=6)

    ax.set_xticks(np.radians(np.arange(-150, 151, 60)))
    ax.set_xticklabels([f"{(-t) % 360:.0f}°" for t in np.arange(-150, 151, 60)],
                       fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("What data exists where: Legacy DR10 vs HSC PDR3 vs LSST DP2"
                 " (with our dataset)", fontsize=13)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legacy_handles + handles, loc="lower right",
              fontsize=8, markerscale=10)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "survey_footprints.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
