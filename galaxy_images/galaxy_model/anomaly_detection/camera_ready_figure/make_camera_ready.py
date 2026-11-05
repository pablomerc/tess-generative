"""Paper Fig. 3 plus a 5th "typical galaxies" row (camera-ready candidates).

Rows 1-4 are the published top-12 stamps, verbatim: `published_top12_stamps.npz`
(from extract_published_stamps.py), placed at the published page coordinates, so
they render pixel-identical to anomaly_detection_figure_science.pdf.

Row 5 (`--variant`):
  band    per column, 3 galaxies drawn at random from that model's 40-60th
          anomaly-score percentile; stamp label = percentile.
  random  one full-width row of 9 galaxies drawn at random from the pool; inset
          = the galaxy's anomaly-score percentile under each model.

Row-5 scores are the 474k rerun restricted to the 367k paper pool (physics
ours/hsc_flat 64-d, AION hsc_mean_full no PCA, instrument ours/hsc_flat), so
they are not bit-identical to the scores behind rows 1-4 (see README.md).

    ~/miniforge3/envs/torchenv/bin/python make_camera_ready.py --variant band
    ~/miniforge3/envs/torchenv/bin/python make_camera_ready.py --variant random

Writes anomaly_detection_figure_science_row5_<variant>.{pdf,png} and
row5_<variant>.csv (row_474k, object_id, ra, dec, percentile under each model).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from make_top_median_bottom import SOURCES, POOL367K_ROWS, load_scores, open_hsc, to_rgb

HERE = Path(__file__).resolve().parent
SEED = 0
BAND = (40.0, 60.0)
SHORT = ["Ours-P", "AION-1", "Ours-I"]
N_COLS_PER = 3

# Published page geometry in PDF points, y measured downwards from the top
# (read off anomaly_detection_figure_science.pdf with PyMuPDF).
PAGE_W, OLD_PAGE_H = 892.8, 413.28
COL_X0 = [8.03, 101.9, 195.76, 307.4, 401.27, 495.14, 606.78, 700.65, 794.51]
ROW_Y0 = [28.28, 124.28, 220.27, 316.27]
STAMP = 90.36                       # 1.255 in -> 251 px at dpi=200, as published
SEP_X, SEP_Y = [296.71, 596.09], (27.36, 407.65)
TITLE_BASELINE = 19.0
ROW5_GAP = 30.0                     # row 4 bottom -> row 5 top; holds the row-5 header
BOTTOM_PAD = OLD_PAGE_H - (ROW_Y0[-1] + STAMP)
DPI = 200

SEP_STYLE = dict(color="#888888", linestyle="--", linewidth=1.2, dashes=(6, 4),
                 solid_capstyle="butt")
LABEL_BOX = dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.7, linewidth=0)


# ─── Row-5 selection ──────────────────────────────────────────────────────────

def pool_percentiles():
    """(pool_rows, pct[3, N]) with pct = rank / N * 100 per model, as in the paper."""
    pool_rows = np.load(POOL367K_ROWS)
    pct = []
    for src in SOURCES:
        s = load_scores(src)[pool_rows]
        assert np.isfinite(s).all()
        rank = np.empty(len(s), dtype=np.int64)
        rank[np.argsort(s, kind="stable")] = np.arange(len(s))
        pct.append(rank / len(s) * 100)
    return pool_rows, np.stack(pct)


def select_row5(variant, pct, rng):
    """Positions into the pool: list of 3 arrays (band) or one array of 9 (random)."""
    if variant == "band":
        return [rng.choice(np.flatnonzero((p >= BAND[0]) & (p < BAND[1])), N_COLS_PER,
                           replace=False) for p in pct]
    return [rng.choice(pct.shape[1], 3 * N_COLS_PER, replace=False)]


# ─── Figure ───────────────────────────────────────────────────────────────────

def upsample_nearest(rgb, size=251):
    """160 -> 251 px nearest-neighbour, matching how the published stamps were rasterized."""
    idx = ((np.arange(size) + 0.5) * rgb.shape[0] / size).astype(int)
    return rgb[idx][:, idx]


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def build_figure(published, row5_imgs, row5_pct, variant):
    row5_y0 = ROW_Y0[-1] + STAMP + ROW5_GAP
    H = row5_y0 + STAMP + BOTTOM_PAD
    fig = plt.figure(figsize=(PAGE_W / 72, H / 72))
    fig.patch.set_facecolor("white")

    def add_stamp(x0, y0, img):
        ax = fig.add_axes([x0 / PAGE_W, (H - y0 - STAMP) / H, STAMP / PAGE_W, STAMP / H])
        ax.set_facecolor("none"); ax.set_zorder(2)
        # "none": the PDF embeds the array as-is, so published stamps stay byte-identical.
        ax.imshow(img, interpolation="none")
        ax.axis("off")
        return ax

    def label(ax, text):
        ax.text(0.5, 0.04, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
                color="black", va="bottom", ha="center", bbox=LABEL_BOX)

    sep_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    sep_ax.set_xlim(0, PAGE_W); sep_ax.set_ylim(H, 0)
    sep_ax.set_axis_off(); sep_ax.patch.set_visible(False)
    for x in SEP_X:
        sep_ax.plot([x, x], [SEP_Y[1], SEP_Y[0]], **SEP_STYLE)
        if variant == "band":
            sep_ax.plot([x, x], [row5_y0 + STAMP + 1.0, row5_y0 - 0.9], **SEP_STYLE)

    # Rows 1-4: published stamps and labels.
    for g, src in enumerate(SOURCES):
        x_mid = (COL_X0[g * N_COLS_PER] + COL_X0[g * N_COLS_PER + 2] + STAMP) / 2
        fig.text(x_mid / PAGE_W, 1 - TITLE_BASELINE / H, src["label"], ha="center",
                 va="baseline", fontsize=17, fontweight="bold")
        for k in range(len(ROW_Y0) * N_COLS_PER):
            r, c = divmod(k, N_COLS_PER)
            label(add_stamp(COL_X0[g * N_COLS_PER + c], ROW_Y0[r], published[g, k]), f"#{k + 1}")

    # Row 5.
    header = ("Typical galaxies (random draws from each model's 40–60th score percentile)"
              if variant == "band" else
              "Random galaxies (inset: anomaly-score percentile under each model)")
    fig.text(0.5, 1 - (ROW_Y0[-1] + STAMP + ROW5_GAP * 0.70) / H, header, ha="center",
             va="baseline", fontsize=13, fontweight="bold")
    if variant == "band":
        for g in range(len(SOURCES)):
            for c in range(N_COLS_PER):
                ax = add_stamp(COL_X0[g * N_COLS_PER + c], row5_y0, row5_imgs[g][c])
                label(ax, f"{ordinal(int(row5_pct[g][c][g]))} pct.")
    else:
        n = len(row5_imgs[0])
        pitch = (COL_X0[-1] - COL_X0[0]) / (n - 1)
        for j in range(n):
            ax = add_stamp(COL_X0[0] + j * pitch, row5_y0, row5_imgs[0][j])
            x_l, x_r, y_t, dy = 0.50, 0.975, 0.965, 0.105
            ax.add_patch(Rectangle((x_l - 0.025, y_t - 3 * dy - 0.03), x_r - x_l + 0.04,
                                   3 * dy + 0.05, transform=ax.transAxes, facecolor="white",
                                   alpha=0.7, linewidth=0, zorder=3))
            for m, name in enumerate(SHORT):
                y = y_t - (m + 0.5) * dy
                kw = dict(transform=ax.transAxes, fontsize=8, fontweight="bold",
                          color="black", va="center", zorder=4)
                ax.text(x_l, y, name, ha="left", **kw)
                ax.text(x_r, y, f"{int(row5_pct[0][j][m])}", ha="right", **kw)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["band", "random"], required=True)
    args = p.parse_args()

    published = np.load(HERE / "published_top12_stamps.npz")["rgb"]
    pool_rows, pct = pool_percentiles()
    picks = select_row5(args.variant, pct, np.random.default_rng(SEED))

    flux, ra, dec, object_id = open_hsc()
    row5_imgs = [[upsample_nearest(to_rgb(np.asarray(flux[pool_rows[i]]))) for i in grp]
                 for grp in picks]
    row5_pct = [[pct[:, i] for i in grp] for grp in picks]

    csv_path = HERE / f"row5_{args.variant}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "slot", "row_474k", "object_id", "ra", "dec"]
                   + [f"pct_{s}" for s in SHORT])
        for g, grp in enumerate(picks):
            for slot, i in enumerate(grp):
                row = int(pool_rows[i])
                w.writerow([SOURCES[g]["label"] if args.variant == "band" else "all", slot + 1,
                            row, int(object_id[row]), f"{ra[row]:.6f}", f"{dec[row]:.6f}"]
                           + [f"{v:.3f}" for v in pct[:, i]])
    print(f"Saved: {csv_path}")

    import scienceplots  # noqa: F401
    # scienceplots sets savefig.bbox=tight, which would shift the published page geometry.
    with plt.style.context(["science", "no-latex", {"savefig.bbox": "standard"}]):
        fig = build_figure(published, row5_imgs, row5_pct, args.variant)
        stem = HERE / f"anomaly_detection_figure_science_row5_{args.variant}"
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", dpi=DPI)
            print(f"Saved: {stem}.{ext}")
        plt.close(fig)


if __name__ == "__main__":
    main()
