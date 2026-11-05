"""Top anomalies vs median-score vs lowest-score galaxies, per latent space.

Exploration figure for the camera-ready anomaly figure: the paper's Fig. 3
(figures_for_paper/anomaly_detection_figure/) shows only the top-12 flow
anomalies per space; this adds what *typical* galaxies look like in each space,
so we can pick a "normal-looking galaxies" row to append.

Run (login node is fine, it only reads ~100 stamps + 3 score vectors):
    python make_top_median_bottom.py                # 367k paper pool (default)
    python make_top_median_bottom.py --pool 474k    # full HSC downstream store

Inputs (all already on Engaging, no retraining / re-encoding):
  - ../outputs/anomaly_scores_{ours,aion,ins}_hscdown_474k.h5
      Flow NLL per galaxy, raw_index == arange(474954) (row in hsc_downstream).
  - /orcd/pool/007/pablomer/outputs/contrastive_anomalies/pool367k_rows.npy
      Rows of the 474k store that form the paper's 367k pool.
  - /home/pablomer/orcd/scratch/hsc_downstream/{hsc_flux.bin,catalog.parquet}
      HSC stamps (float16, N x 5 x 160 x 160) and ra/dec/object_id.

Differences from the published figure (same as engaging/build_joint_figure_hscdown.py):
  - Physics uses ours/hsc_flat (64-d) flow; the paper used ours/hsc_mean (16-d).
  - AION uses aion/hsc_mean_full (768-d, no PCA); the paper used PCA->64.
  - Instrument is the same key (ours/hsc_flat).

Selection per space (same 10" sky dedup as the paper, applied within each section):
  - top:    highest NLL, walked downwards.
  - median: ranks closest to N/2, walked outwards.
  - bottom: lowest NLL (highest flow density), walked upwards.

Writes, next to this script:
  top_median_bottom_<pool>.{png,pdf}   figure (scienceplots style)
  top_median_bottom_<pool>.csv         every displayed stamp: row, object_id, ra, dec, score, pct
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SCORES_DIR = HERE.parent / "outputs"
POOL367K_ROWS = Path("/orcd/pool/007/pablomer/outputs/contrastive_anomalies/pool367k_rows.npy")
HSC_ROOT = Path("/home/pablomer/orcd/scratch/hsc_downstream")

DEDUP_ARCSEC = 10.0
CANDIDATE_MULT = 500

# Column order matches the paper figure.
SOURCES = [
    {"label": "Ours (Physics)",    "scores_file": "anomaly_scores_ours_hscdown_474k.h5",
     "score_key": "ours/hsc_flat/flow"},
    {"label": "AION-1",            "scores_file": "anomaly_scores_aion_hscdown_474k.h5",
     "score_key": "aion/hsc_mean_full/flow"},
    {"label": "Ours (Instrument)", "scores_file": "anomaly_scores_ins_hscdown_474k.h5",
     "score_key": "ours/hsc_flat/flow"},
]

# (key, row label, number of stamps); stamps are laid out 3 per row.
SECTIONS = [
    ("top",    "Top anomalies",             12),
    ("median", "Median score\n(~50th pct.)", 9),
    ("bottom", "Lowest score",               9),
]
N_COLS_PER = 3


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_scores(src):
    with h5py.File(SCORES_DIR / src["scores_file"], "r") as f:
        assert np.array_equal(f["raw_index"][:], np.arange(f["raw_index"].shape[0]))
        return f[src["score_key"]][:]


def open_hsc():
    with open(HSC_ROOT / "meta.json") as f:
        meta = json.load(f)
    pix = meta["pixels"]
    flux = np.memmap(HSC_ROOT / pix["files"]["flux"]["path"],
                     dtype=np.dtype(pix["files"]["flux"]["dtype"]),
                     mode="r", shape=tuple(pix["shape"]))
    cat = pq.read_table(HSC_ROOT / meta["catalog"], columns=["ra", "dec", "object_id"])
    return flux, cat["ra"].to_numpy(), cat["dec"].to_numpy(), cat["object_id"].to_numpy()


# ─── Selection ────────────────────────────────────────────────────────────────

def dedup_by_radec(cand_rows, ra_all, dec_all, n_keep, min_sep_arcsec):
    """Keep candidates (in order) that are >min_sep_arcsec from every kept one.

    Small-angle separation, as in figures_for_paper/anomaly_detection_figure.
    Returns positions into `cand_rows`.
    """
    thr2 = (min_sep_arcsec / 3600.0) ** 2
    kept, k_ra, k_dec = [], [], []
    for i, row in enumerate(cand_rows):
        ra, dec = float(ra_all[row]), float(dec_all[row])
        if kept:
            d_ra = (ra - np.array(k_ra)) * np.cos(np.deg2rad(np.array(k_dec)))
            d_dec = dec - np.array(k_dec)
            if np.any(d_ra ** 2 + d_dec ** 2 < thr2):
                continue
        kept.append(i); k_ra.append(ra); k_dec.append(dec)
        if len(kept) == n_keep:
            break
    return np.array(kept)


def select_sections(scores_all, pool_rows, ra, dec):
    """Return {section: (rows, scores, pcts)} for one latent space."""
    s = scores_all[pool_rows]
    finite = np.isfinite(s)
    rows, s = pool_rows[finite], s[finite]
    n = len(s)
    order = np.argsort(s, kind="stable")          # ascending NLL
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    pct = rank / n * 100                          # paper's definition: rank / N_finite

    out = {}
    for key, _, n_keep in SECTIONS:
        if key == "top":
            cand = order[::-1]
        elif key == "bottom":
            cand = order
        else:
            cand = order[np.argsort(np.abs(np.arange(n) - (n - 1) / 2), kind="stable")]
        cand = cand[:n_keep * CANDIDATE_MULT]
        kept = cand[dedup_by_radec(rows[cand], ra, dec, n_keep, DEDUP_ARCSEC)]
        if key == "median":
            kept = kept[np.argsort(-s[kept])]
        out[key] = (rows[kept], s[kept], pct[kept])
    return out, n, int((~finite).sum())


# ─── Figure ───────────────────────────────────────────────────────────────────

def to_rgb(img_chw, pct_lo=1, pct_hi=99):
    """Percentile-stretched g/r/i RGB, identical to the paper figure."""
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def build_figure(sel, flux, footer, img=1.30, inner=0.05, group_gap=0.30,
                 section_gap=0.32, left_pad=0.42, header_pad=0.44, footer_pad=0.22):
    n_groups = len(SOURCES)
    rows_per = [int(np.ceil(n / N_COLS_PER)) for _, _, n in SECTIONS]
    group_w = N_COLS_PER * img + (N_COLS_PER - 1) * inner
    fig_w = left_pad + n_groups * group_w + (n_groups - 1) * group_gap + 0.03
    sec_h = [r * img + (r - 1) * inner for r in rows_per]
    fig_h = header_pad + sum(sec_h) + (len(SECTIONS) - 1) * section_gap + footer_pad
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    W, H = fig_w, fig_h

    def gx(g):
        return left_pad + g * (group_w + group_gap)

    # Section tops (inches from bottom).
    sec_top, y = [], H - header_pad
    for h in sec_h:
        sec_top.append(y)
        y -= h + section_gap

    # Dashed separators: vertical between groups, horizontal between sections.
    sep_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    sep_ax.set_xlim(0, W); sep_ax.set_ylim(0, H)
    sep_ax.set_axis_off(); sep_ax.patch.set_visible(False)
    y_lo = sec_top[-1] - sec_h[-1]
    for g in range(1, n_groups):
        x = gx(g) - group_gap / 2
        sep_ax.plot([x, x], [y_lo, sec_top[0]], color="#888888",
                    linestyle="--", linewidth=1.2, dashes=(6, 4))
    for i in range(1, len(SECTIONS)):
        yy = sec_top[i] + section_gap / 2
        sep_ax.plot([0.08, W - 0.03], [yy, yy], color="#888888",
                    linestyle="--", linewidth=1.2, dashes=(6, 4))

    for g, src in enumerate(SOURCES):
        fig.text((gx(g) + group_w / 2) / W, (H - 0.06) / H, src["label"],
                 ha="center", va="top", fontsize=17, fontweight="bold")
        for si, (key, _, _) in enumerate(SECTIONS):
            rows, _, pcts = sel[g][key]
            for k, (row, p) in enumerate(zip(rows, pcts)):
                r, c = divmod(k, N_COLS_PER)
                x0 = gx(g) + c * (img + inner)
                y0 = sec_top[si] - (r + 1) * img - r * inner
                ax = fig.add_axes([x0 / W, y0 / H, img / W, img / H])
                ax.imshow(to_rgb(np.asarray(flux[row])), interpolation="nearest")
                ax.axis("off")
                ax.text(0.5, 0.04, f"#{k + 1}  {p:.3f}%" if key != "median" else f"{p:.3f}%",
                        transform=ax.transAxes, fontsize=8, fontweight="bold",
                        color="black", va="bottom", ha="center",
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                                  alpha=0.7, linewidth=0))

    for si, (_, label, _) in enumerate(SECTIONS):
        fig.text(0.20 / W, (sec_top[si] - sec_h[si] / 2) / H, label,
                 rotation=90, ha="center", va="center", fontsize=13, fontweight="bold")

    fig.text(0.5, 0.06 / H, footer, ha="center", va="bottom", fontsize=8, color="#555555")
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", choices=["367k", "474k"], default="367k")
    args = p.parse_args()

    flux, ra, dec, object_id = open_hsc()
    pool_rows = (np.load(POOL367K_ROWS) if args.pool == "367k"
                 else np.arange(flux.shape[0], dtype=np.int64))

    sel = []
    for src in SOURCES:
        out, n, n_bad = select_sections(load_scores(src), pool_rows, ra, dec)
        print(f"{src['label']:<18} N_finite={n}  non-finite={n_bad}")
        sel.append(out)

    stem = HERE / f"top_median_bottom_{args.pool}"
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "score_key", "section", "rank_in_section", "row_474k",
                    "object_id", "ra", "dec", "flow_nll", "pct"])
        for src, out in zip(SOURCES, sel):
            for key, _, _ in SECTIONS:
                for k, (row, s, pc) in enumerate(zip(*out[key])):
                    w.writerow([src["label"], src["score_key"], key, k + 1, int(row),
                                int(object_id[row]), f"{ra[row]:.6f}", f"{dec[row]:.6f}",
                                f"{s:.4f}", f"{pc:.4f}"])

    footer = (f"Flow NLL on the {args.pool} pool (474k rerun scores: physics & instrument "
              f"ours/hsc_flat 64-d, AION hsc_mean_full 768-d no PCA); 10\" sky dedup; "
              f"label = score percentile (100 = most anomalous)")
    import scienceplots  # noqa: F401
    with plt.style.context(["science", "no-latex"]):
        fig = build_figure(sel, flux, footer)
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
            print(f"Saved: {stem}.{ext}")
        plt.close(fig)
    print(f"Saved: {stem}.csv")


if __name__ == "__main__":
    main()
