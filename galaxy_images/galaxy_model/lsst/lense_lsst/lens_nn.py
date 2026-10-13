"""Known HSC lenses as queries: nearest neighbours in the LSST physics latent space of COSMOS.

Query   = physics latent of the lens's HSC image.
Gallery = physics latents of the LSST images of all 33,346 COSMOS anchors, i.e. a cross-survey
          search: HSC image in, LSST images out. Euclidean distance.

The physics latent is 4 spatial tokens x 16 (ResNet-18 -> 2x2 feature map -> 16 channels).
--metric picks how it is compared:
  mean16           tokens averaged to 16-d (first report, 2026-10-07 `cosine-cosmos-13`)
  flat64           all 4 x 16 = 64 numbers
  flat64_dihedral  all 64 numbers, orientation ignored: the query HSC stamp is rotated/flipped
                   8 ways (dihedral group of the square), each version re-encoded by the physics
                   encoder, and each gallery object gets the smallest of the 8 distances.
                   (default; needs the checkpoint, CPU is enough: 13 x 8 encodings)
The token-permutation shortcut (permute the 2x2 token grid instead of re-encoding rotated
images) is always computed alongside flat64_dihedral as a diagnostic, never used for ranking.

The lens's own LSST image is KEPT in the ranking and highlighted (red frame) — its rank is a
sanity check of HSC<->LSST alignment in physics space. Other known lenses are framed gold.

Gallery latents are not recomputed: they come from the anomaly stage's `cosine-anchors-cond`
run, which encoded every COSMOS anchor through the cosine checkpoint (best-epoch=20-step=10000).

    python -m galaxy_images.galaxy_model.lsst.lense_lsst.lens_nn --tag cosine-cosmos-13-flat64-dihedral
    python -m galaxy_images.galaxy_model.lsst.lense_lsst.lens_nn --tag X --metric mean16 --stretch linear
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/work1/jeroenaudenaert/pablomer")
LATENTS = ROOT / "outputs/hsc_lsst/anomaly/cosine-anchors-cond/latents.h5"
LENS_TXT = ROOT / "data/lens_indices.txt"
LENS_H5 = ROOT / "tess-generative/galaxy_images/galaxy_model/lense_reconstruction/lens_reconstruction_dataset.h5"
LSST48 = ROOT / "data/lsst_efficient/lsst48.bin"
LSST_META = ROOT / "data/lsst_efficient/lsst_meta.parquet"
HSC48 = ROOT / "data/efficient_neighs_48/hsc48.bin"
NEIGHBOURS_H5 = ROOT / "data/neighbours_v2.h5"          # parent file: raw 160 px HSC cutouts of every object
OUT_ROOT = Path(__file__).resolve().parent / "outputs"
CROP = 48
METRIC_LABEL = {"mean16": "16-d token mean", "flat64": "64-d (4 tokens × 16)",
                "flat64_dihedral": "64-d, orientation ignored (min over 8 rotations/flips of the query image)",
                "flat64_tokperm": "64-d, token-grid permutation shortcut"}
METRIC_TABLE = {"mean16": "16-d\nmean", "flat64": "64-d", "flat64_dihedral": "64-d\nrot/flip",
                "flat64_tokperm": "64-d\ntoken-perm"}
METRIC_SHORT = {"mean16": "16-d mean", "flat64": "64-d", "flat64_dihedral": "64-d rot/flip",
                "flat64_tokperm": "64-d token-perm"}

# dihedral group of the square: T(x) = rot90(flipLR?(x), k) on the last two axes
DIHEDRAL = [(flip, k) for flip in (False, True) for k in range(4)]
T_NAMES = ["id", "rot90", "rot180", "rot270", "flip", "flip+rot90", "flip+rot180", "flip+rot270"]


def dihedral(a: np.ndarray, flip: bool, k: int) -> np.ndarray:
    if flip:
        a = a[..., ::-1]
    return np.ascontiguousarray(np.rot90(a, k, axes=(-2, -1)))


def token_perm(flip: bool, k: int) -> np.ndarray:
    """Token order after applying T to the 2x2 grid (tokens are row-major over the feature map,
    t = h*2 + w, and the feature map's (h, w) follow the image's (H, W))."""
    return dihedral(np.arange(4).reshape(2, 2), flip, k).ravel()


# ------------------------------------------------------------------- data
def load_latents(path: Path):
    with h5py.File(path, "r") as f:
        lat = {k: f[k][:] for k in ("phys/hsc_mean", "phys/lsst_mean", "phys/hsc_flat", "phys/lsst_flat")}
        meta = {k: f[f"meta/{k}"][:] for k in f["meta"].keys()}
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    meta["object_id_hsc"] = meta["object_id_hsc"].astype(str)
    return lat, meta, attrs


def load_lenses(anchor_ids: np.ndarray):
    """The HSC lens list, restricted to lenses that are COSMOS anchors (positions into the latents)."""
    ids = pd.read_csv(LENS_TXT, sep="\t", dtype={"hsc_object_id": str})["hsc_object_id"].to_numpy()
    with h5py.File(LENS_H5, "r") as f:
        h5_ids = np.array([x.decode() for x in f["object_id_hsc"][:]])
        ra, dec = f["ra"][:], f["dec"][:]
        big = f["images_hsc"][:]                       # (74, 5, 160, 160) grizy linear flux; context only
    pos_of = {s: i for i, s in enumerate(anchor_ids)}
    keep = [i for i, s in enumerate(h5_ids) if s in pos_of]
    lenses = pd.DataFrame({
        "object_id_hsc": h5_ids[keep], "ra": ra[keep], "dec": dec[keep],
        "pos": [pos_of[s] for s in h5_ids[keep]], "h5_row": keep,
    })
    return lenses, big, len(ids), len(h5_ids)


class Stamps:
    """48 px HSC / LSST stamps by seek+read (no 8.6 GB mapping)."""
    rowbytes = 4 * CROP * CROP * 2

    def __init__(self):
        self.fh_hsc = open(HSC48, "rb", buffering=0)
        self.fh_lsst = open(LSST48, "rb", buffering=0)

    def _read(self, fh, row):
        fh.seek(int(row) * self.rowbytes)
        return np.frombuffer(fh.read(self.rowbytes), dtype=np.float16).reshape(4, CROP, CROP).astype(np.float32)

    def hsc(self, parent_row):
        return self._read(self.fh_hsc, parent_row)

    def lsst(self, lsst_row):
        return self._read(self.fh_lsst, lsst_row)


class ContextCutouts:
    """27″ (160 px) raw HSC grizy cutouts of any object, by HSC object id, from neighbours_v2.h5 —
    the parent file the 48 px HSC stamps were centre-cropped from."""

    def __init__(self):
        self.f = h5py.File(NEIGHBOURS_H5, "r")
        ids, st = self.f["object_id_hsc"][:], self.f["source_type"][:]
        self.row_of = {}
        for i in np.argsort(st != 0, kind="stable"):           # source_type 0 rows first, so they win
            s = ids[i].decode() if isinstance(ids[i], bytes) else str(ids[i])
            self.row_of.setdefault(s, int(i))

    def get(self, object_id: str) -> np.ndarray:
        return self.f["images_hsc"][self.row_of[object_id]]     # (5, 160, 160) linear flux, grizy


def check_context(ctx: dict, stamps, meta) -> float:
    """Each cutout, run through the model's HSC preprocessing, must reproduce the stored 48 px stamp
    (same object, same centring). Returns the worst max |diff| (float16 rounding ≈ 4e-3)."""
    from galaxy_images.galaxy_model.neighbors import preprocess_raw_image
    worst = 0.0
    for p, img in ctx.items():
        got = preprocess_raw_image(img, survey="hsc", crop_size=CROP)[:4].numpy()
        worst = max(worst, float(np.abs(got - stamps.hsc(meta["parent_row"][p])).max()))
    return worst


def encode_dihedral(ckpt: str, stamps: list[np.ndarray]) -> np.ndarray:
    """(n, 8, 64) physics latents of the 8 rotations/flips of each 48 px HSC stamp (CPU)."""
    import torch
    from galaxy_images.galaxy_model.lsst.hsc_lsst_anomaly import load_model
    torch.set_num_threads(8)
    model = load_model(ckpt, "cpu")
    x = np.stack([dihedral(s, fl, k) for s in stamps for fl, k in DIHEDRAL]).astype(np.float32)
    with torch.no_grad():
        z = model.encoder_1(torch.from_numpy(x))           # (n*8, 4, 16)
    return z.flatten(1).numpy().reshape(len(stamps), len(DIHEDRAL), -1)


# ------------------------------------------------------------------- search
def l2(q: np.ndarray, db: np.ndarray) -> np.ndarray:
    """q (m, D) or (D,), db (N, D) -> (m, N) or (N,) Euclidean distances (float64)."""
    q2 = np.atleast_2d(q).astype(np.float64)
    d = np.sqrt(np.maximum(((q2[:, None, :] - db[None].astype(np.float64)) ** 2).sum(-1), 0)) if len(q2) == 1 else \
        np.sqrt(np.maximum((q2 ** 2).sum(1)[:, None] - 2 * q2 @ db.T.astype(np.float64)
                           + (db.astype(np.float64) ** 2).sum(1)[None], 0))
    return d[0] if q.ndim == 1 else d


def self_rank(order: np.ndarray, pos: int) -> int:
    return int(np.where(order == pos)[0][0]) + 1          # 1-based


# ------------------------------------------------------------------- display
def _rgb(x, stretch="asinh", q=4.0):
    """i,r,g -> RGB.
    linear: 1–99.5 pct linear stretch (anomaly galleries, first report).
    asinh : background (5th pct) to PEAK, asinh with softening q — cores keep their shape,
            faint structure (arcs, blue companions) is lifted. 48 px stamps are already
            range-compressed model inputs, so a gentle q is enough.
    asinh_ctx: for the raw linear-flux 160 px context cutouts (bg to 99.9 pct, q = 15)."""
    x = np.asarray(x, dtype=np.float32)
    img = np.stack([x[2], x[1], x[0]], -1)
    if stretch == "linear":
        lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    if stretch == "asinh_ctx":
        lo, hi, q = np.percentile(img, 1), np.percentile(img, 99.9), 15.0
    else:
        lo, hi = np.percentile(img, 5), img.max()
    y = np.clip((img - lo) / max(hi - lo, 1e-6), 0, None)
    return np.clip(np.arcsinh(q * y) / np.arcsinh(q), 0, 1)


def _frame(ax, color, lw=3):
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(lw)


def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _show(ax, img, stretch, label=None):
    ax.imshow(_rgb(img, stretch), origin="lower"); _blank(ax)
    if label:
        ax.set_xlabel(label, fontsize=5.5, labelpad=3)


def _text_block(fig, x, y, paras, width=150, size=9, dy=0.026, **kw):
    """Paragraphs wrapped at `width` chars, continuation lines flush with the first."""
    for p in paras:
        for line in textwrap.wrap(p, width) or [""]:
            fig.text(x, y, line, fontsize=size, va="top", **kw)
            y -= dy
        y -= dy * 0.35
    return y


# ------------------------------------------------------------------- pages
def page_summary(pdf, plt, args, attrs, lenses, n_list, n_h5, n_db, metrics, hits, stretch_note):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.955, "Known HSC lenses → nearest neighbours in LSST physics latent space (COSMOS)",
             fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.915, f"Ranking metric: {METRIC_LABEL[args.metric]}", fontsize=11, va="top", color="#1f4e9c")
    ck = Path(attrs["checkpoint"])
    y = _text_block(fig, 0.05, 0.875, [
        f"Model: HSC×LSST dual encoder, cosine LR schedule, checkpoint {ck.name} ({ck.parent.parent.name}). "
        f"Gallery latents from anomaly run '{attrs['tag']}' (no re-encoding of the 33,346 COSMOS objects).",
        "Physics latent: ResNet-18 on the 48 px griz stamp → 2×2 feature map → 4 spatial tokens × 16 numbers = 64 numbers. "
        "'16-d mean' averages the 4 tokens (loses the layout); '64-d' keeps all 64 (keeps the layout, so it depends on orientation).",
        "Orientation ignored (this report): each lens's HSC stamp is rotated by 0/90/180/270° with and without a mirror flip "
        "(8 versions), each version is passed through the physics encoder, and every COSMOS object gets the smallest of its "
        "8 distances. The neighbour titles name the version that matched when it is not the original ('rot90', 'flip', …).",
        f"Query: the lens's HSC 48 px stamp. Searched: the LSST 48 px stamps of all {n_db:,} COSMOS anchors "
        f"(at least 5 LSST visits in each of g, r, i, z). Euclidean distance. Top {args.k_gallery} shown per lens, "
        f"top {args.k_table} in neighbours.csv. The ranked list includes the lens's own LSST image.",
        f"Lens list: {n_list} HSC lenses (data/lens_indices.txt); {len(lenses)} are COSMOS anchors. All {len(lenses)} are in "
        "the model's TRAIN split (no holdout, DECISIONS D7): the search is unsupervised, but these are not unseen objects.",
        "Frames: red = the lens's own LSST image (kept on purpose: its rank tests HSC-LSST alignment); gold = another of "
        f"the {len(lenses)} known lenses. Each neighbour is shown as a pair: the LSST stamp that was retrieved (left) and the "
        + ("same object's 27″ HSC cutout (right, context only; the dashed box marks the 8″ region of the 48 px stamps). "
           if args.pair_hsc == "context" else "same object's HSC stamp (right, context only). ")
        + "n_i = number of LSST visits in i at that position.",
        stretch_note,
    ], width=138, size=8.8, dy=0.0235)

    y -= 0.01
    fig.text(0.05, y, "Results at a glance", fontsize=11, weight="bold", va="top"); y -= 0.033
    rows = []
    for key in metrics:
        r = lenses[f"rank_{key}"]
        rows.append(f"{METRIC_SHORT[key]:<16} own LSST image at rank 1 for {(r == 1).sum():>2}/{len(lenses)}, "
                    f"in top 20 for {(r <= 20).sum():>2}/{len(lenses)}; median rank {r.median():>4.0f}; "
                    f"worst {r.max():>4d} ({lenses.loc[r.idxmax(), 'L']})")
    y = _text_block(fig, 0.05, y, rows, width=200, size=8.8, dy=0.024, family="monospace")
    exp20 = len(lenses) * (len(lenses) - 1) * 20 / n_db
    exp100 = len(lenses) * (len(lenses) - 1) * args.k_table / n_db
    h20 = int(lenses.hits20.sum()); h100 = int(lenses.hits100.sum())
    pairs = ", ".join(f"{h['lens']}→{h['hit']} (#{h['rank']})" for h in hits) or "none"
    y = _text_block(fig, 0.05, y - 0.005, [
        f"Other known lenses among the neighbours ({METRIC_SHORT[args.metric]}): {h20} in the top 20 and {h100} in the top "
        f"{args.k_table}, summed over the {len(lenses)} queries; by chance ≈ {exp20:.2f} and ≈ {exp100:.2f}. Hits: {pairs}. "
        "Caveat: lens galaxies are a biased population (massive red early-types), so a uniform-random baseline overstates "
        "how surprising this is, and mutual pairs are not independent.",
        f"Random baseline for the own-image rank: median ≈ {n_db // 2:,}.",
        *([f"Why the own-image rank is worse with rot/flip than with plain 64-d: the lens's own LSST image has the same "
           f"orientation as the query, so it is matched by the unrotated version ({(lenses.self_transform == 'id').sum()} "
           f"of {len(lenses)} lenses), while every other object gets 8 chances to come closer. This is expected from "
           "ignoring orientation, not a failure of HSC-LSST alignment (plain 64-d puts the own image first for "
           f"{(lenses.rank_flat64 == 1).sum()} of {len(lenses)})."] if "self_transform" in lenses and args.metric == "flat64_dihedral" else []),
    ], width=138, size=8.8, dy=0.0235)
    pdf.savefig(fig); plt.close(fig)


def page_table(pdf, plt, args, lenses, metrics):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, "Per-lens table", fontsize=13, weight="bold", va="top")
    _text_block(fig, 0.05, 0.91, [
        "Own-image rank = where the lens's own LSST image lands among 33,346 (1 = nearest). "
        f"d_self / d_20 and the known-lens hits use the ranking metric ({METRIC_SHORT[args.metric]}, shaded). "
        "'top-20 shared with 16-d mean' = how many of this report's top 20 were also in the first report's top 20."],
        width=150, size=8.5, dy=0.024)
    cols = ["", "HSC object id", "RA", "Dec", "LSST visits\ng/r/i/z"] + \
           [f"own rank\n{METRIC_TABLE[k]}" for k in metrics] + ["d_self / d_20", "known lenses\ntop 20 / 100"]
    if "overlap_mean16" in lenses:
        cols.append("top-20\nshared w/\n16-d mean")
    rows = []
    for _, r in lenses.iterrows():
        row = [r.L, r.object_id_hsc, f"{r.ra:.4f}", f"{r.dec:.4f}", r.nin] + [str(r[f"rank_{k}"]) for k in metrics] + \
              [f"{r.d_self:.2f} / {r.d20:.2f}", f"{r.hits20} / {r.hits100}"]
        if "overlap_mean16" in lenses:
            row.append(f"{r.overlap_mean16} / 20")
        rows.append(row)
    widths = [0.035, 0.15, 0.065, 0.06, 0.1] + [0.075] * len(metrics) + [0.085, 0.085] + \
             ([0.1] if "overlap_mean16" in lenses else [])
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.82]); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center", colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(7.8); t.scale(1, 2.0)
    for (rr, cc), cell in t.get_celld().items():
        if rr == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#e8e8e8"); cell.set_height(cell.get_height() * 1.9)
        if rr > 0 and cols[cc] == f"own rank\n{METRIC_TABLE[args.metric]}":
            cell.set_facecolor("#e3ecfa")
    pdf.savefig(fig); plt.close(fig)


def page_method(pdf, plt, args, lenses, diag, stamps, meta, stretch):
    """Orientation diagnostics: how far rotating the query moves its latent, how good the
    token-permutation shortcut is, and which rotation the neighbours matched."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 8, figure=fig, height_ratios=[0.8, 1.6], hspace=0.38, wspace=0.25,
                  left=0.06, right=0.97, top=0.88, bottom=0.34)
    fig.suptitle("Method check: how much does orientation matter in the 64-d physics latent?", fontsize=12, y=0.965)
    fig.text(0.06, 0.925, f"Row 1: the 8 rotated/flipped versions of {lenses.L[0]}'s HSC query stamp, each encoded "
             "separately. Below: diagnostics over all 13 lenses.", fontsize=8.5, va="top")
    p0 = int(lenses.pos[0])
    s0 = stamps.hsc(meta["parent_row"][p0])
    for j, (fl, k) in enumerate(DIHEDRAL):
        ax = fig.add_subplot(gs[0, j]); _show(ax, dihedral(s0, fl, k), stretch)
        ax.set_title(T_NAMES[j], fontsize=8)

    shift = diag["shift"]            # (n_lens, 8)  ||f(Tq) - f(q)||
    perr = diag["proxy_rel_err"]     # (n_lens, 8)  ||f(Tq) - P_T f(q)|| / ||f(Tq) - f(q)||
    x = np.arange(1, len(DIHEDRAL))
    row2 = gs[1, :].subgridspec(1, 3, width_ratios=[3, 3, 2.2], wspace=0.42)
    ax = fig.add_subplot(row2[0])
    ax.boxplot([shift[:, j] for j in x], positions=x, widths=0.6, showfliers=True)
    ax.axhline(np.median(lenses.d_self_flat64), color="red", ls="--", lw=1, label="median d(own LSST image), 64-d")
    ax.axhline(np.median(lenses.d20_flat64), color="k", ls=":", lw=1, label="median d(20th neighbour), 64-d")
    ax.set_xticks(x); ax.set_xticklabels(T_NAMES[1:], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("latent distance", fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title("How far rotating/flipping the query\nmoves its 64-d latent", fontsize=9)
    ax.set_ylim(top=float(shift[:, 1:].max()) * 1.32)          # headroom so the legend covers no point
    ax.legend(fontsize=6.5, loc="upper left")

    ax = fig.add_subplot(row2[1])
    ax.boxplot([perr[:, j] for j in x], positions=x, widths=0.6)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(T_NAMES[1:], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, max(1.3, float(np.nanmax(perr[:, 1:])) * 1.05)); ax.tick_params(labelsize=7)
    ax.set_ylabel("shortcut error / move", fontsize=8)
    ax.set_title("Token-grid permutation shortcut vs\nactually re-encoding the rotated image", fontsize=9)
    ax.text(0.02, 0.02, "0 = shortcut exact;  1 = no better than not rotating", transform=ax.transAxes, fontsize=6.5)

    ax = fig.add_subplot(row2[2])
    cnt = diag["t_counts"]
    ax.barh(np.arange(len(T_NAMES)), cnt, color="#4a78c2")
    ax.set_yticks(np.arange(len(T_NAMES))); ax.set_yticklabels(T_NAMES, fontsize=7); ax.invert_yaxis()
    ax.tick_params(labelsize=7); ax.set_xlabel("neighbours (13 lenses × top 20)", fontsize=7.5)
    ax.set_title("Which version matched\nthe top-20 neighbours", fontsize=9)

    med_shift = float(np.median(shift[:, 1:])); med_self = float(np.median(lenses.d_self_flat64))
    med20 = float(np.median(lenses.d20_flat64)); med_err = float(np.median(perr[:, 1:]))
    frac_id = cnt[0] / max(cnt.sum(), 1)
    _text_block(fig, 0.06, 0.235, [
        f"Median move from rotating/flipping the query = {med_shift:.2f}, vs median distance to the lens's own LSST image "
        f"{med_self:.2f} and to the 20th neighbour {med20:.2f} (64-d): orientation moves the 64-d latent further than "
        f"the HSC-to-LSST survey change does. {frac_id:.0%} of top-20 neighbours matched the unrotated query.",
        f"Shortcut median relative error {med_err:.2f} (> 1: permuting the token grid is worse than not rotating at all). "
        "Part of this is that each of the 4 token positions carries its own average offset (the 2×2 map is not symmetric "
        "under the network's strides and padding), so any swap of positions is penalised. Diagnostic only; not used for ranking.",
        "Rotation names follow np.rot90 on the stored array; the stamps are drawn with the origin at the bottom, so "
        "'rot90' appears as a 90° clockwise turn on the page."],
        width=150, size=8.5, dy=0.024)
    pdf.savefig(fig); plt.close(fig)


def page_overview(pdf, plt, lenses, big, stamps, meta, stretch, metric):
    # 13 lenses × (HSC 160 px context, HSC 48 px query, LSST 48 px) in 5 rows of 3 lenses
    fig, axes = plt.subplots(5, 9, figsize=(11, 8.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.03, wspace=0.06, hspace=0.42)
    for ax in axes.flat:
        _blank(ax); ax.set_visible(False)
    for i, r in lenses.iterrows():
        rr, cc = divmod(i, 3)
        p = int(r.pos)
        panels = (("HSC 27″ context", big[int(r.h5_row)][:4], "asinh_ctx" if stretch != "linear" else "linear"),
                  ("HSC 48 px (query)", stamps.hsc(meta["parent_row"][p]), stretch),
                  ("LSST 48 px (own)", stamps.lsst(meta["lsst_row"][p]), stretch))
        for j, (lab, img, st) in enumerate(panels):
            ax = axes[rr, 3 * cc + j]; ax.set_visible(True); _show(ax, img, st, lab)
            if j == 0:
                ax.set_title(f"{r.L}  {r.object_id_hsc}   own-image rank {r[f'rank_{metric}']}", fontsize=6.5, loc="left")
            if j == 2:
                _frame(ax, "red", 1.5)
    fig.suptitle("The 13 known HSC lenses in COSMOS: wide HSC cutout (context only) | HSC model input (query) | "
                 "LSST model input", fontsize=10, y=0.975)
    pdf.savefig(fig); plt.close(fig)


def page_lens(pdf, plt, r, order, d, tsel, big, stamps, meta, lens_pos, lens_L, k, stretch, metric, ctx=None):
    """One page per lens: query row, then k neighbours as (LSST retrieved | HSC same object) pairs."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(11, 8.5))
    ncol = 5
    nrow = int(np.ceil(k / ncol))
    outer = GridSpec(nrow + 1, ncol, figure=fig, height_ratios=[1.25] + [1] * nrow,
                     hspace=0.42, wspace=0.12, left=0.02, right=0.98, top=0.905, bottom=0.03)
    p = int(r.pos)
    q_panels = (("HSC 27″ context (not model input)", big[int(r.h5_row)][:4], "asinh_ctx" if stretch != "linear" else "linear"),
                ("HSC 48 px = QUERY", stamps.hsc(meta["parent_row"][p]), stretch),
                ("LSST 48 px (own image)", stamps.lsst(meta["lsst_row"][p]), stretch))
    for j, (lab, img, st) in enumerate(q_panels):
        ax = fig.add_subplot(outer[0, j]); _show(ax, img, st)
        ax.set_title(lab, fontsize=8)
        _frame(ax, {0: "#888888", 1: "#1f77b4", 2: "red"}[j], 2.5)
    ax = fig.add_subplot(outer[0, 3:]); ax.axis("off")
    ax.text(0.03, 0.97, f"{r.L}   HSC {r.object_id_hsc}\nRA {r.ra:.5f}  Dec {r.dec:.5f}\n"
                        f"LSST visits g/r/i/z = {r.nin}\n"
                        f"own LSST image: rank {r[f'rank_{metric}']} (d = {r.d_self:.2f}"
                        + (f", via {r.self_transform}" if r.get("self_transform", "id") not in ("id", "") else "") + ")\n"
                        f"  (16-d mean report: rank {r.rank_mean16})\n"
                        f"d of 1st / {k}th neighbour = {d[order[0]]:.2f} / {d[order[k - 1]]:.2f}\n"
                        f"other known lenses in top {k} / 100: {r.hits20} / {r.hits100}",
            fontsize=8.3, va="top", family="monospace", transform=ax.transAxes)
    for j, n in enumerate(order[:k]):
        rr, cc = divmod(j, ncol)
        n = int(n)
        is_self, is_lens = n == p, (n in lens_pos and n != p)
        inner = outer[rr + 1, cc].subgridspec(1, 2, wspace=0.03)
        a_l = fig.add_subplot(inner[0]); a_h = fig.add_subplot(inner[1])
        _show(a_l, stamps.lsst(meta["lsst_row"][n]), stretch, "LSST (retrieved)")
        if ctx is None:
            _show(a_h, stamps.hsc(meta["parent_row"][n]), stretch, "HSC")
        else:
            from matplotlib.patches import Rectangle
            _show(a_h, ctx[n][:4], "asinh_ctx" if stretch != "linear" else "linear", "HSC 27″ (box = 48 px)")
            o = (ctx[n].shape[-1] - CROP) / 2 - 0.5
            a_h.add_patch(Rectangle((o, o), CROP, CROP, fill=False, ec="w", lw=0.6, ls="--", alpha=0.85))
        tag = "  SELF" if is_self else (f"  LENS {lens_L[lens_pos.index(n)]}" if is_lens else "")
        tname = "" if tsel is None or tsel[n] == 0 else f"  [{T_NAMES[tsel[n]]}]"
        a_l.set_title(f"#{j + 1}  d={d[n]:.2f}  n_i={int(meta['n_input_i'][n])}{tname}{tag}", fontsize=6.3, loc="left",
                      color="red" if is_self else ("#b8860b" if is_lens else "k"),
                      weight="bold" if tag else "normal")
        if is_self:
            _frame(a_l, "red"); _frame(a_h, "red", 1.5)
        elif is_lens:
            _frame(a_l, "gold"); _frame(a_h, "gold", 1.5)
    fig.suptitle(f"{r.L}: HSC lens {r.object_id_hsc} → top {k} nearest LSST images in physics space "
                 f"({METRIC_SHORT[metric]}, {len(meta['lsst_row']):,} COSMOS objects)", fontsize=10.5, y=0.975)
    pdf.savefig(fig); plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cosine-cosmos-13-flat64-dihedral")
    ap.add_argument("--latents", default=str(LATENTS))
    ap.add_argument("--metric", default="flat64_dihedral", choices=["mean16", "flat64", "flat64_dihedral"])
    ap.add_argument("--stretch", default="asinh", choices=["asinh", "linear"])
    ap.add_argument("--pair-hsc", default="stamp", choices=["stamp", "context"],
                    help="right panel of each neighbour pair: the 48 px HSC stamp, or the 27″ (160 px) HSC cutout")
    ap.add_argument("--k-gallery", type=int, default=20)
    ap.add_argument("--k-table", type=int, default=100)
    ap.add_argument("--n-lenses", type=int, default=None, help="first N COSMOS lenses only (default all)")
    args = ap.parse_args()
    t0 = time.time()

    out = OUT_ROOT / args.tag
    if out.exists():
        raise SystemExit(f"{out} exists — refusing to overwrite (pick a new --tag)")

    lat, meta, attrs = load_latents(Path(args.latents))
    assert "cosine" in attrs["tag"] and "2026-09-27_4" in attrs["checkpoint"], attrs
    lenses, big, n_list, n_h5 = load_lenses(meta["object_id_hsc"])
    if args.n_lenses:
        lenses = lenses.iloc[: args.n_lenses].reset_index(drop=True)
    lenses.insert(0, "L", [f"L{i + 1}" for i in range(len(lenses))])
    lens_pos = [int(p) for p in lenses["pos"]]
    lens_L = list(lenses["L"])
    n_db = len(meta["object_id_hsc"])
    print(f"[data] {n_db:,} COSMOS anchors; {len(lenses)} lenses as queries  ({time.time() - t0:.1f}s)", flush=True)

    # store bridges for the lenses (lsst_meta row id + HSC parent row id)
    lmeta = pd.read_parquet(LSST_META, columns=["object_id_hsc", "field"])
    hcat = pd.read_parquet(ROOT / "data/efficient_neighs_48/catalog.parquet", columns=["object_id_hsc"])
    for p in lens_pos:
        assert str(lmeta["object_id_hsc"].iloc[int(meta["lsst_row"][p])]) == meta["object_id_hsc"][p]
        assert lmeta["field"].iloc[int(meta["lsst_row"][p])] == "COSMOS"
        assert str(hcat["object_id_hsc"].iloc[int(meta["parent_row"][p])]) == meta["object_id_hsc"][p]

    stamps = Stamps()
    db16, db64 = lat["phys/lsst_mean"], lat["phys/lsst_flat"]
    metrics = ["mean16", "flat64"]
    zdih = None
    if args.metric == "flat64_dihedral":
        metrics += ["flat64_dihedral", "flat64_tokperm"]
        zdih = encode_dihedral(attrs["checkpoint"], [stamps.hsc(meta["parent_row"][p]) for p in lens_pos])
        err = float(np.abs(zdih[:, 0] - lat["phys/hsc_flat"][lens_pos]).max())
        print(f"[encode] 8 rotations/flips × {len(lens_pos)} lenses on CPU; |identity − saved latent| max {err:.2e} "
              f"({time.time() - t0:.1f}s)", flush=True)
        assert err < 1e-3, "re-encoded identity latent disagrees with the saved latents"
    out.mkdir(parents=True)

    rows, nn_rows, ranked, diag_shift, diag_err, t_counts, hits = [], [], {}, [], [], np.zeros(8, int), []
    for i, r in lenses.iterrows():
        p = int(r.pos)
        dist = {"mean16": l2(lat["phys/hsc_mean"][p], db16), "flat64": l2(lat["phys/hsc_flat"][p], db64)}
        tsel = None
        if zdih is not None:
            D = l2(zdih[i], db64)                                      # (8, N)
            dist["flat64_dihedral"], tsel = D.min(0), D.argmin(0)
            zq = lat["phys/hsc_flat"][p].reshape(4, 16)
            zperm = np.stack([zq[token_perm(fl, k)].ravel() for fl, k in DIHEDRAL])
            dist["flat64_tokperm"] = l2(zperm, db64).min(0)
            mv = np.linalg.norm(zdih[i] - zdih[i, 0], axis=1)
            diag_shift.append(mv)
            diag_err.append(np.where(mv > 0, np.linalg.norm(zdih[i] - zperm, axis=1) / np.maximum(mv, 1e-12), np.nan))
        orders = {k: np.argsort(v, kind="stable") for k, v in dist.items()}
        order, d = orders[args.metric], dist[args.metric]
        ranked[i] = (order, d, tsel)
        others = [q for q in lens_pos if q != p]
        if tsel is not None:
            t_counts += np.bincount(tsel[order[: args.k_gallery]], minlength=8)
        row = {f"rank_{k}": self_rank(o, p) for k, o in orders.items()}
        row.update(d_self=float(d[p]), d20=float(d[order[args.k_gallery - 1]]),
                   d_self_flat64=float(dist["flat64"][p]),
                   d20_flat64=float(np.sort(dist["flat64"])[args.k_gallery - 1]),
                   nin="/".join(str(int(meta[f"n_input_{b}"][p])) for b in "griz"),
                   hits20=int(np.isin(order[: args.k_gallery], others).sum()),
                   hits100=int(np.isin(order[: args.k_table], others).sum()),
                   overlap_mean16=int(np.isin(order[: args.k_gallery], orders["mean16"][: args.k_gallery]).sum()),
                   self_transform=T_NAMES[tsel[p]] if tsel is not None else "")
        rows.append(row)
        for k, n in enumerate(order[: args.k_table]):
            n = int(n)
            if n in others:
                hits.append(dict(lens=r.L, hit=lens_L[lens_pos.index(n)], rank=k + 1))
            nn_rows.append(dict(lens=r.L, lens_object_id_hsc=r.object_id_hsc, rank=k + 1,
                                object_id_hsc=meta["object_id_hsc"][n], lsst_row=int(meta["lsst_row"][n]),
                                parent_row=int(meta["parent_row"][n]), dist=float(d[n]),
                                transform=T_NAMES[tsel[n]] if tsel is not None else "",
                                dist_mean16=float(dist["mean16"][n]), dist_flat64=float(dist["flat64"][n]),
                                is_self=n == p, is_known_lens=n in others,
                                n_input_i=float(meta["n_input_i"][n]), psf_fwhm_i=float(meta["psf_fwhm_i"][n]),
                                is_val=bool(meta["is_val"][n])))
    lenses = pd.concat([lenses, pd.DataFrame(rows)], axis=1)
    lenses.to_csv(out / "lenses.csv", index=False)
    pd.DataFrame(nn_rows).to_csv(out / "neighbours.csv", index=False)
    print(lenses[["L", "object_id_hsc"] + [f"rank_{k}" for k in metrics] + ["d_self", "d20", "hits20", "hits100",
                                                                           "overlap_mean16"]].to_string(), flush=True)
    print(f"[search] done; hits: {hits}  ({time.time() - t0:.1f}s)", flush=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    stretch_note = ("Display: 48 px stamps shown with an asinh stretch from background to the peak (q = 4) so galaxy cores keep "
                    "their shape and faint blue structure (arcs, companions) is lifted; the 27″ HSC context cutouts are raw "
                    "linear flux, shown with their own asinh stretch. Display only — the model sees the stamps as stored."
                    if args.stretch == "asinh" else
                    "Display: linear 1–99.5 percentile stretch per stamp (same as the anomaly galleries).")
    ctx = None
    if args.pair_hsc == "context":
        cc = ContextCutouts()
        need = sorted({int(n) for o, _, _ in ranked.values() for n in o[: args.k_gallery]})
        ctx = {p: cc.get(meta["object_id_hsc"][p]) for p in need}
        worst = check_context(ctx, stamps, meta)
        print(f"[context] {len(ctx)} 27″ cutouts; preprocessed centre vs stored 48 px stamp: max |diff| {worst:.2e} "
              f"({time.time() - t0:.0f}s)", flush=True)
        assert worst < 0.05, "a 27″ cutout does not reproduce its 48 px stamp — wrong object or offset"
    pdf_path = out / f"HSC_lenses_LSST_physics_NN_{args.tag}.pdf"
    diag = None
    if zdih is not None:
        diag = dict(shift=np.array(diag_shift), proxy_rel_err=np.array(diag_err), t_counts=t_counts)
    with PdfPages(pdf_path) as pdf:
        page_summary(pdf, plt, args, attrs, lenses, n_list, n_h5, n_db, metrics, hits, stretch_note)
        page_table(pdf, plt, args, lenses, metrics)
        if diag is not None:
            page_method(pdf, plt, args, lenses, diag, stamps, meta, args.stretch)
        page_overview(pdf, plt, lenses, big, stamps, meta, args.stretch, args.metric)
        for i, r in lenses.iterrows():
            order, d, tsel = ranked[i]
            page_lens(pdf, plt, r, order, d, tsel, big, stamps, meta, lens_pos, lens_L, args.k_gallery,
                      args.stretch, args.metric, ctx)
        print(f"[pdf] {len(lenses)} lens pages ({time.time() - t0:.0f}s)", flush=True)
    summ = dict(tag=args.tag, latents=args.latents, checkpoint=attrs["checkpoint"], n_db=n_db, n_lenses=len(lenses),
                metric=args.metric, metric_label=METRIC_LABEL[args.metric], stretch=args.stretch, pair_hsc=args.pair_hsc,
                own_rank={k: dict(rank1=int((lenses[f"rank_{k}"] == 1).sum()), top20=int((lenses[f"rank_{k}"] <= 20).sum()),
                                  median=float(lenses[f"rank_{k}"].median())) for k in metrics},
                known_lens_hits=hits, wall_s=round(time.time() - t0, 1))
    if diag is not None:
        summ["orientation"] = dict(median_shift=float(np.median(diag["shift"][:, 1:])),
                                   median_proxy_rel_err=float(np.nanmedian(diag["proxy_rel_err"][:, 1:])),
                                   top20_transform_counts=dict(zip(T_NAMES, map(int, t_counts))))
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    print(f"[done] {pdf_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
