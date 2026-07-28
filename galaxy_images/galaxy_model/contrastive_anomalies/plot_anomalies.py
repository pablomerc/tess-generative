"""Stage 4 — figures: top-anomaly stamp grids, a joint physics-vs-instrument panel, and ROC curves.

Stamps are pulled from the ram48 store the encoder actually saw (48x48, bands
g/r/i/z; RGB uses the first three), read as a handful of individual memmap rows —
a few dozen 18 KB reads, so no Ceph traversal.

The stamp helpers (``to_rgb``, ``dedup_by_radec``) mirror
``anomaly_detection/engaging/plot_top12_hscdown.py`` so these figures are directly
comparable to the paper's anomaly figure; they are inlined rather than imported
because that module pulls in the 474k HSC-downstream memmap at import time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from anchor_data import DEFAULT_DATA_DIR  # noqa: E402
from evaluate import _ranks, load_scores  # noqa: E402

DEFAULT_OUT_DIR = "/orcd/pool/007/pablomer/outputs/contrastive_anomalies"
# The raw 160x160 store the paper figure renders from (the 48x48 ram48 store is
# built from it by build_ram48.py, so row indices are shared).
DEFAULT_FULL_DIR = "/orcd/pool/007/pablomer/efficient_neighs"
TOP_N, N_ROWS, N_COLS = 12, 4, 3
DEDUP_ARCSEC = 10.0
CANDIDATE_MULT = 500

# Hue encodes the *space* (categorical slots 1 and 2 of the validated reference
# palette: blue = physics, orange = instrument) so colour tracks the entity and
# never the rank. The survey view is a secondary encoding — a step within the same
# hue ramp plus a dash pattern — because three shades of one blue would not be
# separable by colour alone.
C_PHYSICS, C_INSTRUMENT, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
C_GRID, C_MUTED, C_INK = "#e3e2df", "#8a8984", "#0b0b0b"
C_LENS = "#008300"  # slot 6 green — the lens marker, distinct from both series

# rep -> (hex, dash). Steps are from the palette's blue/orange ramps.
REP_STYLE = {
    "physics_hsc":       (C_PHYSICS,   "solid"),
    "physics_legacy":    ("#184f95",   (0, (5, 2))),
    "physics_mean":      ("#6da7ec",   (0, (1.5, 1.5))),
    "instrument_hsc":    (C_INSTRUMENT, "solid"),
    "instrument_legacy": ("#a33c15",   (0, (5, 2))),
}

METHOD_TITLE = {
    "flow": "Normalizing flow (NSF)",
    "iforest": "Isolation Forest",
    "knn": "k-NN (k=5)",
    "mahal": "Mahalanobis",
}


def rep_color(rep: str) -> str:
    if rep in REP_STYLE:
        return REP_STYLE[rep][0]
    return C_PHYSICS if rep.startswith("physics") else (
        C_INSTRUMENT if rep.startswith("instrument") else C_THIRD)


def rep_dash(rep: str):
    return REP_STYLE.get(rep, (None, "solid"))[1]


def rep_survey(rep: str) -> str:
    return "legacy" if rep.endswith("_legacy") else "hsc"


def pretty(rep: str) -> str:
    space, _, view = rep.partition("_")
    return f"{space.capitalize()} ({ {'hsc': 'HSC', 'legacy': 'Legacy', 'mean': 'x-survey mean'}[view] })"


# ---------------------------------------------------------------------------
# Stamps
# ---------------------------------------------------------------------------

class StampReader:
    """Random-access row reader over the ram48 bins (a few dozen rows per figure).

    These are the *preprocessed* 48x48 tensors the encoder actually scored.
    """

    def __init__(self, data_dir=DEFAULT_DATA_DIR, crop=48, bands=4):
        data_dir = Path(data_dir)
        n = int(json.loads((data_dir / "meta.json").read_text())["n_records"])
        shape = (n, bands, crop, crop)
        self._m = {
            "hsc": np.memmap(data_dir / "hsc48.bin", dtype="float16", mode="r", shape=shape),
            "legacy": np.memmap(data_dir / "legacy48.bin", dtype="float16", mode="r", shape=shape),
        }

    def __call__(self, survey: str, row: int) -> np.ndarray:
        return np.asarray(self._m[survey][int(row)], dtype=np.float32)


class Full160StampReader:
    """Raw 160x160 stamps from the efficient_neighs store — the paper-figure view.

    HSC rows are 5-band (g,r,i,z,y) and Legacy rows 4-band (g,r,i,z); ``to_rgb``
    takes the first three either way, so both surveys render as g/r/i exactly as
    ``figures_for_paper/anomaly_detection_figure/make_figure.py`` does.

    Note these are *raw* cutouts, not the zoomed-and-normalised 48x48 tensors the
    detector scored — the anomalous object sits at the centre and the extra field
    is context. Pass ``crop`` to centre-crop if the object is too small to read.

    Reads individual rows straight off the 120 GB / 96 GB memmaps: ~250 KB per
    HSC row, a few dozen rows per figure. No traversal.
    """

    def __init__(self, data_dir=DEFAULT_FULL_DIR, crop: int | None = None):
        data_dir = Path(data_dir)
        meta = json.loads((data_dir / "meta.json").read_text())
        self._m = {}
        for survey, key in (("hsc", "hsc_image"), ("legacy", "legacy_image")):
            spec = meta[key]
            self._m[survey] = np.memmap(data_dir / spec["path"], dtype=spec["dtype"],
                                        mode="r", shape=tuple(spec["shape"]))
        self.crop = crop

    def __call__(self, survey: str, row: int) -> np.ndarray:
        img = np.asarray(self._m[survey][int(row)], dtype=np.float32)
        if self.crop:
            h = img.shape[-1]
            o = max((h - self.crop) // 2, 0)
            img = img[:, o:o + self.crop, o:o + self.crop]
        return img


class CachedStamps:
    """Dedupe + parallel-prefetch wrapper around a stamp reader.

    A cold random row on the 160x160 Ceph store costs ~4.4 s — the same latency
    cliff that made the 160x160 precompute pass unusable. Rendering every
    representation x method grid one row at a time would be ~25 min of pure
    waiting, and the same object appears in several grids.

    So: collect every (survey, row) the figures need, drop duplicates, and fetch
    them with a thread pool. The reads are latency-bound, not bandwidth-bound
    (~250 KB each), so concurrency converts the wait almost linearly while adding
    only a few MB of in-flight traffic.
    """

    def __init__(self, reader, workers: int = 16):
        self._reader = reader
        self._workers = workers
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def prefetch(self, pairs):
        from concurrent.futures import ThreadPoolExecutor
        import time

        todo = sorted({(s, int(r)) for s, r in pairs} - self._cache.keys())
        if not todo:
            return
        t0 = time.time()
        print(f"[plot] prefetching {len(todo)} distinct stamps with {self._workers} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=self._workers) as ex:
            for (s, r), img in zip(todo, ex.map(lambda p: self._reader(*p), todo)):
                self._cache[(s, r)] = img
        dt = time.time() - t0
        print(f"[plot] prefetched in {dt:.1f}s ({dt / len(todo) * 1000:.0f} ms/stamp effective)", flush=True)

    def __call__(self, survey: str, row: int) -> np.ndarray:
        key = (survey, int(row))
        if key not in self._cache:
            self._cache[key] = self._reader(survey, row)
        return self._cache[key]


def to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def dedup_by_radec(order, ra, dec, n_keep, min_sep_arcsec):
    """Keep the highest-ranked object within each `min_sep_arcsec` sky patch.

    Operates on positions into the anchor arrays (not catalog rows).
    """
    thr2 = (min_sep_arcsec / 3600.0) ** 2
    kept, kra, kdec, kcd = [], [], [], []
    for i in order:
        a, d = float(ra[i]), float(dec[i])
        if any(((a - r) * c) ** 2 + (d - e) ** 2 < thr2 for r, e, c in zip(kra, kdec, kcd)):
            continue
        kept.append(int(i))
        kra.append(a); kdec.append(d); kcd.append(np.cos(np.deg2rad(d)))
        if len(kept) >= n_keep:
            break
    return np.asarray(kept, dtype=np.int64)


def top_positions(scores, ra, dec, top_n, dedup_arcsec, candidate_mult=CANDIDATE_MULT):
    """Return (positions, corpus rank) of the top-N anomalies after RA/Dec dedup.

    Rank, not percentile: for a top-12 out of 103,741 every percentile rounds to
    "0.00%", which tells the reader nothing. The 1-based pre-dedup rank does — and
    where it runs ahead of the panel number, that gap is exactly how many
    same-neighbourhood duplicates the 10" dedup removed.
    """
    r = _ranks(scores)
    n_cand = min(len(scores), top_n * candidate_mult)
    cand = np.argsort(r, kind="stable")[:n_cand]
    kept = dedup_by_radec(cand, ra, dec, top_n, dedup_arcsec)
    return kept, r[kept] + 1


def draw_panel(ax, img, caption, is_lens):
    ax.imshow(to_rgb(img), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(is_lens)
        if is_lens:
            s.set_color(C_LENS); s.set_linewidth(2.0)
    ax.text(0.5, 0.035, caption, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color=C_INK,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.78, linewidth=0))
    if is_lens:
        ax.text(0.96, 0.94, "lens", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.18", facecolor=C_LENS, linewidth=0))


def render_grid(imgs, ranks, lens_flags, title, out_path):
    size, header = 1.4, 0.44
    fig = plt.figure(figsize=(N_COLS * size, N_ROWS * size + header), dpi=150)
    gs = fig.add_gridspec(N_ROWS, N_COLS, left=0.01, right=0.99,
                          top=1.0 - header / (N_ROWS * size + header), bottom=0.01,
                          hspace=0.04, wspace=0.04)
    for i in range(min(N_ROWS * N_COLS, len(imgs))):
        draw_panel(fig.add_subplot(gs[i // N_COLS, i % N_COLS]), imgs[i],
                   f"#{i + 1}  ·  rank {int(ranks[i]):,}", bool(lens_flags[i]))
    fig.suptitle(title, fontsize=11, color=C_INK, y=0.995)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def render_joint(groups, method, tag, out_path):
    n_g = len(groups)
    size, gap, header = 1.30, 0.34, 0.52
    fig_w = n_g * N_COLS * size + (n_g - 1) * gap
    fig_h = N_ROWS * size + header
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
    outer = fig.add_gridspec(1, n_g, left=0.005, right=0.995,
                             top=1.0 - (header * 0.72) / fig_h, bottom=0.005,
                             wspace=gap / (N_COLS * size))
    sep = fig.add_axes([0, 0, 1, 1], frameon=False)
    sep.set_xlim(0, 1); sep.set_ylim(0, 1); sep.set_xticks([]); sep.set_yticks([])
    sep.patch.set_alpha(0); sep.set_zorder(-1)
    for i in range(1, n_g):
        sep.plot([i / n_g, i / n_g], [0.02, 0.94], color=C_GRID, linestyle="--",
                 dashes=(6, 4), linewidth=1.0)

    for gi, g in enumerate(groups):
        inner = outer[0, gi].subgridspec(N_ROWS, N_COLS, hspace=0.04, wspace=0.04)
        n = min(N_ROWS * N_COLS, len(g["imgs"]))
        for k in range(n):
            draw_panel(fig.add_subplot(inner[k // N_COLS, k % N_COLS]), g["imgs"][k],
                       f"#{k + 1}  ·  rank {int(g['ranks'][k]):,}", bool(g["lens"][k]))
        first = fig.axes[-n].get_position()
        last_first_row = fig.axes[-n + min(N_COLS, n) - 1].get_position()
        fig.text((first.x0 + last_first_row.x1) / 2, first.y1 + 0.012, g["label"],
                 ha="center", va="bottom", fontsize=12, fontweight="bold", color=g["color"])
    fig.suptitle(f"Top-{N_ROWS * N_COLS} anomalies — {method} — {tag}",
                 fontsize=10, color=C_MUTED, y=0.998)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

def render_roc(scores, is_lens, reps, methods, tag, out_path):
    from sklearn.metrics import roc_auc_score, roc_curve

    n_m = len(methods)
    fig, axes = plt.subplots(1, n_m, figsize=(3.1 * n_m, 3.4), dpi=170, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1.0, ls=(0, (4, 4)), zorder=1)
        for rep in reps:
            key = f"{rep}/{method}"
            if key not in scores:
                continue
            s = scores[key]
            ok = np.isfinite(s)
            fpr, tpr, _ = roc_curve(is_lens[ok], s[ok])
            auc = roc_auc_score(is_lens[ok], s[ok])
            ax.plot(fpr, tpr, color=rep_color(rep), lw=2.0, zorder=3,
                    linestyle=rep_dash(rep), solid_capstyle="round",
                    label=f"{pretty(rep)} — {auc:.3f}")
        ax.set_title(METHOD_TITLE.get(method, method), fontsize=10.5, color=C_INK)
        ax.set_xlabel("False positive rate", fontsize=9, color=C_MUTED)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, color=C_GRID, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.legend(loc="lower right", fontsize=7.2, frameon=False, labelcolor=C_INK)
    axes[0].set_ylabel(f"Strong-lens recall  (n={int(is_lens.sum())})", fontsize=9, color=C_MUTED)
    fig.suptitle(f"Strong-lens retrieval by latent space — {tag}", fontsize=11.5, color=C_INK)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--stamp-store", choices=["ram48", "full160"], default="ram48",
                   help="ram48 = the preprocessed 48x48 tensors the detector scored; "
                        "full160 = the raw 160x160 cutouts, as in the paper figure")
    p.add_argument("--full-dir", default=DEFAULT_FULL_DIR)
    p.add_argument("--stamp-crop", type=int, default=None,
                   help="full160 only: centre-crop to this many pixels before rendering")
    p.add_argument("--stamp-workers", type=int, default=16,
                   help="Parallel stamp reads. Cold rows on the 160x160 Ceph store cost "
                        "~4.4 s each and are latency-bound, so this is the difference "
                        "between ~30 s and ~25 min.")
    p.add_argument("--out-dir", default=None, help="Default: <scores dir>/figures_<tag>")
    p.add_argument("--joint-reps", nargs="+", default=["physics_hsc", "instrument_hsc"])
    p.add_argument("--roc-reps", nargs="+", default=None, help="Default: all representations")
    p.add_argument("--methods", nargs="+", default=None, help="Default: all methods present")
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--dedup-arcsec", type=float, default=DEDUP_ARCSEC)
    p.add_argument("--skip-individual", action="store_true")
    args = p.parse_args()

    scores_path = Path(args.scores)
    meta, scores, attrs = load_scores(scores_path)
    tag = str(attrs.get("tag", scores_path.stem.replace("scores_", "")))
    out_dir = Path(args.out_dir) if args.out_dir else scores_path.parent / f"figures_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    row_idx, ra, dec = meta["row_idx"], meta["ra"], meta["dec"]
    is_lens = meta["is_lens"].astype(bool)
    reps = sorted({k.rsplit("/", 1)[0] for k in scores})
    methods = args.methods or sorted({k.rsplit("/", 1)[1] for k in scores})
    # Distinguish the two stamp views in the filename so rendering one never
    # silently overwrites the other.
    if args.stamp_store == "full160":
        stamps = Full160StampReader(args.full_dir, crop=args.stamp_crop)
        view = f"160" if not args.stamp_crop else f"{args.stamp_crop}c"
        sfx = f"{tag}_{view}"
    else:
        stamps = StampReader(args.data_dir)
        sfx = tag
    print(f"[plot] tag={tag}  stamps={args.stamp_store}  reps={reps}  methods={methods}  out={out_dir}")

    stamps = CachedStamps(stamps, workers=args.stamp_workers)

    # Resolve every grid's top-N first, so the whole figure set's stamps can be
    # deduped and fetched in one parallel pass. Ranking is pure numpy on arrays
    # already in memory, so doing it up front costs nothing.
    wanted = {}
    for method in methods:
        needed = set(reps if not args.skip_individual else []) | set(args.joint_reps)
        for rep in sorted(needed):
            key = f"{rep}/{method}"
            if key in scores:
                wanted[(rep, method)] = top_positions(
                    scores[key], ra, dec, args.top_n, args.dedup_arcsec)
    stamps.prefetch(
        (rep_survey(rep), row_idx[i]) for (rep, _), (pos, _) in wanted.items() for i in pos
    )

    def gather(rep, method):
        pos, rank = wanted[(rep, method)]
        survey = rep_survey(rep)
        imgs = np.stack([stamps(survey, row_idx[i]) for i in pos])
        return {"label": pretty(rep), "color": rep_color(rep), "imgs": imgs,
                "ranks": rank, "lens": is_lens[pos], "pos": pos}

    for method in methods:
        if not args.skip_individual:
            for rep in reps:
                if (rep, method) not in wanted:
                    continue
                g = gather(rep, method)
                render_grid(g["imgs"], g["ranks"], g["lens"],
                            f"{pretty(rep)} — {method}",
                            out_dir / f"top{args.top_n}_{rep}_{method}_{sfx}.png")
        joint = [gather(r, method) for r in args.joint_reps if (r, method) in wanted]
        if len(joint) >= 2:
            render_joint(joint, method, tag,
                         out_dir / f"joint_top{args.top_n}_{method}_{sfx}.png")

    if is_lens.sum() > 0 and args.stamp_store == "ram48":
        render_roc(scores, is_lens, args.roc_reps or reps, methods, tag,
                   out_dir / f"roc_lens_{tag}.png")


if __name__ == "__main__":
    main()
