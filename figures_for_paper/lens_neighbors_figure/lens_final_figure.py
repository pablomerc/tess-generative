"""
Standalone lens nearest-neighbour figure.

Two query lenses; for each, a row of "Physics" nearest neighbours from the
dual-encoder flow-matching model. The bundled cache stores the top-12 NNs per
lens (image + survey + object id), so any subset of `nn_ranks` from 1..12 can
be plotted without touching the original 100k+ embedding gallery, the model
checkpoint, or the giant `neighbours_v2.h5`.

Usage:
    python lens_final_figure.py
    python lens_final_figure.py --nn-ranks 33:2,3,4,5,6,7  48:2,3,4,5,6,7
    python lens_final_figure.py --extra-lens 33 --extra-lens 48 --n-nn 8
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

_here = Path(__file__).resolve().parent
CACHE_DIR = _here / "_cache"
OUT_DIR = _here

# Default selection — matches the original paper figure.
# `nn_ranks` are 1-indexed positions in the bundled top-12 NN list.
DEFAULT_LENSES = [
    {"user_idx": 33, "nn_ranks": [2, 3, 4, 7, 8, 9]},
    {"user_idx": 48, "nn_ranks": [2, 4, 5, 6, 7, 9]},
]

CACHE_TOP_K = 12  # number of NNs available per lens in the bundle


def load_lens(user_idx: int) -> dict:
    p = CACHE_DIR / f"lens_{user_idx}.npz"
    if not p.exists():
        avail = sorted(int(f.stem.split("_")[1]) for f in CACHE_DIR.glob("lens_*.npz"))
        raise FileNotFoundError(f"No cache for lens {user_idx}. Available: {avail}")
    with np.load(p, allow_pickle=False) as d:
        return {
            "user_idx":      int(d["user_idx"]),
            "h5_row":        int(d["h5_row"]),
            "obj_id":        str(d["obj_id"]),
            "query_img":     d["query_img"],          # (3, 64, 64) float32
            "nn_survey":     [str(s) for s in d["nn_survey"]],
            "nn_raw_h5_row": d["nn_raw_h5_row"].astype(int),
            "nn_obj_id":     [str(s) for s in d["nn_obj_id"]],
            "nn_imgs":       d["nn_imgs"],            # (12, 3, 64, 64) float32
        }


def array_to_rgb(arr_chw, percentile_clip: float = 99.5) -> np.ndarray:
    """(C,H,W) → (H,W,3) with per-channel percentile stretch + min/max norm."""
    rgb = arr_chw[:3].copy().transpose(1, 2, 0)  # (H,W,3)
    for i in range(3):
        lo = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        hi = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], lo, hi)
    for i in range(3):
        ch = rgb[:, :, i]; lo, hi = ch.min(), ch.max()
        rgb[:, :, i] = (ch - lo) / (hi - lo) if hi > lo else 0.0
    return rgb


def show_img(ax, arr_chw, text=None, text_color="black", text_fontsize=14):
    ax.imshow(array_to_rgb(arr_chw))
    if text:
        ax.text(
            0.5, 0.96, text,
            transform=ax.transAxes,
            fontsize=text_fontsize, fontweight="bold",
            color=text_color,
            va="top", ha="center",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.7, linewidth=0),
        )
    ax.set_axis_off()


def make_figure(lens_specs, out_path):
    """lens_specs: list of dicts with keys user_idx, nn_ranks (list of 1-indexed ranks)."""
    lens_data = []
    for spec in lens_specs:
        d = load_lens(spec["user_idx"])
        ranks = spec["nn_ranks"]
        for r in ranks:
            if not (1 <= r <= CACHE_TOP_K):
                raise ValueError(f"nn rank {r} out of bundled range 1..{CACHE_TOP_K}")
        d["selected"] = [(r, d["nn_survey"][r - 1], d["nn_raw_h5_row"][r - 1],
                          d["nn_obj_id"][r - 1], d["nn_imgs"][r - 1]) for r in ranks]
        lens_data.append(d)

    n_nn_cols = max(len(d["selected"]) for d in lens_data)
    n_cols    = 1 + n_nn_cols
    n_rows    = len(lens_data)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[None, :]
    for ax in axes.flat:
        ax.set_axis_off()

    def src_lbl(s): return "(HSC)" if s == "hsc" else "(Legacy)"

    for r, d in enumerate(lens_data):
        show_img(axes[r, 0], d["query_img"], text=f"HSC {d['obj_id']}", text_fontsize=10)
        for c, (rank, survey, _, _, img) in enumerate(d["selected"], start=1):
            show_img(axes[r, c], img,
                     text=f"NN #{rank - 1} {src_lbl(survey)}",
                     text_color="#B8860B" if survey == "legacy" else "black")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.1, hspace=0.05)
    fig.canvas.draw()

    row_colors = ["lightgray", "silver"]
    for r in range(n_rows):
        y0 = min(axes[r, c].get_position().y0 for c in range(n_cols)) - 0.01
        y1 = max(axes[r, c].get_position().y1 for c in range(n_cols)) + 0.01
        fig.patches.append(Rectangle((0, y0), 1, y1 - y0,
                                     transform=fig.transFigure,
                                     facecolor=row_colors[r % 2],
                                     edgecolor="none", zorder=-10))

    bq  = axes[0, 0].get_position()
    bn1 = axes[0, 1].get_position()
    sep_x = (bq.x1 + bn1.x0) / 2
    fig.add_artist(Line2D([sep_x, sep_x], [0.02, 0.90], transform=fig.transFigure,
                          color="black", linewidth=4, linestyle="--"))

    fig.text((bq.x0 + bq.x1) / 2, 0.94, "Query",
             ha="center", va="bottom", fontsize=20, fontweight="bold")
    x0n = axes[0, 1].get_position().x0
    x1n = axes[0, -1].get_position().x1
    fig.text((x0n + x1n) / 2, 0.94, "Physics NNs",
             ha="center", va="bottom", fontsize=20, fontweight="bold", color="#2E86AB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")


def parse_nn_ranks(spec: str):
    """'33:2,3,4,5,6,7' → {'user_idx': 33, 'nn_ranks': [2,3,4,5,6,7]}"""
    uid_s, ranks_s = spec.split(":")
    return {"user_idx": int(uid_s), "nn_ranks": [int(x) for x in ranks_s.split(",")]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nn-ranks", nargs="+", default=None,
                   help="Per-lens rank list, e.g. '33:2,3,4,5,6,7' '48:2,3,4,5,6,7'.")
    p.add_argument("--n-nn", type=int, default=None,
                   help="Use the first N (rank-1 excluded) for every lens — convenience flag.")
    p.add_argument("--extra-lens", type=int, action="append", default=None,
                   help="If --n-nn is set, use these lens user_idx values instead of defaults.")
    p.add_argument("--out", default=str(OUT_DIR / "lens_neighbors_final_figure.png"))
    args = p.parse_args()

    if args.nn_ranks:
        lens_specs = [parse_nn_ranks(s) for s in args.nn_ranks]
    elif args.n_nn is not None:
        # Skip rank 1 (often duplicate of query) by default; take next N.
        ranks = list(range(2, 2 + args.n_nn))
        if max(ranks) > CACHE_TOP_K:
            raise SystemExit(f"--n-nn={args.n_nn} would need rank {max(ranks)}, only {CACHE_TOP_K} cached")
        uids = args.extra_lens or [s["user_idx"] for s in DEFAULT_LENSES]
        lens_specs = [{"user_idx": u, "nn_ranks": ranks} for u in uids]
    else:
        lens_specs = DEFAULT_LENSES

    make_figure(lens_specs, Path(args.out))


if __name__ == "__main__":
    main()
