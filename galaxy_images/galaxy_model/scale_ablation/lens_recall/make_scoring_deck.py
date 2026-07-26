#!/usr/bin/env python
"""Build BLINDED visual-scoring decks for the lens-discovery metric (Metric 1).

WHY BLINDED. The paper's Fig-5 claim is a *discovery* claim: the retrieved neighbours are
lens-LIKE objects that are not in the known catalog (verified — 0 of the recorded Fig-5
neighbours appear in the 74-lens catalog). So the ground truth is a human visual judgement,
which makes the human the instrument. If the scorer can see which row came from the 1k model
they will unconsciously find fewer lenses there, and that bias becomes the result. Rows are
therefore shuffled and labelled A, B, C...; the mapping is written to a separate file that
the scorer does not open until after scoring.

DECOYS. Two rows per deck are drawn from the INSTRUMENT latent instead of the physics
latent. Physics space should surface lens-like morphology; instrument space should not. If
the scorer marks the instrument rows near zero without knowing which they are, that is
independent evidence the scoring is discriminative rather than noise -- and it doubles as
the disentanglement control.

Outputs per query:
  deck_<query>_page{1,2}.png   what gets posted to Discord
  deck_<query>_KEY.json        row-letter -> (checkpoint, space). DO NOT POST.
  deck_<query>_scoresheet.txt  blank sheet for the scorer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

_here = Path(__file__).resolve().parent
_gm = _here.parents[1]
_root = _gm.parents[1]
for p in (str(_gm), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

NEIGHBOURS_H5 = "/orcd/pool/007/pablomer/neighbours_v2.h5"
QUERIES = {"lens48": 183713, "lens33": 166616}   # verified rows in neighbours_v2.h5
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"


def topk(gal: Path, space: str, qrow: int, k: int, drop_counterpart: bool):
    """Top-k neighbours of qrow in `space`, excluding the query's own two gallery entries."""
    with h5py.File(gal, "r") as f:
        idx = f["index_mmu"][:]
        hp = f[f"hsc_{space}"][:]
        lp = f[f"legacy_{space}"][:]
    pos = int(np.where(idx == qrow)[0][0])
    q = hp[pos:pos + 1]
    comb = np.concatenate([hp, lp], 0)
    n_hsc = len(hp)
    d = np.linalg.norm(comb - q, axis=1)
    out = []
    for p in np.argsort(d, kind="stable"):
        survey = "hsc" if p < n_hsc else "legacy"
        row = int(idx[p if p < n_hsc else p - n_hsc])
        if row == qrow and (drop_counterpart or survey == "hsc"):
            continue   # own HSC entry always; own Legacy counterpart if requested
        out.append((survey, row, float(d[p])))
        if len(out) == k:
            break
    return out


_img_cache: dict = {}


def rgb(row: int, survey: str):
    key = (row, survey)
    if key in _img_cache:
        return _img_cache[key]
    import torch
    from galaxy_images.image_preprocessing import CenterCrop
    with h5py.File(NEIGHBOURS_H5, "r") as f:
        k = "images_hsc" if survey == "hsc" else "images_legacy"
        t = torch.from_numpy(f[k][row][:4].astype(np.float32))
    t = CenterCrop(crop_size=64)(t.unsqueeze(0)).squeeze(0)[:3].numpy().transpose(1, 2, 0)
    for c in range(3):
        lo, hi = np.percentile(t[:, :, c], 0.5), np.percentile(t[:, :, c], 99.5)
        t[:, :, c] = np.clip((np.clip(t[:, :, c], lo, hi) - lo) / max(hi - lo, 1e-8), 0, 1)
    _img_cache[key] = t
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--query", choices=list(QUERIES), required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--rows-per-page", type=int, default=6)
    ap.add_argument("--decoys", default="base-87k,scale-31622-75k",
                    help="checkpoints to also render in INSTRUMENT space as controls")
    ap.add_argument("--keep-counterpart", action="store_true",
                    help="keep the query's own cross-survey counterpart (paper-comparable)")
    args = ap.parse_args()

    qrow = QUERIES[args.query]
    gals = sorted(args.latents_dir.glob("*.h5"))
    if not gals:
        raise SystemExit(f"no galleries in {args.latents_dir}")
    decoys = [d.strip() for d in args.decoys.split(",") if d.strip()]

    entries = []   # (checkpoint_label, space, neighbours)
    for g in gals:
        entries.append((g.stem, "physics",
                        topk(g, "physics", qrow, args.k, not args.keep_counterpart)))
    for d in decoys:
        g = args.latents_dir / f"{d}.h5"
        if g.exists():
            entries.append((d, "instrument",
                            topk(g, "instrument", qrow, args.k, not args.keep_counterpart)))

    rng = np.random.default_rng(args.seed + qrow)
    order = rng.permutation(len(entries))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    key = {"query": args.query, "query_row": qrow, "seed": args.seed,
           "counterpart_excluded": not args.keep_counterpart, "rows": {}}
    for slot, ei in enumerate(order):
        label, space, nns = entries[ei]
        key["rows"][LETTERS[slot]] = {
            "checkpoint": label, "space": space,
            "neighbors": [{"rank": i + 1, "survey": s, "row": r, "dist": round(dd, 5)}
                          for i, (s, r, dd) in enumerate(nns)],
        }
    kp = args.out_dir / f"deck_{args.query}_KEY.json"
    kp.write_text(json.dumps(key, indent=2))
    print(f"wrote {kp}  (PRIVATE — do not post)")

    sheet = [f"SCORING SHEET — query {args.query} (HSC row {qrow})", "",
             "For each row, list the rank numbers whose images look like plausible strong",
             "lenses (arcs, rings, close double/quad images around a bright early-type).",
             "Write `none` if there are none. Ranks run 1..%d left to right." % args.k, ""]
    for slot in range(len(entries)):
        sheet.append(f"  {LETTERS[slot]}: ")
    (args.out_dir / f"deck_{args.query}_scoresheet.txt").write_text("\n".join(sheet) + "\n")

    # ---- render -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(entries)
    pages = [(s, min(s + args.rows_per_page, n_rows))
             for s in range(0, n_rows, args.rows_per_page)]
    paths = []
    for pi, (lo, hi) in enumerate(pages, 1):
        nr = hi - lo
        fig, axes = plt.subplots(nr, args.k + 1, figsize=((args.k + 1) * 1.75, nr * 2.0))
        if nr == 1:
            axes = axes[None, :]
        for ax in axes.flat:
            ax.set_axis_off()
        for r, slot in enumerate(range(lo, hi)):
            letter = LETTERS[slot]
            nns = key["rows"][letter]["neighbors"]
            axes[r, 0].imshow(rgb(qrow, "hsc"))
            axes[r, 0].set_title("QUERY", fontsize=8, fontweight="bold")
            axes[r, 0].text(-0.22, 0.5, letter, transform=axes[r, 0].transAxes,
                            fontsize=22, fontweight="bold", va="center", ha="center")
            for j, nb in enumerate(nns):
                ax = axes[r, j + 1]
                ax.imshow(rgb(nb["row"], nb["survey"]))
                ax.set_title(f"{nb['rank']}", fontsize=9, fontweight="bold")
        fig.suptitle(
            f"Lens-discovery scoring — query {args.query} (HSC row {qrow}) — page {pi}/{len(pages)}\n"
            f"Rows are SHUFFLED and anonymised. For each row list the ranks that look like lenses.",
            fontsize=11)
        plt.tight_layout()
        p = args.out_dir / f"deck_{args.query}_page{pi}.png"
        plt.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
        print(f"wrote {p}")
    print("PAGES:" + ",".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
