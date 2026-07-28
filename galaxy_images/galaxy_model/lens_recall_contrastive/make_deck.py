#!/usr/bin/env python
"""Build a BLINDED scoring deck from an EXPLICIT arm list.

Difference from `scale_ablation/lens_recall/make_scoring_deck.py`: that script composes the deck
implicitly -- one physics row per `*.h5` found in the latents directory, plus named decoys. That is
right for a sweep, wrong here, where we want exactly four hand-picked rows drawn from two different
latents directories. So this takes `--arm LABEL:SPACE` and resolves LABEL against a search path.

The retrieval and rendering themselves are IMPORTED from that module (`topk`, `rgb`), so the
neighbour ranking, the self-match filtering, the crop and the percentile stretch are bit-identical
to the decks already scored. That is what makes scores comparable across the two sessions.

Blinding: rows are shuffled and relabelled A/B/C...; the mapping goes to `*_KEY.json`, which is NOT
posted. A blank scoresheet is emitted for the scorer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_gm = _here.parent
_root = _gm.parents[1]
_lr = _gm / "scale_ablation" / "lens_recall"
for p in (str(_lr), str(_gm), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from make_scoring_deck import LETTERS, QUERIES, rgb, topk  # noqa: E402


def _resolve(label: str, dirs: list[Path]) -> Path:
    for d in dirs:
        p = d / f"{label}.h5"
        if p.exists():
            return p
    raise SystemExit(f"latents for {label!r} not found in: {', '.join(str(d) for d in dirs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL:SPACE",
                    help="repeatable, ordered; SPACE is 'physics' or 'instrument'")
    ap.add_argument("--latents-dir", action="append", type=Path, default=[],
                    help="repeatable; searched in order")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--query", default="lens48",
                    help=f"named query ({', '.join(QUERIES)}) or use --query-row")
    ap.add_argument("--query-row", type=int, default=None,
                    help="explicit neighbours_v2.h5 row; overrides --query lookup")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--rows-per-page", type=int, default=4)
    ap.add_argument("--out-prefix", default=None, help="default: deck_<query>")
    ap.add_argument("--keep-counterpart", action="store_true",
                    help="keep the query's own cross-survey counterpart (paper-comparable)")
    ap.add_argument("--note", default="", help="free text recorded in the KEY for provenance")
    args = ap.parse_args()

    if args.query_row is not None:
        qrow = args.query_row
    elif args.query in QUERIES:
        qrow = QUERIES[args.query]
    else:
        raise SystemExit(f"unknown query {args.query!r}; pass --query-row")

    if not args.arm:
        raise SystemExit("need at least one --arm LABEL:SPACE")
    dirs = args.latents_dir or [_here / "latents"]

    entries = []
    for a in args.arm:
        label, space = a.rsplit(":", 1)
        if space not in ("physics", "instrument"):
            raise SystemExit(f"space must be physics|instrument, got {space!r} in {a!r}")
        gal = _resolve(label, dirs)
        nns = topk(gal, space, qrow, args.k, not args.keep_counterpart)
        entries.append((label, space, str(gal), nns))
        print(f"[deck] {label:<38s} {space:<10s} top-{args.k} from {Path(gal).parent.name}/",
              flush=True)

    rng = np.random.default_rng(args.seed + qrow)
    order = rng.permutation(len(entries))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"deck_{args.query}"

    key = {"query": args.query, "query_row": qrow, "seed": args.seed, "k": args.k,
           "counterpart_excluded": not args.keep_counterpart, "note": args.note, "rows": {}}
    for slot, ei in enumerate(order):
        label, space, gal, nns = entries[ei]
        key["rows"][LETTERS[slot]] = {
            "checkpoint": label, "space": space, "latents": gal,
            "neighbors": [{"rank": i + 1, "survey": s, "row": r, "dist": round(dd, 5)}
                          for i, (s, r, dd) in enumerate(nns)],
        }
    kp = args.out_dir / f"{prefix}_KEY.json"
    kp.write_text(json.dumps(key, indent=2))
    print(f"[deck] wrote {kp}  (PRIVATE -- do not post)")

    sheet = [f"SCORING SHEET -- query {args.query} (HSC row {qrow})", "",
             "For each row, list the rank numbers whose images look like plausible strong",
             "lenses (arcs, rings, close double/quad images around a bright early-type).",
             f"Write `none` if there are none. Ranks run 1..{args.k} left to right.", ""]
    sheet += [f"  {LETTERS[s]}: " for s in range(len(entries))]
    (args.out_dir / f"{prefix}_scoresheet.txt").write_text("\n".join(sheet) + "\n")

    # ---- render ---------------------------------------------------------------------
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
        suffix = f" -- page {pi}/{len(pages)}" if len(pages) > 1 else ""
        fig.suptitle(
            f"Lens-discovery scoring -- query {args.query} (HSC row {qrow}){suffix}\n"
            f"Rows are SHUFFLED and anonymised. For each row list the ranks that look like lenses.",
            fontsize=11)
        plt.tight_layout()
        p = args.out_dir / (f"{prefix}_page{pi}.png" if len(pages) > 1 else f"{prefix}.png")
        plt.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
        print(f"[deck] wrote {p}")
    print("PAGES:" + ",".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
