#!/usr/bin/env python
"""Reproduce the paper's Figure 5 (bottom) lens retrieval, and check it against the record.

Paper query: HSC 42692125246129290 = "Lens 48" in the neighbor_search_lenses numbering,
raw row 183713 of neighbours_v2.h5 (RA 352.9561, Dec +0.5864 — verified).

GROUND TRUTH. We do not have the AMD-only lens catalog on this cluster, so we cannot yet
label which retrieved galaxies *are* lenses. What we CAN do — and what actually validates
the pipeline — is check that the search returns the SAME neighbour rows at the SAME ranks
as the run that produced the figure. Those rows were recorded in
neighbor_search_lenses/check_objects/objects.json.

RANK CONVENTION (subtle, and it matters). lens_final_figure.py encoded the query from
lens_reconstruction_dataset.h5 while the gallery came from neighbours_v2.h5. Same galaxy,
slightly different preprocessing path, so the query's own gallery entry came back at a small
but nonzero distance and survived the `dist < 1e-6` self-match filter as all_nns[0]. Its
displayed labels were therefore "NN #i" == all_nns[i], starting the display at i=1.

Here the query embedding is taken straight FROM the gallery, so the self-match distance is
exactly 0 and is filtered cleanly. Our all_nns[i] should equal their all_nns[i+1], i.e.

    our 0-based index  ==  their displayed "NN #k"  with  k = index + 1

Expected (their displayed rank -> raw row), from objects.json:
    #1 legacy 183713 (the cross-survey counterpart)   #3 hsc 187190   #4 hsc 15375
    #5 hsc 181039    #6 hsc 40202                     #8 hsc 23615
(#2 and #7 were not recorded — only the six panels shown in the figure were.)
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

QUERY_ROW = 183713
QUERY_OID = "42692125246129290"
NEIGHBOURS_H5 = "/orcd/pool/007/pablomer/neighbours_v2.h5"

# displayed rank -> (survey, raw row), transcribed from check_objects/objects.json (lens 48)
EXPECTED = {1: ("legacy", 183713), 3: ("hsc", 187190), 4: ("hsc", 15375),
            5: ("hsc", 181039), 6: ("hsc", 40202), 8: ("hsc", 23615)}


def knn(gal_path: Path, space: str, k: int):
    with h5py.File(gal_path, "r") as f:
        idx = f["index_mmu"][:]
        hp = f[f"hsc_{space}"][:]
        lp = f[f"legacy_{space}"][:]
    pos = int(np.where(idx == QUERY_ROW)[0][0])
    q = hp[pos:pos + 1]
    combined = np.concatenate([hp, lp], 0)
    n_hsc = len(hp)
    d = np.linalg.norm(combined - q, axis=1)
    order = np.argsort(d, kind="stable")
    out = []
    for p in order:
        if d[p] < 1e-6:      # exact self-match (query is drawn from the gallery)
            continue
        survey = "hsc" if p < n_hsc else "legacy"
        out.append((survey, int(idx[p if p < n_hsc else p - n_hsc]), float(d[p])))
        if len(out) == k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    report = {"gallery": str(args.gallery), "label": args.label, "query_oid": QUERY_OID,
              "query_row": QUERY_ROW, "spaces": {}}

    for space in ("physics", "instrument"):
        nns = knn(args.gallery, space, args.k)
        hits, total = 0, 0
        lines = []
        for i, (sv, row, dist) in enumerate(nns):
            disp = i + 1
            exp = EXPECTED.get(disp)
            mark = ""
            if exp is not None:
                total += 1
                ok = (exp[0] == sv and exp[1] == row)
                hits += ok
                mark = "  <-- MATCH" if ok else f"  <-- MISMATCH (expected {exp[0]} {exp[1]})"
            lines.append(f"  #{disp:<3d} {sv:<7s} row {row:<8d} d={dist:.4f}{mark}")
        report["spaces"][space] = {
            "neighbors": [{"rank": i + 1, "survey": s, "row": r, "dist": d}
                          for i, (s, r, d) in enumerate(nns)],
            "recorded_ranks_matched": hits, "recorded_ranks_checked": total,
        }
        print(f"\n=== {space.upper()} space (query {QUERY_OID}, row {QUERY_ROW}) ===")
        print("\n".join(lines))
        if space == "physics":
            print(f"\n  reproduction: {hits}/{total} recorded ranks match exactly")

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    jp = args.out_prefix.with_suffix(".json")
    jp.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {jp}")

    # ---- figure: query + top-8 in each space -------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from galaxy_images.image_preprocessing import CenterCrop
    crop = CenterCrop(crop_size=64)

    def rgb(row, survey):
        with h5py.File(NEIGHBOURS_H5, "r") as f:
            key = "images_hsc" if survey == "hsc" else "images_legacy"
            t = torch.from_numpy(f[key][row][:4].astype(np.float32))
        t = crop(t.unsqueeze(0)).squeeze(0)[:3].numpy().transpose(1, 2, 0)
        for c in range(3):
            lo, hi = np.percentile(t[:, :, c], 0.5), np.percentile(t[:, :, c], 99.5)
            t[:, :, c] = np.clip((np.clip(t[:, :, c], lo, hi) - lo) / max(hi - lo, 1e-8), 0, 1)
        return t

    n_show = 8
    fig, axes = plt.subplots(2, n_show + 1, figsize=((n_show + 1) * 1.9, 4.4))
    for ax in axes.flat:
        ax.set_axis_off()
    for r, space in enumerate(("physics", "instrument")):
        axes[r, 0].imshow(rgb(QUERY_ROW, "hsc"))
        axes[r, 0].set_title(f"QUERY (HSC)\n{QUERY_OID}", fontsize=7, fontweight="bold")
        axes[r, 0].set_axis_off()
        for j, nb in enumerate(report["spaces"][space]["neighbors"][:n_show]):
            ax = axes[r, j + 1]
            ax.imshow(rgb(nb["row"], nb["survey"]))
            exp = EXPECTED.get(nb["rank"])
            ok = exp is not None and exp[0] == nb["survey"] and exp[1] == nb["row"]
            colr = "green" if ok else ("red" if exp is not None else "black")
            ax.set_title(f"#{nb['rank']} ({nb['survey'][:3]})\nrow {nb['row']}",
                         fontsize=7, color=colr, fontweight="bold" if ok else "normal")
            ax.set_axis_off()
        axes[r, 0].text(-0.35, 0.5, space.capitalize(), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    ph = report["spaces"]["physics"]
    fig.suptitle(
        f"Fig. 5 reproduction — {args.label}\n"
        f"physics space reproduces {ph['recorded_ranks_matched']}/{ph['recorded_ranks_checked']} "
        f"recorded ranks (green); instrument space = disentanglement control",
        fontsize=10)
    plt.tight_layout()
    png = args.out_prefix.with_suffix(".png")
    plt.savefig(png, dpi=140, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
