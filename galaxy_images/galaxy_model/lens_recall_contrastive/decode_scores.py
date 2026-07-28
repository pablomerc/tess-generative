#!/usr/bin/env python
"""Decode blinded deck scores: row letters -> arms, and emit the results table.

Usage (scores as JSON, letters -> list of ranks judged lens-like; [] means none):

    python decode_scores.py --key decks/deck_lens48_contrastive_KEY.json \
        --scores '{"A":[1,3,4],"B":[],"C":[1,2,5,7],"D":[1,2,3,4,5,7,8]}'
    python decode_scores.py --key ... --scores-file scores.json

Reports precision@k per arm (the paper's "N of top 8" framing). Deliberately does NOT report
recall: that needs the lens catalog as a denominator, and none of the recorded Fig-5 neighbours are
in the 74-lens catalog, so a catalog-based recall would measure something different from the visual
judgement being decoded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Provenance of each arm on the one axis that matters for reading the table: did this model ever
# see the 74 catalog lenses (73 of which, including the lens48 query, sit in the DWNVAL holdout)?
EXPOSURE = {
    "contrastive-spatial-conv1x1": ("contrastive conv1x1", "saw lenses (unlabeled)"),
    "contrastive-spatial-conv1x1-DWNVAL": ("contrastive conv1x1", "zero-shot"),
    "contrastive-spatial-avg": ("contrastive avg-pool", "saw lenses (unlabeled)"),
    "contrastive-random-conv1x1": ("contrastive conv1x1 (rand nb)", "saw lenses (unlabeled)"),
    "contrastive-random-avg": ("contrastive avg-pool (rand nb)", "saw lenses (unlabeled)"),
    "contrastive-nohead-DWNVAL": ("contrastive conv1x1 (no head)", "zero-shot"),
    "base-87k": ("flow matching (paper)", "saw lenses (unlabeled)"),
    "base-5NB-DWNVAL": ("flow matching", "zero-shot"),
    "fm-control-75k": ("flow matching (FM control)", "saw lenses (unlabeled)"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--scores", default=None, help="JSON: {\"A\": [1,3], \"B\": []}")
    ap.add_argument("--scores-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the markdown table here")
    args = ap.parse_args()

    if not (args.scores or args.scores_file):
        raise SystemExit("pass --scores or --scores-file")
    raw = args.scores if args.scores else args.scores_file.read_text()
    scores = json.loads(raw)
    scores = {k.strip().upper(): ([] if v in (None, "none", "None") else list(v))
              for k, v in scores.items()}

    key = json.loads(args.key.read_text())
    rows, k = key["rows"], key.get("k", 8)

    unknown = sorted(set(scores) - set(rows))
    unscored = sorted(set(rows) - set(scores))
    if unknown:
        raise SystemExit(f"scored letters not in the key: {unknown}")

    recs = []
    for letter, meta in rows.items():
        if letter not in scores:
            continue
        hits = sorted(set(scores[letter]))
        bad = [h for h in hits if not (1 <= h <= k)]
        if bad:
            raise SystemExit(f"row {letter}: ranks {bad} outside 1..{k}")
        arm, space = meta["checkpoint"], meta["space"]
        fam, exposure = EXPOSURE.get(arm, (arm, "?"))
        recs.append({"letter": letter, "arm": arm, "space": space, "family": fam,
                     "exposure": exposure, "hits": hits, "n": len(hits),
                     "precision": len(hits) / k,
                     "best_rank": min(hits) if hits else None})

    recs.sort(key=lambda r: (-r["n"], r["arm"]))

    lines = [f"# Blinded lens-retrieval scores — query `{key['query']}` "
             f"(HSC row {key['query_row']})", ""]
    if key.get("note"):
        lines += [f"> {key['note']}", ""]
    lines += [f"Scored by eye, blind to arm. Rows shuffled with seed {key['seed']}. "
              f"k={k}; query's own cross-survey counterpart "
              f"{'excluded' if key.get('counterpart_excluded') else 'kept'}.", "",
              f"| row | arm | family | lens exposure | space | lens-like / {k} | ranks | best rank |",
              "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(f"| {r['letter']} | `{r['arm']}` | {r['family']} | {r['exposure']} | "
                     f"{r['space']} | **{r['n']}/{k}** | "
                     f"{','.join(map(str, r['hits'])) or '—'} | "
                     f"{r['best_rank'] if r['best_rank'] else '—'} |")

    # Head-to-head on the axis the deck was built to test.
    def grp(pred):
        v = [r["n"] for r in recs if r["space"] == "physics" and pred(r)]
        return (sum(v) / len(v), len(v)) if v else (None, 0)
    con, ncon = grp(lambda r: r["family"].startswith("contrastive"))
    fm, nfm = grp(lambda r: r["family"].startswith("flow"))
    if con is not None and fm is not None:
        lines += ["", f"**Physics-latent mean:** contrastive {con:.1f}/{k} (n={ncon}) · "
                      f"flow matching {fm:.1f}/{k} (n={nfm}) · "
                      f"difference {con - fm:+.1f}"]
        lines += ["", f"With n={ncon} and n={nfm} single-query rows this is a direction, not a "
                      f"measurement — no error bar is claimable from one query."]
    if unscored:
        lines += ["", f"⚠️ Not scored, so excluded: {', '.join(unscored)}."]

    out = "\n".join(lines) + "\n"
    print(out)
    if args.out:
        args.out.write_text(out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
