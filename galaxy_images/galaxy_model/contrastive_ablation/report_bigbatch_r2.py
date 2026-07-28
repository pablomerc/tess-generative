"""Post the arm-I (batch 640) downstream-R2 comparison to Discord.

Reads the RESULT_*.md files that compare_dwnval.py writes and turns them into a
readable Discord report, one message per evaluated checkpoint plus a verdict line.

Runs as a dependent SLURM job after the eval jobs (see report_bigbatch_r2.slurm), so
the report lands even if nobody is watching a terminal. Failure is reported too: a
missing RESULT file means its eval died, and saying so is the whole point.

  python report_bigbatch_r2.py \
      --arm "latest (step 100,000, 64.0M imgs)=RESULT_contrastive-...-DWNVAL.md" \
      --arm "best-val=RESULT_contrastive-...-bestval-DWNVAL.md" \
      --webhook "$DISCORD_WEBHOOK"

Column naming: compare_dwnval.py labels its two columns "reference" and "holdout"
because it was written for the DWNVAL contamination test. Here they mean arm A
(batch 64) and this arm (batch 640) — the report relabels them so nobody reads the
table as a contamination result.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

DISCORD_LIMIT = 2000


def post(webhook: str, content: str) -> None:
    if not webhook:
        print("[report] no webhook; message would have been:\n" + content)
        return
    # Discord hard-rejects >2000 chars, so split on line boundaries rather than lose the tail.
    chunks, cur = [], ""
    for line in content.splitlines(keepends=True):
        if len(cur) + len(line) > DISCORD_LIMIT - 8:
            chunks.append(cur)
            cur = ""
        cur += line
    if cur:
        chunks.append(cur)

    for i, chunk in enumerate(chunks):
        body = json.dumps({"content": chunk}).encode()
        req = urllib.request.Request(
            webhook, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=30).read()
            print(f"[report] posted chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        except Exception as e:                      # never fail the job over logging
            print(f"[report] WARNING: post failed: {e}")


def parse_md(path: Path) -> dict:
    """Pull the aggregate stats, headline cells and verdict out of a RESULT_*.md."""
    text = path.read_text()
    out = {"aggregate": {}, "headline": [], "verdict": "", "shared_cells": None}

    m = re.search(r"shared cells\s*:\s*(\d+)", text)
    if m:
        out["shared_cells"] = int(m.group(1))

    def rows(section: str):
        # Take the markdown table that follows a "## <section>" heading.
        m = re.search(rf"##\s+{re.escape(section)}.*?\n(.*?)(?=\n##|\Z)", text, re.S)
        if not m:
            return []
        found = []
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line.startswith("|") or set(line) <= set("|-: "):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and cells[0].lower() not in ("statistic", "cell"):
                found.append(cells)
        return found

    for cells in rows("Aggregate"):
        if len(cells) >= 2:
            out["aggregate"][cells[0]] = cells[1]
    for cells in rows("Headline cells"):
        if len(cells) >= 4:
            out["headline"].append(cells[:4])

    m = re.search(r"##\s+Verdict\s*\n+(.*?)(?=\n##|\Z)", text, re.S)
    if m:
        out["verdict"] = " ".join(m.group(1).split())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True,
                   help="'<label>=<path to RESULT_*.md>', repeatable.")
    p.add_argument("--reference-label",
                   default="arm A = contrastive-spatial-conv1x1-DWNVAL (batch 64, best-val step 70k)")
    p.add_argument("--noise-floor", default="paired 5-fold std ~0.020, so |dR2| < ~0.04 is not resolvable")
    p.add_argument("--webhook", default="")
    args = p.parse_args()

    header = ("**Arm I downstream R2 — contrastive base conv1x1 at batch 640 (10x negatives)**\n"
              f"reference: {args.reference_label}\n"
              f"noise floor: {args.noise_floor}\n"
              "columns below: `ref` = batch 64, `new` = batch 640\n")
    post(args.webhook, header)

    summary = []
    for spec in args.arm:
        label, _, path_s = spec.partition("=")
        path = Path(path_s)
        if not path.exists():
            msg = (f"__**{label}**__\n"
                   f"XX **eval produced no result file** (`{path.name}`) — its job most likely "
                   f"failed. Check `contrastive_ablation/logs/contr-abl-eval_*.err`.")
            post(args.webhook, msg)
            summary.append(f"{label}: FAILED (no RESULT file)")
            continue

        d = parse_md(path)
        agg = d["aggregate"]
        lines = [f"__**{label}**__"]
        if agg:
            bits = [f"{k} {v}" for k, v in agg.items()]
            lines.append("aggregate dR2 (new - ref): " + " | ".join(bits))
        if d["shared_cells"]:
            lines.append(f"shared cells: {d['shared_cells']}")
        if d["headline"]:
            lines.append("```")
            lines.append(f"{'cell':38s} {'ref':>7s} {'new':>7s} {'delta':>7s}")
            for cell, ref, new, delta in d["headline"]:
                lines.append(f"{cell[:38]:38s} {ref:>7s} {new:>7s} {delta:>7s}")
            lines.append("```")
        if d["verdict"]:
            # compare_dwnval.py's verdict is phrased for the contamination test it was
            # written for ("contamination affects..."); flag that before quoting it.
            lines.append("_auto-verdict — compare_dwnval.py's wording is about "
                         "contamination; here the same delta means batch 64 vs 640:_ "
                         f"{d['verdict'][:600]}")
        post(args.webhook, "\n".join(lines))

        mean = agg.get("mean", "?")
        summary.append(f"{label}: mean dR2 {mean}")

    post(args.webhook,
         "**Read it this way:** a near-zero delta means 10x more InfoNCE negatives does not "
         "move the downstream R2 pattern, which is the result that makes every "
         "contrastive-loses cell robust to the 'you starved it of negatives' objection. "
         "A large positive delta on physics cells would mean the published contrastive "
         "baseline was genuinely negative-starved.\n"
         + " · ".join(summary))


if __name__ == "__main__":
    main()
