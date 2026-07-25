#!/usr/bin/env python3
"""Render FM-vs-DDPM comparison tables as a booktabs-style PDF + .tex (+ Discord payload).

Kinds:
  mse : reads the long-form recon-MSE grid CSV (model_label, eta, num_steps,
        anchor_survey, mse_mean); pivots to rows=steps, cols=FM + DDPM per-eta,
        cells "HSC / Legacy".
  r2  : reads two predict_*.csv files (FM and DDPM downstream probes); rows =
        (target, latent) pairs present in either; cols = FM, DDPM, Delta.

The PDF is drawn with matplotlib (no LaTeX needed); the .tex is a booktabs
tabular ready for the paper. --payload-out writes a Discord message JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def build_mse(csv_path: Path):
    rows = _read_csv(csv_path)
    cells: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    etas: set[str] = set()
    steps_set: set[int] = set()
    for r in rows:
        label = r["model_label"]
        if label not in ("fm-control", "ddpm-eps"):
            continue
        steps = int(float(r["num_steps"])) if r.get("num_steps") else 0
        steps_set.add(steps)
        eta = r.get("eta", "")
        if label == "ddpm-eps":
            etas.add(eta)
        col = "FM (Euler)" if label == "fm-control" else f"DDPM η={float(eta):g}"
        cells[(str(steps), col)][r["anchor_survey"]] = float(r["mse_mean"])

    col_names = ["FM (Euler)"] + [
        f"DDPM η={float(e):g}" for e in sorted(etas, key=lambda s: float(s))
    ]
    row_names = [str(s) for s in sorted(steps_set)]
    table = []
    for rn in row_names:
        line = []
        for cn in col_names:
            d = cells.get((rn, cn), {})
            line.append(
                f"{_fmt(d.get('hsc'))} / {_fmt(d.get('legacy'))}" if d else "—"
            )
        table.append(line)
    title = "Reconstruction MSE (HSC / Legacy), matched noise, 256 held-out galaxies"
    return ["Steps"] + col_names, row_names, table, title


def build_r2(fm_csv: Path, ddpm_csv: Path):
    def load(p):
        out = {}
        for r in _read_csv(p):
            try:
                out[(r["target"], r["latent_variant"])] = float(r["score"])
            except (KeyError, ValueError):
                continue
        return out

    fm, dd = load(fm_csv), load(ddpm_csv)
    keys = [
        k
        for k in sorted(set(fm) | set(dd), key=lambda k: (k[1], k[0]))
        if k[1] in ("combined_e1", "combined_e2")
    ]
    row_names, table = [], []
    for target, latent in keys:
        tag = "e1" if latent.endswith("e1") else "e2"
        row_names.append(f"{target} [{tag}]")
        f, d = fm.get((target, latent)), dd.get((target, latent))
        delta = (d - f) if (f is not None and d is not None) else None
        table.append([_fmt(f), _fmt(d), _fmt(delta) if delta is not None else "—"])
    title = "Downstream R² — FM control vs DDPM (n=5469, 90/10 MLP probes)"
    return ["Target [latent]", "FM", "DDPM", "Δ (DDPM−FM)"], row_names, table, title


def render_pdf(header, row_names, table, title, out_pdf: Path):
    n_rows = len(row_names)
    fig_h = max(1.6, 0.32 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(header)), fig_h))
    ax.axis("off")
    cell_text = [[rn] + row for rn, row in zip(row_names, table)]
    tab = ax.table(
        cellText=cell_text, colLabels=header, loc="center", cellLoc="center"
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.25)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("none")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
        if c == 0:
            cell.set_text_props(ha="left")
    ax.set_title(title, fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_tex(header, row_names, table, title, out_tex: Path):
    align = "l" + "c" * (len(header) - 1)
    lines = [
        "% " + title,
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(h.replace("η", "$\\eta$").replace("Δ", "$\\Delta$").replace("²", "$^2$") for h in header) + " \\\\",
        "\\midrule",
    ]
    for rn, row in zip(row_names, table):
        safe = [str(c).replace("_", "\\_").replace("—", "--") for c in [rn] + row]
        lines.append(" & ".join(safe) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_tex.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["mse", "r2"], required=True)
    p.add_argument("--csv", type=Path, help="mse: the recon grid CSV")
    p.add_argument("--fm-csv", type=Path, help="r2: FM predict CSV")
    p.add_argument("--ddpm-csv", type=Path, help="r2: DDPM predict CSV")
    p.add_argument("--out-stem", type=Path, required=True)
    p.add_argument("--payload-out", type=Path, default=None)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    if args.kind == "mse":
        header, row_names, table, title = build_mse(args.csv)
    else:
        header, row_names, table, title = build_r2(args.fm_csv, args.ddpm_csv)
    if args.tag:
        title = f"{title} — {args.tag}"

    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    render_pdf(header, row_names, table, title, args.out_stem.with_suffix(".pdf"))
    write_tex(header, row_names, table, title, args.out_stem.with_suffix(".tex"))
    print(f"[table] wrote {args.out_stem.with_suffix('.pdf')} and .tex")

    if args.payload_out:
        lines = [f"📋 **{title}**"]
        for rn, row in zip(row_names, table):
            lines.append(f"`{rn}`: " + " | ".join(f"{h}={v}" for h, v in zip(header[1:], row)))
        msg = "\n".join(lines)
        if len(msg) > 1900:
            msg = msg[:1850] + "\n…(truncated; see PDF)"
        msg += f"\n📄 {args.out_stem.with_suffix('.pdf')}"
        args.payload_out.write_text(json.dumps({"content": msg}))
        print(f"[table] wrote payload {args.payload_out}")


if __name__ == "__main__":
    main()
