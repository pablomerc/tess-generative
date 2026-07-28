"""Stage 3 — quantify the anomaly scores and compare the physics vs instrument spaces.

Two things get measured.

**A. Strong-lens retrieval.** 74 catalogued strong lenses sit inside the 103,741
anchors, so "does this space rank rare morphology highly?" has a number, not just
a stamp grid: ROC-AUC, average precision, recall@k and the lenses' median
percentile rank, each against its chance baseline.

  Caveat to carry into any writeup: the contrastive encoder was trained on this
  corpus with no lens holdout, and 74 positives is a small sample (a single lens
  moving in or out of the top-100 shifts recall@100 by 1.4 points). Read AUC and
  AP as the stable numbers and recall@k as indicative.

**B. Space disagreement.** Spearman correlation and top-k Jaccard overlap between
every pair of (representation, method). If the physics and instrument spaces are
doing genuinely different work, their top-anomaly sets should barely intersect —
that is the claim this experiment exists to check.

Outputs ``lens_metrics_<tag>.csv``, ``space_agreement_<tag>.csv`` and a rendered
``summary_<tag>.md``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_OUT_DIR = "/orcd/pool/007/pablomer/outputs/contrastive_anomalies"
RECALL_KS = (100, 500, 1000, 5000)


def load_scores(path: Path):
    with h5py.File(path, "r") as f:
        meta = {k: f[k][:] for k in ("row_idx", "ra", "dec", "is_lens", "in_train")}
        keys = [k for k in _walk(f) if "/" in k]
        scores = {k: _get(f, k)[:] for k in keys}
        attrs = {k: f.attrs[k] for k in f.attrs}
    return meta, scores, attrs


def _walk(group, prefix=""):
    for k, v in group.items():
        path = f"{prefix}{k}"
        if isinstance(v, h5py.Group):
            yield from _walk(v, prefix=f"{path}/")
        else:
            yield path


def _get(f, key):
    node = f
    for part in key.split("/"):
        node = node[part]
    return node


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Descending competition-free rank: 0 = most anomalous. NaNs sort last."""
    s = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.int64)
    r[order] = np.arange(len(s))
    return r


def lens_metrics(scores: np.ndarray, is_lens: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    finite = np.isfinite(scores)
    if finite.sum() == 0 or is_lens.sum() == 0:
        return {}
    s, y = scores[finite], is_lens[finite]
    n, n_pos = len(s), int(y.sum())
    r = _ranks(s)
    out = {
        "n": n,
        "n_lens": n_pos,
        "roc_auc": float(roc_auc_score(y, s)),
        "avg_precision": float(average_precision_score(y, s)),
        "ap_chance": n_pos / n,
        "lens_pct_rank_median": float(np.median(r[y]) / n * 100.0),
        "lens_pct_rank_best": float(r[y].min() / n * 100.0),
    }
    for k in RECALL_KS:
        if k >= n:
            continue
        hit = int((r[y] < k).sum())
        out[f"recall@{k}"] = hit / n_pos
        out[f"lift@{k}"] = (hit / n_pos) / (k / n) if hit else 0.0
    return out


def space_agreement(scores: dict, top_k: int = 1000) -> list[dict]:
    from scipy.stats import spearmanr

    keys = sorted(scores)
    rows = []
    tops = {}
    for k in keys:
        r = _ranks(scores[k])
        tops[k] = set(np.flatnonzero(r < top_k).tolist())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            fa, fb = np.isfinite(scores[a]), np.isfinite(scores[b])
            both = fa & fb
            rho = float(spearmanr(scores[a][both], scores[b][both]).statistic) if both.sum() > 2 else np.nan
            inter = len(tops[a] & tops[b])
            rows.append({
                "a": a, "b": b,
                "spearman": rho,
                f"top{top_k}_overlap": inter,
                f"top{top_k}_jaccard": inter / len(tops[a] | tops[b]),
            })
    return rows


def _write_csv(path: Path, rows: list[dict]):
    import csv

    if not rows:
        return
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[eval] wrote {path}")


def _md_table(rows: list[dict], cols: list[str], fmt: dict) -> str:
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(fmt.get(c, lambda x: str(x))(v) if v != "" else "")
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--top-k", type=int, default=1000, help="Top-k set size for the overlap metric")
    args = p.parse_args()

    scores_path = Path(args.scores)
    meta, scores, attrs = load_scores(scores_path)
    tag = str(attrs.get("tag", scores_path.stem.replace("scores_", "")))
    is_lens = meta["is_lens"].astype(bool)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] {scores_path}  ({len(scores)} score arrays, {is_lens.sum()} lenses)")

    lens_rows = []
    for key in sorted(scores):
        rep, method = key.rsplit("/", 1)
        m = lens_metrics(scores[key], is_lens)
        if m:
            lens_rows.append({"representation": rep, "method": method, **m})
    _write_csv(out_dir / f"lens_metrics_{tag}.csv", lens_rows)

    agree_rows = space_agreement(scores, top_k=args.top_k)
    _write_csv(out_dir / f"space_agreement_{tag}.csv", agree_rows)

    # ---- markdown summary -------------------------------------------------
    n = int(attrs.get("n", len(is_lens)))
    n_lens = int(is_lens.sum())
    lines = [
        f"# Contrastive anomaly detection — `{tag}`",
        "",
        f"- checkpoint: `{attrs.get('checkpoint', '?')}`",
        f"- encoder pool: `{attrs.get('encoder_pool', '?')}`",
        f"- anchors: **{n:,}**, strong lenses: **{n_lens}**",
        f"- fit on {float(attrs.get('train_frac', 0.8)) * 100:.0f}% train split "
        f"(seed {attrs.get('seed', 42)}), scored on all anchors",
        f"- standardized: `{bool(attrs.get('standardized', True))}`, "
        f"kNN self-excluded: `{bool(attrs.get('knn_self_excluded', True))}`",
        "",
        "## A. Strong-lens retrieval",
        "",
        f"Chance AUC = 0.500, chance AP = {n_lens / max(n, 1):.5f}, "
        f"chance recall@1000 = {1000 / max(n, 1):.3f}.",
        "",
    ]
    cols = ["representation", "method", "roc_auc", "avg_precision",
            "recall@100", "recall@1000", "recall@5000", "lift@1000", "lens_pct_rank_median"]
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    fmt = {"roc_auc": f3, "avg_precision": f4, "recall@100": f3, "recall@1000": f3,
           "recall@5000": f3, "lift@1000": lambda v: f"{v:.1f}x",
           "lens_pct_rank_median": lambda v: f"{v:.1f}%"}
    lines += [_md_table(lens_rows, cols, fmt), ""]

    best = max(lens_rows, key=lambda r: r["roc_auc"]) if lens_rows else None
    if best:
        lines += [f"Best AUC: **{best['representation']} / {best['method']}** "
                  f"= {best['roc_auc']:.3f} (AP {best['avg_precision']:.4f}).", ""]

    lines += [
        "## B. Do the two spaces find the same objects?",
        "",
        f"Spearman rho over all anchors and Jaccard overlap of the top-{args.top_k} sets.",
        "Physics-vs-instrument pairs at the same method are the ones to read.",
        "",
    ]
    xspace = [r for r in agree_rows
              if r["a"].split("/")[-1] == r["b"].split("/")[-1]
              and r["a"].split("_")[0] != r["b"].split("_")[0]]
    acols = ["a", "b", "spearman", f"top{args.top_k}_overlap", f"top{args.top_k}_jaccard"]
    afmt = {"spearman": f3, f"top{args.top_k}_jaccard": f3}
    lines += [_md_table(xspace, acols, afmt), "",
              "_Full pairwise matrix in `space_agreement_" + tag + ".csv`._", ""]

    md_path = out_dir / f"summary_{tag}.md"
    md_path.write_text("\n".join(lines))
    print(f"[eval] wrote {md_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
