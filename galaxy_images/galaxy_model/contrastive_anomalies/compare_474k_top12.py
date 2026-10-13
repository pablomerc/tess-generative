"""Top-12 anomaly comparison across encoders on the full 474,954 HSC-downstream pool.

Consumes the arm scores written by refit_flows_474k.py (every arm fit through one
NSF code path, standardized and unstandardized), and for each variant:

  * resolves each arm's top-12 after the 10" RA/Dec dedup used by the paper figure
  * renders a paper-column-order joint panel plus per-model pairs
  * cross-references every panel against every other arm's panel, and writes the
    "#3 of contrastive is the #7 instrument anomaly of base" sentences

With ``--pool 367k`` (the default) every arm is ranked on exactly the pool the
published anomaly figure used: ``source_type in {0,1}`` of neighbours_v2.h5,
366,706 galaxies, which is a strict subset of the 474k hsc_downstream store.

Panel indices refer to *these* figures. They are directly comparable to published
Figure 3 in pool and column order, but not guaranteed identical to it: the paper's
physics column scored ``ours/hsc_mean`` (16D) and its AION column ``hsc_mean_pca64``,
whereas these arms use ``hsc_flat`` (64D) and ``hsc_mean_full`` (768D), and the
figure's ``_cache/`` with its 12 row indices is not in the repo.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLOWS = Path("/orcd/pool/007/pablomer/outputs/contrastive_anomalies/flows474k")
HSCDOWN = Path("/home/pablomer/orcd/scratch/hsc_downstream")
OUT_DIR = Path("/orcd/pool/007/pablomer/outputs/contrastive_anomalies/compare474k")

TOP_N, N_ROWS, N_COLS = 12, 4, 3
DEDUP_ARCSEC, CANDIDATE_MULT = 10.0, 500
MATCH_TOL = 5.0

C_PHYS, C_INST, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
C_GRID, C_MUTED, C_INK = "#e3e2df", "#8a8984", "#0b0b0b"

# Display order: the paper's three columns first, then the extra encoders.
ARM_META = {
    "base-physics":      ("Ours (Physics)", C_PHYS),
    "aion":              ("AION-1", C_THIRD),
    "base-instrument":   ("Ours (Instrument)", C_INST),
    "contra-physics":    ("Contrastive (Physics)", C_PHYS),
    "contra-instrument": ("Contrastive (Instrument)", C_INST),
    "fmctrl-physics":    ("FM-control (Physics)", C_PHYS),
    "fmctrl-instrument": ("FM-control (Instrument)", C_INST),
}
PAPER_ORDER = ["base-physics", "aion", "base-instrument"]
MODEL_PAIRS = {
    "base":   ("base-physics", "base-instrument"),
    "contra": ("contra-physics", "contra-instrument"),
    "fmctrl": ("fmctrl-physics", "fmctrl-instrument"),
}
MODEL_LABEL = {"base": "Base (paper ckpt)", "contra": "Contrastive (spatial-conv1x1)",
               "fmctrl": "FM control (diffusion ablation)"}


def dedup(order, ra, dec, n_keep, thr_arcsec=DEDUP_ARCSEC):
    thr2 = (thr_arcsec / 3600.0) ** 2
    kept, kra, kdec, kcd, ndrop = [], [], [], [], 0
    for i in order:
        a, d = float(ra[i]), float(dec[i])
        if any(((a - r) * c) ** 2 + (d - e) ** 2 < thr2 for r, e, c in zip(kra, kdec, kcd)):
            ndrop += 1
            continue
        kept.append(int(i)); kra.append(a); kdec.append(d); kcd.append(np.cos(np.deg2rad(d)))
        if len(kept) >= n_keep:
            break
    return np.asarray(kept, dtype=np.int64), ndrop


def sep_arcsec(ra1, dec1, ra2, dec2):
    d_ra = (np.asarray(ra1)[..., None] - np.asarray(ra2)) * np.cos(np.deg2rad(np.asarray(dec1)[..., None]))
    return np.hypot(d_ra, np.asarray(dec1)[..., None] - np.asarray(dec2)) * 3600.0


def to_rgb(img_chw, pct_lo=1, pct_hi=99):
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    return (np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def load_arm(arm, variant, pool, o_ra, o_dec):
    p = FLOWS / f"scores_{arm}_{variant}_{pool}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        s = f["flow"][:]
        raw_index = f["raw_index"][:]
    # rank over the score array, then map to catalog rows for dedup
    finite = np.isfinite(s)
    order = np.argsort(np.where(finite, s, -np.inf), kind="stable")[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    cand_pos = order[: TOP_N * CANDIDATE_MULT]
    cand_raw = raw_index[cand_pos]
    kept_raw, ndrop = dedup(cand_raw, o_ra, o_dec, TOP_N)
    pos_of_raw = {int(r): i for i, r in enumerate(raw_index)}
    return {
        "arm": arm, "raw": kept_raw, "ndrop": ndrop,
        "rank": np.array([ranks[pos_of_raw[int(r)]] for r in kept_raw]),
        "ra": o_ra[kept_raw], "dec": o_dec[kept_raw],
        "ranks_all": ranks, "pos_of_raw": pos_of_raw, "n": len(s),
    }


def render_joint(groups, imgs, title, out_path):
    n_g = len(groups)
    size, gap, header = 1.30, 0.34, 0.52
    fig_w = n_g * N_COLS * size + (n_g - 1) * gap
    fig_h = N_ROWS * size + header
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=170)
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
        label, color = ARM_META[g["arm"]]
        n = min(N_ROWS * N_COLS, len(g["raw"]))
        for k in range(n):
            ax = fig.add_subplot(inner[k // N_COLS, k % N_COLS])
            ax.imshow(to_rgb(imgs[int(g["raw"][k])]), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.text(0.5, 0.035, f"#{k+1}  ·  rank {int(g['rank'][k]):,}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=C_INK,
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.78, linewidth=0))
        first = fig.axes[-n].get_position()
        last = fig.axes[-n + min(N_COLS, n) - 1].get_position()
        fig.text((first.x0 + last.x1) / 2, first.y1 + 0.012, label, ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold", color=color)
    fig.suptitle(title, fontsize=10, color=C_MUTED, y=0.999)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def sentences(src_arm, arms, lines):
    """Emit '#i of SRC is the #j <space> anomaly of MODEL' for every panel of src."""
    src = arms[src_arm]
    src_label = ARM_META[src_arm][0]
    for model, (pa, pb) in MODEL_PAIRS.items():
        if pa == src_arm or pb == src_arm:
            continue
        if pa not in arms or pb not in arms:
            continue
        hits_p = sep_arcsec(src["ra"], src["dec"], arms[pa]["ra"], arms[pa]["dec"])
        hits_i = sep_arcsec(src["ra"], src["dec"], arms[pb]["ra"], arms[pb]["dec"])
        n_p = int((hits_p.min(axis=1) < MATCH_TOL).sum())
        n_i = int((hits_i.min(axis=1) < MATCH_TOL).sum())
        lines.append(f"\n**{src_label} → {MODEL_LABEL[model]}**  "
                     f"({n_p}/12 in its physics, {n_i}/12 in its instrument)")
        for k in range(len(src["raw"])):
            bits = []
            jp, ji = int(hits_p[k].argmin()), int(hits_i[k].argmin())
            if hits_p[k, jp] < MATCH_TOL:
                bits.append(f"the **#{jp+1} physics** anomaly")
            if hits_i[k, ji] < MATCH_TOL:
                bits.append(f"the **#{ji+1} instrument** anomaly")
            if bits:
                lines.append(f"- #{k+1} is {' and '.join(bits)} of {MODEL_LABEL[model]}.")
            else:
                # not in either top-12: report how far down it sits
                rp = arms[pa]["ranks_all"][arms[pa]["pos_of_raw"][int(src['raw'][k])]]
                ri = arms[pb]["ranks_all"][arms[pb]["pos_of_raw"][int(src['raw'][k])]]
                lines.append(f"- #{k+1} is in neither top-12 "
                             f"(ranks {int(rp):,} physics / {int(ri):,} instrument of "
                             f"{arms[pa]['n']:,}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["std", "raw"])
    ap.add_argument("--pool", default="367k", choices=["367k", "474k"])
    args = ap.parse_args()
    pool = args.pool
    pool_label = ("366,706 HSC galaxies — the paper-figure pool" if pool == "367k"
                  else "474,954 HSC galaxies — full hsc_downstream store")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cat = pq.read_table(HSCDOWN / "catalog.parquet", columns=["ra", "dec", "object_id"])
    o_ra, o_dec = cat["ra"].to_numpy(), cat["dec"].to_numpy()
    o_oid = cat["object_id"].to_numpy()

    meta = json.load(open(HSCDOWN / "meta.json"))
    flux = np.memmap(HSCDOWN / meta["pixels"]["files"]["flux"]["path"],
                     dtype=np.dtype(meta["pixels"]["files"]["flux"]["dtype"]),
                     mode="r", shape=tuple(meta["pixels"]["shape"]))

    for variant in args.variants:
        vlabel = "standardized" if variant == "std" else "unstandardized"
        arms = {}
        for arm in ARM_META:
            d = load_arm(arm, variant, pool, o_ra, o_dec)
            if d is not None:
                arms[arm] = d
        if not arms:
            print(f"[skip] no arms available for variant {variant}")
            continue
        print(f"\n=== variant {variant} ({vlabel}) — arms: {list(arms)} ===")

        need = sorted({int(r) for d in arms.values() for r in d["raw"]})
        print(f"[stamps] fetching {len(need)} rows...", flush=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            imgs = dict(zip(need, ex.map(lambda r: np.asarray(flux[r], dtype=np.float32), need)))

        # figures
        n_pool = next(iter(arms.values()))["n"]
        stem = f"{pool}_{variant}"
        # Only meaningful with the AION column present; without it this figure would
        # just duplicate joint_base.
        paper = [arms[a] for a in PAPER_ORDER if a in arms]
        if "aion" in arms and len(paper) >= 2:
            render_joint(paper, imgs,
                         f"Top-12 flow anomalies — paper column order — {n_pool:,} HSC ({vlabel})",
                         OUT_DIR / f"joint_paper3_{stem}.png")
        for model, (pa, pb) in MODEL_PAIRS.items():
            if pa in arms and pb in arms:
                render_joint([arms[pa], arms[pb]], imgs,
                             f"Top-12 flow anomalies — {MODEL_LABEL[model]} — {n_pool:,} HSC ({vlabel})",
                             OUT_DIR / f"joint_{model}_{stem}.png")
        allcols = [arms[a] for a in ARM_META if a in arms]
        if len(allcols) > 3:
            render_joint(allcols, imgs,
                         f"Top-12 flow anomalies — all encoders — {n_pool:,} HSC ({vlabel})",
                         OUT_DIR / f"joint_all_{stem}.png")

        # report
        lines = [f"# Top-12 overlap — {pool} pool ({n_pool:,} galaxies) — {vlabel} flows", ""]
        lines.append(f"Pool: {pool_label}. Dedup: {DEDUP_ARCSEC}\" RA/Dec. "
                     f"Sky match: {MATCH_TOL}\". Panel indices refer to these figures.")
        lines.append("")
        lines.append("| arm | dedup dropped | rank of #1..#12 |")
        lines.append("|---|---|---|")
        for a, d in arms.items():
            lines.append(f"| {ARM_META[a][0]} | {d['ndrop']} | "
                         f"{', '.join(str(int(x)) for x in d['rank'])} |")

        lines.append("")
        lines.append("## Pairwise counts (physics / instrument of the target model)")
        lines.append("")
        lines.append("| source arm | → base | → contrastive | → FM control |")
        lines.append("|---|---|---|---|")
        for a in arms:
            row = [ARM_META[a][0]]
            for model, (pa, pb) in MODEL_PAIRS.items():
                if pa == a or pb == a or pa not in arms or pb not in arms:
                    row.append("—")
                    continue
                sp = sep_arcsec(arms[a]["ra"], arms[a]["dec"], arms[pa]["ra"], arms[pa]["dec"])
                si = sep_arcsec(arms[a]["ra"], arms[a]["dec"], arms[pb]["ra"], arms[pb]["dec"])
                row.append(f"{int((sp.min(axis=1) < MATCH_TOL).sum())}p / "
                           f"{int((si.min(axis=1) < MATCH_TOL).sum())}i")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("## Sentences")
        for a in arms:
            sentences(a, arms, lines)

        lines.append("")
        lines.append("## Internal purity (own physics top-12 vs own instrument top-12)")
        for model, (pa, pb) in MODEL_PAIRS.items():
            if pa in arms and pb in arms:
                s = sep_arcsec(arms[pa]["ra"], arms[pa]["dec"], arms[pb]["ra"], arms[pb]["dec"])
                lines.append(f"- {MODEL_LABEL[model]}: "
                             f"{int((s.min(axis=1) < MATCH_TOL).sum())}/12 objects in BOTH its panels.")

        lines.append("")
        lines.append("## Object IDs")
        for a, d in arms.items():
            lines.append(f"- **{ARM_META[a][0]}**: " +
                         ", ".join(f"#{i+1}={o_oid[r]}" for i, r in enumerate(d["raw"])))

        rep = OUT_DIR / f"report_{stem}.md"
        rep.write_text("\n".join(lines) + "\n")
        print(f"[report] {rep}")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
