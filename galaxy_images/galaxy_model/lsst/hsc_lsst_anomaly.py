"""Anomaly detection in physics and instrument latent space for an HSC x LSST checkpoint.

Same recipe as the paper pipeline (anomaly_detection/{encode_latents_ours,
encode_latents_instrument,fit_and_score,run_flow_only}.py), re-hosted on the
HSC x LSST RAM stores so it needs neither neighbours_v2.h5 nor HSC/Legacy
naming. Three stages, one command:

  1. ENCODE   every COSMOS anchor passing the D3 cut (33,346), in BOTH surveys,
              through BOTH encoders:
                phys = encoder_1 (physics, fed the anchor image)
                ins  = encoder_2 (instrument, fed the anchor image — as the
                       paper did; during training it sees neighbours)
              saved per (encoder, survey): *_flat (N, n_tokens*dim = 64) and
              *_mean (N, dim = 16), plus a joint mean over both surveys' tokens.
  2. SCORE    per representation, four detectors fitted on the model's own
              TRAIN split (seed-42 5 % val recomputed exactly as data_factory
              does) and scored on all rows:
                nsf     zuko NSF, 6 transforms, hidden [64, 64], 50 epochs,
                        Adam 1e-3, batch 512, best-epoch weights (paper recipe)
                iforest IsolationForest(200 trees, contamination auto, seed 42)
                knn     mean distance to k=5 nearest train points
                mahal   Mahalanobis distance to the train Gaussian
              Score convention everywhere: HIGHER = more anomalous.
  3. REPORT   top-16 galleries per (space, survey) shown as HSC | LSST pairs
              of the same galaxy with n_input_i / psf_fwhm_i annotated, a
              physics-vs-instrument score scatter with top-1 % overlaps, score
              vs depth/PSF trends (Spearman), a score correlation matrix, and
              summary.json. The two diagnostics that matter for
              disentanglement: instrument-space scores SHOULD track n_input /
              psf_fwhm; physics-space scores should NOT, and the top-k sets of
              the two spaces should barely overlap.

Outputs (never overwritten; pass --force to rebuild):
  /work1/jeroenaudenaert/pablomer/outputs/hsc_lsst/anomaly/<tag>/
      latents.h5  scores.h5  scores.parquet  summary.json  figures/*.png

Run (GPU; the login node's 32 GB vmem cap kills the encoder forward):
  sbatch galaxy_images/galaxy_model/lsst/hsc_lsst_anomaly.sbatch \
      --checkpoint <best.ckpt> --tag cosine-best10k
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

try:
    torch.backends.cuda.preferred_blas_library("hipblas")   # MI210 hipBLASLt bug
except Exception:
    pass

from galaxy_images.galaxy_model.lsst.hsc_lsst_dataset import HscLsstRAM48Dataset, LSST_DIR_DEFAULT

OUT_ROOT = Path("/work1/jeroenaudenaert/pablomer/outputs/hsc_lsst/anomaly")
TOP_K_UMAP = 10                            # anomalies marked on the UMAP plots


def discord_post(text: str, file_paths=None) -> bool:
    """Post text (+ up to 10 files) to Pablo's Discord webhook. The URL is read
    from $DISCORD_WEBHOOK_URL or ~/.config/discord_webhook and never committed.
    Never raises: a dead webhook must not kill the job."""
    try:
        from galaxy_images.galaxy_model.lsst.single_latent.ae_baseline.notify import discord_post as _post
        return _post(text, list(file_paths or []))
    except Exception:
        pass
    try:                                                   # fallback: same logic inline
        import requests
        url = os.environ.get("DISCORD_WEBHOOK_URL") or open(os.path.expanduser("~/.config/discord_webhook")).read().strip()
        handles = [open(p, "rb") for p in (file_paths or [])]
        files = {f"file{i}": (os.path.basename(p), h) for i, (p, h) in enumerate(zip(file_paths or [], handles))}
        r = requests.post(url, data={"content": text[:1990]}, files=files or None, timeout=60)
        for h in handles:
            h.close()
        return r.status_code in (200, 204)
    except Exception as e:                                 # noqa: BLE001
        print(f"[notify] discord post failed: {e}", flush=True)
        return False
BANDS = ("g", "r", "i", "z")
SPLIT_SEED, VAL_RATIO = 42, 0.05          # must match configs/hsc_lsst_cosmos.json
METHODS = ("nsf", "iforest", "knn", "mahal")
# representations that get galleries/diagnostics (the paper's choice: physics =
# pooled encoder_1, instrument = flattened encoder_2), on each survey's image
GALLERY_KEYS = ["phys/hsc_mean", "phys/lsst_mean", "ins/hsc_flat", "ins/lsst_flat"]


def gallery_keys(views):
    """--views lsst restricts every search/diagnostic to latents of the LSST image."""
    return [k for k in GALLERY_KEYS if any(f"/{v}_" in k for v in views)]


# ------------------------------------------------------------------ helpers
def train_val_positions(n: int, val_ratio: float = VAL_RATIO, seed: int = SPLIT_SEED):
    """Reproduce data_factory's split: random_split over positions with a
    seeded generator. random_split only uses randperm(n, generator), so the
    dataset content is irrelevant and this is exact. With the defaults this is
    the training run's split; other ratios give a held-out slice for fields
    the model never trained on (DEEP2-3)."""
    val_size = int(n * val_ratio)
    train_size = n - val_size
    g = torch.Generator().manual_seed(seed)
    tr, va = torch.utils.data.random_split(list(range(n)), [train_size, val_size], generator=g)
    is_val = np.zeros(n, dtype=bool)
    is_val[np.asarray(va.indices)] = True
    return is_val


def load_model(ckpt: str, device):
    from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
    m = ConditionalFlowMatchingModule.load_from_checkpoint(ckpt, map_location="cpu")
    m.eval()
    return m.to(device)


# ------------------------------------------------------------------- encode
@torch.no_grad()
def encode(model, ds: HscLsstRAM48Dataset, device, batch_size=512, n_max=None, views=("hsc", "lsst")):
    """Latents of every row through both encoders, for the requested image views.
    With views=("lsst",) only the LSST image is encoded (the HSC image is never
    touched by the model); the joint mean needs both views and is skipped."""
    n = len(ds) if n_max is None else min(n_max, len(ds))
    both = set(views) == {"hsc", "lsst"}
    out = {f"{e}/{s}_{r}": [] for e in ("phys", "ins") for s in views for r in ("flat", "mean")}
    if both:
        out.update({"phys/joint_mean": [], "ins/joint_mean": []})
    enc = {"phys": model.encoder_1, "ins": model.encoder_2}
    t0 = time.time()
    for s0 in range(0, n, batch_size):
        pos = np.arange(s0, min(s0 + batch_size, n))
        rows = ds.anchor_rows[pos]
        imgs = {}
        if "hsc" in views:
            imgs["hsc"] = torch.from_numpy(np.stack([ds._hsc_img(int(ds.parent_row[r])) for r in rows]).astype(np.float32)).to(device)
        if "lsst" in views:
            imgs["lsst"] = torch.from_numpy(ds.lsst[rows].astype(np.float32)).to(device)
        for e, f in enc.items():
            z = {s: f(x) for s, x in imgs.items()}                 # (B, n_tokens, dim) per view
            for s, zs in z.items():
                out[f"{e}/{s}_flat"].append(zs.flatten(1).cpu().numpy())
                out[f"{e}/{s}_mean"].append(zs.mean(1).cpu().numpy())
            if both:
                out[f"{e}/joint_mean"].append(torch.cat([z["hsc"], z["lsst"]], 1).mean(1).cpu().numpy())
        if (s0 // batch_size) % 10 == 0:
            print(f"[encode] {min(s0 + batch_size, n):,}/{n:,}  {time.time() - t0:.0f}s", flush=True)
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}, n


# -------------------------------------------------------------------- score
def score_nsf(train, allz, device, epochs=50, lr=1e-3, batch=512, seed=0):
    import zuko
    torch.manual_seed(seed)
    flow = zuko.flows.NSF(features=allz.shape[1], context=0, transforms=6, hidden_features=[64, 64]).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    tr = torch.tensor(train, dtype=torch.float32, device=device)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        flow.train()
        perm = torch.randperm(len(tr), device=device)
        tot, nb = 0.0, 0
        for i in range(0, len(tr), batch):
            loss = -flow().log_prob(tr[perm[i:i + batch]]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if tot / nb < best:
            best, best_state = tot / nb, {k: v.clone() for k, v in flow.state_dict().items()}
    flow.load_state_dict(best_state); flow.eval()
    al = torch.tensor(allz, dtype=torch.float32, device=device)
    with torch.no_grad():
        s = np.concatenate([(-flow().log_prob(al[i:i + batch])).cpu().numpy() for i in range(0, len(al), batch)])
    return s.astype(np.float32), float(best)


def score_iforest(train, allz):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1).fit(train)
    return (-clf.score_samples(allz)).astype(np.float32)          # higher = more anomalous


def score_knn(train, allz, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(train)
    d, _ = nn.kneighbors(allz)
    # for train rows the nearest neighbour is the point itself (distance 0): drop it
    return d[:, 1:].mean(1).astype(np.float32)


def score_mahal(train, allz):
    mu = train.mean(0)
    cov = np.cov(train, rowvar=False) + 1e-6 * np.eye(train.shape[1])
    icov = np.linalg.inv(cov)
    d = allz - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, icov, d)).astype(np.float32)


def score_all(latents: dict, fit_mask: np.ndarray, device, nsf_epochs: int):
    """fit_mask: rows the detectors are fitted on (the in-distribution set);
    every row is scored. For --scope anchors that is the model's train split;
    for --scope lsst_all it is the COSMOS train split inside the full store."""
    scores, nsf_nll = {}, {}
    for key, z in latents.items():
        tr = z[fit_mask]
        t0 = time.time()
        s, nll = score_nsf(tr, z, device, epochs=nsf_epochs)
        scores[f"{key}/nsf"] = s; nsf_nll[key] = nll
        scores[f"{key}/iforest"] = score_iforest(tr, z)
        scores[f"{key}/knn"] = score_knn(tr, z)
        scores[f"{key}/mahal"] = score_mahal(tr, z)
        print(f"[score] {key:<18} dim={z.shape[1]:>3}  nsf train NLL {nll:8.2f}  ({time.time() - t0:.0f}s)", flush=True)
    return scores, nsf_nll


# ------------------------------------------------------------------- report
def _rgb(x):
    x = np.asarray(x, dtype=np.float32)
    img = np.stack([x[2], x[1], x[0]], -1)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def gallery(ds, meta, order, scores, title, path, k=16):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = order[:k]
    fig, axes = plt.subplots(4, 8, figsize=(13, 7.2))
    for j, p in enumerate(top):
        r, c = divmod(j, 4)
        row = int(ds.anchor_rows[p])
        for cc, (lab, img) in enumerate((("HSC", ds._hsc_img(int(ds.parent_row[row]))), ("LSST", ds.lsst[row]))):
            ax = axes[r, 2 * c + cc]; ax.imshow(_rgb(img), origin="lower"); ax.set_xticks([]); ax.set_yticks([])
            if cc == 0:
                ax.set_title(f"#{j + 1} s={scores[p]:.1f}  n_i={int(meta['n_input_i'][p])} psf={meta['psf_fwhm_i'][p]:.2f}″"
                             + ("  [val]" if meta["is_val"][p] else ""), fontsize=7, loc="left")
            ax.text(2, 2, lab, color="w", fontsize=7, va="bottom")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def umap_with_top_anomalies(latents: dict, scores: dict, meta: dict, figs: Path, tag: str,
                            n_points=15000, k=TOP_K_UMAP, seed=0, views=("hsc", "lsst")):
    """Where do the top-k anomalies sit in latent space?

    Two kinds of panels, all fitted on a random subsample of n_points plus the
    top-k so the marked objects are always embedded:
      * per gallery key (e.g. phys/lsst_mean): one survey's latents, points
        coloured by n_input_i, top-k NSF anomalies as numbered red markers;
      * joint, per encoder: HSC and LSST latents of the same 33k galaxies
        embedded together, coloured by survey (the disentanglement view: the
        physics encoder should mix the surveys, the instrument encoder should
        separate them), with the top-k physics anomalies (circles) and top-k
        instrument anomalies (squares) of the LSST view marked.
    Returns the list of figure paths.
    """
    import umap
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    n = len(meta["is_val"])
    base = rng.choice(n, min(n_points, n), replace=False)
    paths = []

    def _mark(ax, xy, order, marker="o", color="#d62728", label="top anomalies"):
        for rank, p in enumerate(order[:k], 1):
            ax.scatter(*xy[p], s=140, marker=marker, facecolor="none", edgecolor=color, linewidth=1.8, zorder=5)
            ax.annotate(str(rank), xy[p], xytext=(5, 5), textcoords="offset points", fontsize=8, color=color, weight="bold", zorder=6)
        ax.scatter([], [], s=80, marker=marker, facecolor="none", edgecolor=color, label=label)

    # --- per-space panels
    for key in gallery_keys(views):
        z, s = latents[key], scores[f"{key}/nsf"]
        top = np.argsort(-s)[:k]
        idx = np.unique(np.concatenate([base, top]))
        pos = {p: i for i, p in enumerate(idx)}
        emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(z[idx])
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=3, c=meta["n_input_i"][idx], cmap="viridis", alpha=0.55)
        plt.colorbar(sc, ax=ax, label="n_input_i")
        xy = np.zeros((n, 2)); xy[idx] = emb
        _mark(ax, xy, top, label=f"top-{k} NSF anomalies")
        ax.set_title(f"{tag}: UMAP of {key} ({len(idx):,} pts) with top-{k} anomalies", fontsize=10)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.legend(loc="best", fontsize=8)
        p = figs / f"umap_top{k}_{key.replace('/', '_')}.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # --- joint HSC+LSST panels, one per encoder (needs both views)
    for enc, title in ((("phys", "physics encoder (encoder_1)"), ("ins", "instrument encoder (encoder_2)")) if set(views) == {"hsc", "lsst"} else ()):
        rep = "mean" if enc == "phys" else "flat"
        zh, zl = latents[f"{enc}/hsc_{rep}"], latents[f"{enc}/lsst_{rep}"]
        top_phys = np.argsort(-scores[f"phys/lsst_mean/nsf"])[:k]
        top_ins = np.argsort(-scores[f"ins/lsst_flat/nsf"])[:k]
        idx = np.unique(np.concatenate([base, top_phys, top_ins]))
        Z = np.concatenate([zh[idx], zl[idx]])
        emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(Z)
        m = len(idx)
        fig, ax = plt.subplots(figsize=(8, 6.8))
        ax.scatter(emb[:m, 0], emb[:m, 1], s=3, c="#2a78d6", alpha=0.35, label="HSC image")
        ax.scatter(emb[m:, 0], emb[m:, 1], s=3, c="#eb6834", alpha=0.35, label="LSST image")
        xy_l = np.zeros((n, 2)); xy_l[idx] = emb[m:]                 # LSST-view coordinates
        _mark(ax, xy_l, top_phys, marker="o", color="#d62728", label=f"top-{k} physics-space anomalies (LSST view)")
        _mark(ax, xy_l, top_ins, marker="s", color="#111111", label=f"top-{k} instrument-space anomalies (LSST view)")
        ax.set_title(f"{tag}: joint UMAP of the {title}, both surveys ({m:,} galaxies × 2)", fontsize=10)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.legend(loc="best", fontsize=8, markerscale=2)
        p = figs / f"umap_joint_{enc}_top{k}.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # --- lsst_all only: LSST-view UMAP per encoder coloured by GROUP (COSMOS train/val/low-exposure, DEEP2-3),
    #     the domain-shift picture; top-k of that space marked.
    groups = np.asarray(meta.get("group", np.array(["all"] * n)))
    gnames = [g for g in GROUP_ORDER if (groups == g).any()]
    if len(gnames) > 1:
        colors = GROUP_COLORS
        for key, title in (("phys/lsst_mean", "physics encoder"), ("ins/lsst_flat", "instrument encoder")):
            z, s = latents[key], scores[f"{key}/nsf"]
            top = np.argsort(-s)[:k]
            idx = np.unique(np.concatenate([base, top]))
            emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(z[idx])
            fig, ax = plt.subplots(figsize=(8, 6.8))
            for g in gnames:
                m_ = groups[idx] == g
                ax.scatter(emb[m_, 0], emb[m_, 1], s=3, c=colors[g], alpha=0.45, label=f"{g} ({int(m_.sum()):,})")
            xy = np.zeros((n, 2)); xy[idx] = emb
            _mark(ax, xy, top, label=f"top-{k} NSF anomalies (all groups)")
            ax.set_title(f"{tag}: LSST-view UMAP of the {title} by group ({len(idx):,} pts)", fontsize=10)
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.legend(loc="best", fontsize=8, markerscale=2)
            p = figs / f"umap_groups_{key.replace('/', '_')}_top{k}.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    return paths


GROUP_ORDER = ("cosmos_train", "cosmos_val", "cosmos_lowexp", "deep23_fit", "deep23_held", "deep23")
GROUP_COLORS = {"cosmos_train": "#2a78d6", "cosmos_val": "#1baf7a", "cosmos_lowexp": "#eda100",
                "deep23_fit": "#eb6834", "deep23_held": "#e87ba4", "deep23": "#eb6834"}


def report(ds, meta: dict, scores: dict, out: Path, tag: str, views=("hsc", "lsst")):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    figs = out / "figures"; figs.mkdir(exist_ok=True)
    n = len(meta["is_val"])
    summary = {"tag": tag, "n": int(n), "n_val": int(meta["is_val"].sum()), "views": list(views)}
    gkeys = gallery_keys(views)

    # 1) galleries for the paper's physics / instrument representations, NSF and IForest
    for key in gkeys:
        for m in ("nsf", "iforest"):
            s = scores[f"{key}/{m}"]; order = np.argsort(-s)
            gallery(ds, meta, order, s, f"{tag}: top anomalies in {key} by {m} (each galaxy shown as HSC | LSST)",
                    figs / f"top16_{key.replace('/', '_')}_{m}.png")

    # 2) physics vs instrument: overlap of top-1% sets and a scatter, per survey
    top_frac = 0.01; k = max(1, int(n * top_frac)); summary["top_frac"] = top_frac
    summary["overlap_top1pct"] = {}
    fig, axes = plt.subplots(1, len(views), figsize=(5.5 * len(views), 4.8), squeeze=False); axes = axes[0]
    for ax, sv in zip(axes, views):
        sp, si = scores[f"phys/{sv}_mean/nsf"], scores[f"ins/{sv}_flat/nsf"]
        tp, ti = set(np.argsort(-sp)[:k]), set(np.argsort(-si)[:k])
        ov = len(tp & ti); summary["overlap_top1pct"][sv] = {"k": k, "overlap": ov, "expected_random": k * k / n}
        sc = ax.scatter(sp, si, s=3, c=meta["n_input_i"], cmap="viridis", alpha=0.6)
        ax.set_xlabel(f"physics NSF score (encoder_1, {sv} image)"); ax.set_ylabel(f"instrument NSF score (encoder_2, {sv} image)")
        ax.set_title(f"{sv}: top-1% overlap {ov}/{k} (random ≈ {k * k / n:.1f})", fontsize=10)
        plt.colorbar(sc, ax=ax, label="n_input_i")
    fig.suptitle(f"{tag}: physics-space vs instrument-space anomaly scores", fontsize=10); fig.tight_layout()
    fig.savefig(figs / "phys_vs_ins_scatter.png", dpi=130); plt.close(fig)

    # 3) does the score track depth / PSF? Spearman per space, per survey (LSST metadata only exists for LSST)
    summary["spearman_vs_lsst_meta"] = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, (key, lab) in enumerate((("phys/lsst_mean/nsf", "physics (LSST image)"), ("ins/lsst_flat/nsf", "instrument (LSST image)"))):
        s = scores[key]
        for j, (col, xl) in enumerate((("n_input_i", "n_input_i (exposures in the i coadd)"), ("psf_fwhm_i", "psf_fwhm_i [″]"))):
            rho = spearmanr(meta[col], s, nan_policy="omit").statistic     # DEEP2-3 has rows without a PSF value
            summary["spearman_vs_lsst_meta"][f"{key} vs {col}"] = float(rho)
            ax = axes[i, j]; ax.scatter(meta[col], s, s=3, alpha=0.4); ax.set_xlabel(xl); ax.set_ylabel(f"{lab} NSF score")
            ax.set_title(f"Spearman ρ = {rho:+.3f}", fontsize=10)
    fig.suptitle(f"{tag}: anomaly score vs LSST observing conditions (instrument space should track these; physics should not)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(figs / "score_vs_depth_psf.png", dpi=130); plt.close(fig)

    # 4) score correlation matrix (Spearman) across every representation x method
    keys = sorted(scores); M = np.stack([scores[k] for k in keys], 1)
    C = pd.DataFrame(M, columns=keys).corr(method="spearman").to_numpy()
    fig, ax = plt.subplots(figsize=(0.32 * len(keys) + 3, 0.32 * len(keys) + 2.5))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r"); plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=90, fontsize=6); ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=6)
    ax.set_title(f"{tag}: Spearman correlation of anomaly scores", fontsize=10); fig.tight_layout()
    fig.savefig(figs / "score_correlation.png", dpi=130); plt.close(fig)

    # 5) held-out vs fit score shift (a flow that memorised its fit set scores held-out rows higher)
    held = ~meta["fit_mask"]
    summary["heldout_minus_fit_median_nsf"] = {k: float(np.median(scores[f"{k}/nsf"][held]) - np.median(scores[f"{k}/nsf"][~held])) for k in gkeys} if held.any() else {}

    # 6) group breakdown (any scope with >1 group): how each group scores relative
    #    to the group the detectors were fitted on, in physics vs instrument space.
    #    Across fields (--scope lsst_all) the disentanglement claim predicts DEEP2-3
    #    (13x fewer exposures) is far more anomalous in INSTRUMENT space than in
    #    PHYSICS space. Plus per-group galleries, because one ranking over mixed
    #    groups would be dominated by the out-of-distribution group.
    groups = np.asarray(meta.get("group", np.array(["all"] * n)))
    gnames = [g for g in GROUP_ORDER if (groups == g).any()]
    ref_group = str(groups[meta["fit_mask"]][0])
    if len(gnames) > 1:
        summary["group_sizes"] = {g: int((groups == g).sum()) for g in gnames}
        summary["fit_group"] = ref_group
        summary["group_median_nsf_minus_fit_group"] = {}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, (key, lab) in zip(axes, (("phys/lsst_mean/nsf", "physics space (encoder_1, LSST image)"),
                                        ("ins/lsst_flat/nsf", "instrument space (encoder_2, LSST image)"))):
            s = scores[key]; ref = np.median(s[groups == ref_group])
            lo, hi = np.percentile(s, [0.5, 99.5]); bins = np.linspace(lo, hi, 80)
            for g in gnames:
                ax.hist(s[groups == g], bins=bins, histtype="step", lw=1.6, density=True, color=GROUP_COLORS.get(g), label=f"{g} (n={int((groups == g).sum()):,})")
                summary["group_median_nsf_minus_fit_group"][f"{key} | {g}"] = float(np.median(s[groups == g]) - ref)
            ax.set_xlabel(f"NSF anomaly score (−log p, fitted on {ref_group})"); ax.set_ylabel("density"); ax.set_title(lab, fontsize=10)
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f"{tag}: score distributions per group (detectors fitted on {ref_group})", fontsize=9.5)
        fig.tight_layout(); fig.savefig(figs / "group_score_distributions.png", dpi=130); plt.close(fig)
        # per-group galleries for the two LSST-view spaces
        for g in gnames:
            if g == ref_group:
                continue
            sel = np.where(groups == g)[0]
            for key in ("phys/lsst_mean", "ins/lsst_flat"):
                s = scores[f"{key}/nsf"]; order = sel[np.argsort(-s[sel])]
                gallery(ds, meta, order, s, f"{tag}: top anomalies within {g} in {key} by nsf (HSC | LSST)",
                        figs / f"top16_{g}_{key.replace('/', '_')}_nsf.png")
    return summary


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="output subdir name, e.g. cosine-best10k")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--nsf-epochs", type=int, default=50)
    ap.add_argument("--n-max", type=int, default=None, help="encode only the first N anchors (smoke)")
    ap.add_argument("--scope", choices=("anchors", "lsst_all", "field"), default="anchors",
                    help="anchors: the 33,346 COSMOS training anchors (default). lsst_all: every row of the LSST "
                         "store (65,163: COSMOS incl. low-exposure objects + DEEP2-3); detectors are still fitted on "
                         "the COSMOS train split, so DEEP2-3 and low-exposure COSMOS are out-of-distribution probes. "
                         "field: one field on its own (--field), detectors fitted WITHIN that field — COSMOS on the "
                         "model's train split, DEEP2-3 (never trained on) on a seeded 80 %% slice — and scored within it.")
    ap.add_argument("--field", choices=("COSMOS", "DEEP2-3"), default=None, help="required for --scope field")
    ap.add_argument("--views", choices=("both", "lsst"), default="both",
                    help="both: encode HSC and LSST images (default). lsst: encode only the LSST image; every "
                         "flow, ranking, UMAP and diagnostic then uses latents of the LSST image alone.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.scope == "field" and not args.field:
        ap.error("--scope field needs --field COSMOS|DEEP2-3")
    views = ("hsc", "lsst") if args.views == "both" else ("lsst",)

    out = OUT_ROOT / args.tag
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pick another --tag or pass --force (no silent overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    lm = pd.read_parquet(Path(LSST_DIR_DEFAULT) / "lsst_meta.parquet")
    # The training selection (D1-D3) and its seed-42 split, expressed as store rows.
    # Recomputed here exactly as HscLsstRAM48Dataset does it, so it is valid whichever
    # scope the dataset below is built with.
    cos_ok = (lm["field"] == "COSMOS").to_numpy() & np.all([lm[f"n_input_{b}"].to_numpy() >= 5 for b in BANDS], axis=0)
    cos_rows = np.where(cos_ok)[0]
    cos_is_val = train_val_positions(len(cos_rows))
    train_rows = set(cos_rows[~cos_is_val].tolist()); val_rows = set(cos_rows[cos_is_val].tolist())

    if args.scope == "anchors":
        ds = HscLsstRAM48Dataset()                   # D1-D3 selection, same as training
        assert np.array_equal(ds.anchor_rows, cos_rows)
    elif args.scope == "lsst_all":
        ds = HscLsstRAM48Dataset(field=None, min_n_input=0)   # every LSST row; HSC side via parent_row
    else:
        ds = HscLsstRAM48Dataset(field=args.field, min_n_input=0)   # one whole field, no exposure cut
    model = load_model(args.checkpoint, device)
    print(f"[setup] scope={args.scope}{'/' + args.field if args.field else ''} views={views}: {len(ds):,} rows | "
          f"ckpt {args.checkpoint} | survey_names {model.survey_names} | {time.time() - t0:.0f}s", flush=True)

    latents, n = encode(model, ds, device, args.batch_size, args.n_max, views=views)
    rows = ds.anchor_rows[:n]
    field = lm["field"].to_numpy()[rows]
    if args.scope == "field" and args.field == "DEEP2-3":
        # never trained on: fit the detectors on a seeded 80 % slice of the field, hold out 20 %
        d_held = train_val_positions(n, val_ratio=0.2, seed=SPLIT_SEED)
        is_val, fit_mask = d_held, ~d_held
        group = np.where(fit_mask, "deep23_fit", "deep23_held")
    else:
        is_val = np.array([int(r) in val_rows for r in rows])
        fit_mask = np.array([int(r) in train_rows for r in rows])       # the model's own COSMOS train split
        group = np.where(fit_mask, "cosmos_train", np.where(is_val, "cosmos_val", np.where(field == "COSMOS", "cosmos_lowexp", "deep23")))
    meta = {"object_id_hsc": lm["object_id_hsc"].to_numpy()[rows], "lsst_row": rows,
            "parent_row": ds.parent_row[rows], "is_val": is_val, "fit_mask": fit_mask, "field": field, "group": group}
    for b in BANDS:
        meta[f"n_input_{b}"] = lm[f"n_input_{b}"].to_numpy()[rows]
        meta[f"psf_fwhm_{b}"] = lm[f"psf_fwhm_{b}"].to_numpy()[rows]
    print(f"[setup] groups: " + ", ".join(f"{g}={int((group == g).sum()):,}" for g in np.unique(group)), flush=True)
    if fit_mask.sum() < 1000:
        raise SystemExit(f"only {fit_mask.sum()} in-distribution rows to fit on (smoke with --n-max too small for lsst_all?)")

    with h5py.File(out / "latents.h5", "w") as f:
        for k, v in latents.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        for k, v in meta.items():
            # h5py has no unicode dtype: store string columns as fixed-width bytes
            arr = np.asarray(v)
            if arr.dtype.kind in ("U", "O"):
                arr = arr.astype(str).astype("S")
            f.create_dataset(f"meta/{k}", data=arr)
        f.attrs.update({"checkpoint": args.checkpoint, "tag": args.tag, "scope": args.scope, "field": args.field or "", "views": ",".join(views),
                        "n": n, "split_seed": SPLIT_SEED, "val_ratio": VAL_RATIO, "n_tokens": 4,
                        "dim": latents[f"phys/{views[-1]}_mean"].shape[1], "flow_fit_on": "meta/fit_mask == True"})
    print(f"[encode] saved latents for {n:,} rows ({time.time() - t0:.0f}s)", flush=True)

    scores, nsf_nll = score_all(latents, fit_mask, device, args.nsf_epochs)
    with h5py.File(out / "scores.h5", "w") as f:
        for k, v in scores.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        f.create_dataset("is_val", data=is_val)
        f.attrs.update({"checkpoint": args.checkpoint, "tag": args.tag, "nsf_epochs": args.nsf_epochs, "methods": list(METHODS)})
    df = pd.DataFrame({k.replace("/", "__"): v for k, v in scores.items()})
    for k in ("object_id_hsc", "lsst_row", "parent_row", "is_val", "fit_mask", "field", "group", "n_input_i", "psf_fwhm_i"):
        df[k] = meta[k]
    df.to_parquet(out / "scores.parquet", index=False)

    summary = report(ds, meta, scores, out, args.tag, views=views)
    figs = out / "figures"
    umap_paths = umap_with_top_anomalies(latents, scores, meta, figs, args.tag, views=views)
    summary.update({"checkpoint": args.checkpoint, "nsf_train_nll": nsf_nll, "wall_min": round((time.time() - t0) / 60, 1)})
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: summary.get(k) for k in ("overlap_top1pct", "spearman_vs_lsst_meta", "heldout_minus_fit_median_nsf", "group_median_nsf_minus_fit_group")}, indent=1, default=float))
    print(f"[done] {out}  ({summary['wall_min']} min)", flush=True)

    # ---------------------------------------------------------- Discord
    ov = summary["overlap_top1pct"]; sp = summary["spearman_vs_lsst_meta"]
    shift = ""
    if "group_median_nsf_minus_fit_group" in summary:
        gm = summary["group_median_nsf_minus_fit_group"]; ref = summary["fit_group"]
        parts = [f"{g}: physics {gm.get(f'phys/lsst_mean/nsf | {g}', float('nan')):+.2f} | instrument {gm.get(f'ins/lsst_flat/nsf | {g}', float('nan')):+.2f}"
                 for g in summary["group_sizes"] if g != ref]
        shift = f"median NSF score shift vs {ref} — " + "; ".join(parts) + "\n"
    ovl = "; ".join(f"{sv.upper()} {ov[sv]['overlap']}/{ov[sv]['k']}" for sv in views)
    scope_txt = f"{args.scope}{'/' + args.field if args.field else ''}, views={'+'.join(views)}"
    text = (f"**HSC × LSST anomaly stage — `{args.tag}`** ({scope_txt}) done in {summary['wall_min']} min, N={n:,} LSST rows, "
            f"detectors fitted on {int(fit_mask.sum()):,} rows ({summary.get('fit_group', 'fit set')})\n"
            f"ckpt: `{Path(args.checkpoint).parent.parent.name}/{Path(args.checkpoint).name}`\n" + shift +
            f"top-1% overlap physics∩instrument: {ovl} (random ≈ {ov['lsst']['expected_random']:.1f})\n"
            f"Spearman vs n_input_i: physics {sp['phys/lsst_mean/nsf vs n_input_i']:+.3f} | instrument {sp['ins/lsst_flat/nsf vs n_input_i']:+.3f}\n"
            f"Spearman vs psf_fwhm_i: physics {sp['phys/lsst_mean/nsf vs psf_fwhm_i']:+.3f} | instrument {sp['ins/lsst_flat/nsf vs psf_fwhm_i']:+.3f}\n"
            f"outputs: `{out}`")
    batch1 = [figs / "top16_phys_lsst_mean_nsf.png", figs / "top16_ins_lsst_flat_nsf.png",
              figs / "phys_vs_ins_scatter.png", figs / "score_vs_depth_psf.png"]
    batch2 = umap_paths + [figs / "top16_phys_hsc_mean_nsf.png", figs / "top16_ins_hsc_flat_nsf.png", figs / "score_correlation.png"]
    ok1 = discord_post(text, [str(p) for p in batch1 if Path(p).exists()])
    ok2 = discord_post(f"`{args.tag}` — UMAPs with the top-{TOP_K_UMAP} anomalies marked, HSC-view galleries, score correlations",
                       [str(p) for p in batch2 if Path(p).exists()][:10])
    ok3 = True
    if (figs / "group_score_distributions.png").exists():
        batch3 = [figs / "group_score_distributions.png"] + sorted(p for p in figs.glob("top16_*_nsf.png")
                                                                    if p.name.split("_")[1] in ("deep23", "cosmos") and not p.name.startswith(("top16_phys", "top16_ins")))
        ok3 = discord_post(f"`{args.tag}` — score distributions per group (fitted on {summary.get('fit_group')}) and within-group galleries",
                           [str(p) for p in batch3 if Path(p).exists()][:10])
    print(f"[notify] discord: summary {'sent' if ok1 else 'FAILED'}, figures {'sent' if ok2 else 'FAILED'}, groups {'sent' if ok3 else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
