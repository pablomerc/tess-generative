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
# Primary flow score used for galleries, UMAP markers, overlaps and the Discord
# headline: "nsf" (unconditional) by default; main() switches it to "cnsf" when
# --cond n_input is given, i.e. p(z | log exposures per band) — "anomalous for
# its number of visits" — and to "ccal" when --calibrate K is added on top.
# The unconditional flow is still fitted alongside so every diagnostic can show
# what the conditioning removed.
#
# Why "ccal" exists. −log p(z | c) is NOT comparable across depths: the latents
# of shallow stamps are spread over a much larger volume of latent space than
# those of deep stamps (noisier images -> noisier tokens), so even a perfect
# conditional flow assigns them a lower density. In the 64-d instrument space
# the in-sample median −log p(z | c) runs from −8 at 5–10 visits to −65 at
# 80–100 visits (D11): the conditional NLL is still depth-ranked, because it
# measures the conditional ENTROPY at that depth, not how unusual the object
# is for its depth. "ccal" is the highest-density-region tail probability
# under the same conditional flow, p = P[−log p(z'|c) ≥ −log p(z|c)] for
# z' ~ p(·|c): uniform on [0, 1] for typical objects at every depth if the
# flow is right, so −log10 p is the depth-calibrated anomaly score.
PM = "nsf"
PM_LABEL = {"nsf": "−log p(z)", "cnsf": "−log p(z | log n_input g,r,i,z)",
            "ccal": "−log10 P[−log p(z′|c) ≥ −log p(z|c)], z′ ~ p(·|c)  (depth-calibrated)",
            "cemp": "−log10 (within-context-cluster percentile of −log p(z|c) among fit rows)"}
COND_BANDS = ("g", "r", "i", "z")


def build_context(meta: dict, fit_mask: np.ndarray, with_psf: bool = False) -> np.ndarray:
    """Flow context: log1p(n_input) per griz band (column 0 = i band, the
    reference for balancing), optionally followed by psf_fwhm per band;
    standardised on the fit set. log because exposures span 1–114 and depth ∝
    log N; standardised so the spline flow's context network sees O(1) inputs.
    Missing PSF values (some DEEP2-3 rows) are filled with the fit-set median."""
    cols = [np.log1p(meta["n_input_i"].astype(np.float32))] + \
           [np.log1p(meta[f"n_input_{b}"].astype(np.float32)) for b in COND_BANDS if b != "i"]
    if with_psf:
        for b in COND_BANDS:
            p = meta[f"psf_fwhm_{b}"].astype(np.float32).copy()
            p[~np.isfinite(p)] = np.nanmedian(p[fit_mask])
            cols.append(p)
    c = np.stack(cols, 1)
    mu, sd = c[fit_mask].mean(0), c[fit_mask].std(0) + 1e-6
    return ((c - mu) / sd).astype(np.float32)
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
def context_weights(ctx_train: np.ndarray, n_bins: int = 12) -> np.ndarray:
    """Inverse-density weights on the first context column (log exposures in the
    reference band), so that a fit set dominated by one depth (COSMOS: 75–105
    exposures) still teaches the conditional flow what latents look like at the
    depths where it is sparse. Weights are clipped to [0.2, 20] and normalised
    to mean 1, so the loss scale is unchanged."""
    c = ctx_train[:, 0]
    edges = np.quantile(c, np.linspace(0, 1, n_bins + 1)); edges[0] -= 1e-6; edges[-1] += 1e-6
    # quantile bins have equal counts by construction, so density ∝ 1 / bin width
    width = np.diff(edges)
    idx = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, n_bins - 1)
    w = width[idx] / width.mean()                     # wide (sparse) bins get large weights
    w = np.clip(w, 0.2, 20.0)
    return (w / w.mean()).astype(np.float32)


def score_nsf(train, allz, device, epochs=50, lr=1e-3, batch=512, seed=0, ctx_train=None, ctx_all=None, weights=None,
              return_flow=False):
    """Neural spline flow anomaly score −log p(z) (paper recipe), or, when a
    context is given, the CONDITIONAL score −log p(z | c): the flow learns the
    latent distribution at each observing depth, so a stamp is only anomalous
    if it is unusual for its own number of exposures. `weights` (per fit row,
    mean 1) turn the NLL into a weighted NLL — used to balance the fit set
    across the context range. With `return_flow` the fitted flow is returned as
    a third value (needed for the calibrated tail-probability score)."""
    import zuko
    torch.manual_seed(seed)
    cdim = 0 if ctx_train is None else ctx_train.shape[1]
    flow = zuko.flows.NSF(features=allz.shape[1], context=cdim, transforms=6, hidden_features=[64, 64]).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    tr = torch.tensor(train, dtype=torch.float32, device=device)
    ctr = None if cdim == 0 else torch.tensor(ctx_train, dtype=torch.float32, device=device)
    wtr = None if weights is None else torch.tensor(weights, dtype=torch.float32, device=device)
    dist = (lambda c: flow()) if cdim == 0 else (lambda c: flow(c))
    best, best_state = float("inf"), None
    for _ in range(epochs):
        flow.train()
        perm = torch.randperm(len(tr), device=device)
        tot, nb = 0.0, 0
        for i in range(0, len(tr), batch):
            sel = perm[i:i + batch]
            nll = -dist(None if ctr is None else ctr[sel]).log_prob(tr[sel])
            loss = nll.mean() if wtr is None else (nll * wtr[sel]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if tot / nb < best:
            best, best_state = tot / nb, {k: v.clone() for k, v in flow.state_dict().items()}
    flow.load_state_dict(best_state); flow.eval()
    al = torch.tensor(allz, dtype=torch.float32, device=device)
    cal = None if cdim == 0 else torch.tensor(ctx_all, dtype=torch.float32, device=device)
    with torch.no_grad():
        s = np.concatenate([(-dist(None if cal is None else cal[i:i + batch]).log_prob(al[i:i + batch])).cpu().numpy()
                            for i in range(0, len(al), batch)])
    if return_flow:
        return s.astype(np.float32), float(best), flow
    return s.astype(np.float32), float(best)


def context_clusters(ctx_all: np.ndarray, n_centroids: int = 256, seed: int = 1):
    """k-means over the (standardised) context. The context is 4 strongly
    correlated log-exposure numbers, so 256 centroids tile it densely; every
    calibration below is done per cluster."""
    from sklearn.cluster import KMeans
    k = int(min(n_centroids, len(np.unique(ctx_all, axis=0))))
    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(ctx_all.astype(np.float64))
    return km.labels_.astype(np.int64), km.cluster_centers_.astype(np.float32)


def model_tail_prob(flow, z_rows, ctx_rows, lab_rows, centers, device, n_samples=1024, seed=1, batch=512):
    """Model-based depth-calibrated score ("ccal") from a fitted CONDITIONAL flow.

    For a context c the flow implies a distribution of its own NLL values,
    −log p(z' | c) for z' ~ p(· | c). The highest-density-region tail
    probability of row i is
        p_i = P[ −log p(z' | c_i)  >=  −log p(z_i | c_i) ],
    the probability mass of the region of latent space LESS likely than z_i at
    its own depth. If the flow is right, p is Uniform(0, 1) for in-distribution
    rows at every depth, whatever the spread of the latents there — which
    −log p(z | c) itself is not (see PM_LABEL). Sampling an autoregressive flow
    costs one network pass per latent dimension, so the reference NLL
    distribution is drawn once per context CENTROID (n_samples each) and every
    row is compared with the reference of its own cluster (~250k samples
    instead of 30M). Returns (p, z): p = (count + 1) / (n_samples + 1), and
    z = (−log p(z_i|c_i) − mean_ref) / std_ref, used to break ties at the floor."""
    torch.manual_seed(seed)
    flow.eval()
    k = len(centers); cen = torch.tensor(centers, dtype=torch.float32, device=device)
    ref = torch.empty(k, n_samples, device=device)
    with torch.no_grad():
        for j in range(0, k, 16):                               # 16 centroids × n_samples per sampling call
            d = flow(cen[j:j + 16])
            smp = d.sample((n_samples,))                        # (K, B, D) — one context per column
            ref[j:j + 16] = (-d.log_prob(smp)).T                # (B, K)
        ref, _ = torch.sort(ref, dim=1)
        al = torch.tensor(z_rows, dtype=torch.float32, device=device)
        cal = torch.tensor(ctx_rows, dtype=torch.float32, device=device)
        nll = torch.cat([-flow(cal[i:i + batch]).log_prob(al[i:i + batch]) for i in range(0, len(al), batch)])
        r = ref[torch.tensor(lab_rows, device=device)]          # (rows, K) reference NLLs of each row's centroid
        cnt = (r >= nll[:, None]).sum(1).float()                # samples less likely than the row
        p = ((cnt + 1.0) / (n_samples + 1.0)).cpu().numpy().astype(np.float32)
        z = ((nll - r.mean(1)) / r.std(1).clamp_min(1e-6)).cpu().numpy().astype(np.float32)
    return p, z


def empirical_tail_prob(nll_all: np.ndarray, lab_all: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Empirical depth-calibrated score ("cemp"): the reference set for row i is
    the FIT rows in the same context cluster and p_i is the fraction of them
    with a larger conditional NLL (the row itself excluded). With out-of-fold
    scoring every NLL here is out-of-sample, so fit and non-fit rows are
    exchangeable. Calibrated on fit rows by construction; noisier where a
    cluster is small. Agreement with the model-based p validates the flow."""
    n = len(nll_all); p_emp = np.ones(n, dtype=np.float32)
    for c in np.unique(lab_all):
        rows = np.where(lab_all == c)[0]; refrows = rows[fit_mask[rows]]
        if len(refrows) == 0:
            continue                                            # no reference at this context: p = 1 (cannot judge)
        rs = np.sort(nll_all[refrows])
        ge = len(rs) - np.searchsorted(rs, nll_all[rows], side="left")     # ref rows with NLL >= own
        self_in = fit_mask[rows].astype(np.int64)                          # a fit row counts itself once: drop it
        p_emp[rows] = (ge - self_in + 1.0) / (len(rs) - self_in + 1.0)
    return p_emp

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


def score_all(latents: dict, fit_mask: np.ndarray, device, nsf_epochs: int, ctx: np.ndarray | None = None,
              cond_epochs: int | None = None, balance: bool = False, calibrate: int = 0, calib_centroids: int = 256,
              folds: int = 1):
    """fit_mask: rows the detectors are fitted on (the in-distribution set);
    every row is scored. For --scope anchors that is the model's train split;
    for --scope lsst_all it is the COSMOS train split inside the full store
    (or both fields' fit slices with --fit-on both_fields).
    ctx: optional (N, d) context; when given a second flow p(z | ctx) is fitted
    and stored under <key>/cnsf next to the unconditional <key>/nsf. It gets
    `cond_epochs` (default: same as the unconditional budget) and, if `balance`,
    inverse-density weights over the context so sparse depths are learned.
    calibrate > 0: also the depth-calibrated tail-probability score <key>/ccal
    (model-based, `calibrate` flow samples per context centroid) and the
    empirical within-cluster percentile <key>/cemp; p-values, z-scores and
    cluster ids come back in `calib`.
    folds > 1: OUT-OF-FOLD scoring of both flows. A flow fitted for 50–150
    epochs on ~25k points memorises: rows it was fitted on get systematically
    lower NLL than rows it never saw, so held-out rows crowd the top of the
    ranking (DEEP2-3: 20 % of the rows but 80 % of the top 0.1 % under the
    150-epoch conditional flow). With folds, the fit rows are split into
    `folds` seeded folds, one flow per fold is fitted on the other folds and
    scores the held-out fold; every non-fit row is assigned to one fold and
    scored by that fold's flow. Every row is therefore scored by a flow that
    did not see it, fit and non-fit rows are exchangeable, and the reported
    NLL is the mean over folds. IForest / kNN / Mahalanobis stay in-sample."""
    scores, nsf_nll, calib = {}, {}, {}
    n = len(fit_mask); tr_idx = np.where(fit_mask)[0]; other_idx = np.where(~fit_mask)[0]
    w = context_weights(ctx[fit_mask]) if (ctx is not None and balance) else None
    w_full = None
    if w is not None:
        w_full = np.zeros(n, dtype=np.float32); w_full[fit_mask] = w
        print(f"[score] context balancing: weights min {w.min():.2f} max {w.max():.2f} "
              f"(effective sample size {int(w.sum() ** 2 / (w ** 2).sum()):,} of {int(fit_mask.sum()):,})", flush=True)
    folds = max(1, int(folds))
    fold_of = np.zeros(n, dtype=np.int64)
    if folds > 1:
        rng = np.random.default_rng(SPLIT_SEED)
        fold_of[tr_idx] = rng.permutation(len(tr_idx)) % folds
        fold_of[other_idx] = rng.integers(0, folds, len(other_idx))
        print(f"[score] out-of-fold scoring with {folds} folds: {len(tr_idx):,} fit rows -> "
              f"{np.bincount(fold_of[tr_idx]).tolist()} per fold; {len(other_idx):,} non-fit rows spread over the folds", flush=True)
    lab = centers = None
    if calibrate > 0 and ctx is not None:
        lab, centers = context_clusters(ctx, calib_centroids)
    for key, z in latents.items():
        t0 = time.time()
        s_u = np.empty(n, np.float32); s_c = np.empty(n, np.float32); p = np.empty(n, np.float32); zs = np.empty(n, np.float32)
        nll_u_f, nll_c_f = [], []
        for f in range(folds):
            if folds > 1:
                train_rows = tr_idx[fold_of[tr_idx] != f]; score_rows = np.where(fold_of == f)[0]
            else:
                train_rows = tr_idx; score_rows = np.arange(n)
            su, nll_u = score_nsf(z[train_rows], z[score_rows], device, epochs=nsf_epochs, seed=f)
            s_u[score_rows] = su; nll_u_f.append(nll_u)
            if ctx is not None:
                sc, nll_c, cflow = score_nsf(z[train_rows], z[score_rows], device, epochs=cond_epochs or nsf_epochs, seed=f,
                                             ctx_train=ctx[train_rows], ctx_all=ctx[score_rows],
                                             weights=None if w_full is None else w_full[train_rows], return_flow=True)
                s_c[score_rows] = sc; nll_c_f.append(nll_c)
                if calibrate > 0:
                    pf, zf = model_tail_prob(cflow, z[score_rows], ctx[score_rows], lab[score_rows], centers, device,
                                             n_samples=calibrate, seed=f)
                    p[score_rows] = pf; zs[score_rows] = zf
                del cflow
        scores[f"{key}/nsf"] = s_u; nsf_nll[key] = float(np.mean(nll_u_f))
        msg = f"nsf NLL {nsf_nll[key]:8.2f}"
        if ctx is not None:
            scores[f"{key}/cnsf"] = s_c; nsf_nll[f"{key}/cond"] = float(np.mean(nll_c_f))
            msg += f"  cond-nsf NLL {nsf_nll[f'{key}/cond']:8.2f}"
            if calibrate > 0:
                p_emp = empirical_tail_prob(s_c, lab, fit_mask)          # s_c IS −log p(z|c) of the (out-of-fold) flow
                scores[f"{key}/ccal"] = (-np.log10(p) + 1e-3 * zs).astype(np.float32)
                scores[f"{key}/cemp"] = (-np.log10(p_emp)).astype(np.float32)
                calib[key] = {"p": p, "p_emp": p_emp, "z": zs, "cluster": lab.astype(np.int32)}
                # calibration check: fraction of FIT rows with p < 0.1 (0.10 if the flow is right)
                from scipy.stats import spearmanr as _sp
                msg += (f"  calib K={calibrate}: fit-row frac(p<0.1)={float((p[fit_mask] < 0.1).mean()):.3f} "
                        f"rho(ccal,cemp)={_sp(scores[f'{key}/ccal'], scores[f'{key}/cemp']).statistic:+.2f}")
        tr = z[fit_mask]
        scores[f"{key}/iforest"] = score_iforest(tr, z)
        scores[f"{key}/knn"] = score_knn(tr, z)
        scores[f"{key}/mahal"] = score_mahal(tr, z)
        print(f"[score] {key:<18} dim={z.shape[1]:>3}  {msg}  ({time.time() - t0:.0f}s)", flush=True)
    return scores, nsf_nll, calib

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
        z, s = latents[key], scores[f"{key}/{PM}"]
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
        top_phys = np.argsort(-scores[f"phys/lsst_mean/{PM}"])[:k]
        top_ins = np.argsort(-scores[f"ins/lsst_flat/{PM}"])[:k]
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
            z, s = latents[key], scores[f"{key}/{PM}"]
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


def report(ds, meta: dict, scores: dict, out: Path, tag: str, views=("hsc", "lsst"), calib: dict | None = None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    figs = out / "figures"; figs.mkdir(exist_ok=True)
    n = len(meta["is_val"])
    summary = {"tag": tag, "n": int(n), "n_val": int(meta["is_val"].sum()), "views": list(views)}
    gkeys = gallery_keys(views)

    # 1) galleries for the paper's physics / instrument representations: the primary
    #    flow, the unconditional flow when they differ, and IForest
    gal_methods = tuple(dict.fromkeys((PM, "nsf", "iforest")))
    for key in gkeys:
        for m in gal_methods:
            s = scores[f"{key}/{m}"]; order = np.argsort(-s)
            gallery(ds, meta, order, s, f"{tag}: top anomalies in {key} by {m} (each galaxy shown as HSC | LSST)",
                    figs / f"top16_{key.replace('/', '_')}_{m}.png")

    # 2) physics vs instrument: overlap of top-1% sets and a scatter, per survey
    top_frac = 0.01; k = max(1, int(n * top_frac)); summary["top_frac"] = top_frac
    summary["overlap_top1pct"] = {}
    fig, axes = plt.subplots(1, len(views), figsize=(5.5 * len(views), 4.8), squeeze=False); axes = axes[0]
    for ax, sv in zip(axes, views):
        sp, si = scores[f"phys/{sv}_mean/{PM}"], scores[f"ins/{sv}_flat/{PM}"]
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
    # Spearman for every flow variant present (nsf; cnsf and ccal when conditioning is on),
    # so the summary can show how much conditioning / calibration removed the depth trend
    flow_methods = [m for m in dict.fromkeys(("nsf", "cnsf", "ccal", "cemp", PM)) if f"ins/lsst_flat/{m}" in scores]
    for m in flow_methods:
        for key in (f"phys/lsst_mean/{m}", f"ins/lsst_flat/{m}"):
            for col in ("n_input_i", "psf_fwhm_i"):
                summary["spearman_vs_lsst_meta"][f"{key} vs {col}"] = float(spearmanr(meta[col], scores[key], nan_policy="omit").statistic)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, (key, lab) in enumerate(((f"phys/lsst_mean/{PM}", "physics (LSST image)"), (f"ins/lsst_flat/{PM}", "instrument (LSST image)"))):
        s = scores[key]
        for j, (col, xl) in enumerate((("n_input_i", "n_input_i (exposures in the i coadd)"), ("psf_fwhm_i", "psf_fwhm_i [″]"))):
            rho = summary["spearman_vs_lsst_meta"][f"{key} vs {col}"]
            ax = axes[i, j]; ax.scatter(meta[col], s, s=3, alpha=0.4); ax.set_xlabel(xl); ax.set_ylabel(f"{lab} {PM.upper()} score")
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
    summary["heldout_minus_fit_median_nsf"] = {k: float(np.median(scores[f"{k}/{PM}"][held]) - np.median(scores[f"{k}/{PM}"][~held])) for k in gkeys} if held.any() else {}

    # 5b) conditional vs unconditional flow: what did conditioning on exposures remove?
    if PM != "nsf":
        summary["cond_vs_uncond"] = {}
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (key, lab) in zip(axes, (("phys/lsst_mean", "physics (LSST image)"), ("ins/lsst_flat", "instrument (LSST image)"))):
            su, sc_ = scores[f"{key}/nsf"], scores[f"{key}/{PM}"]
            kk = max(1, int(0.01 * n)); tu, tc = set(np.argsort(-su)[:kk]), set(np.argsort(-sc_)[:kk])
            rho = spearmanr(su, sc_).statistic
            summary["cond_vs_uncond"][key] = {"spearman_scores": float(rho), "top1pct_overlap": len(tu & tc), "k": kk,
                                              "spearman_uncond_vs_n_input_i": summary["spearman_vs_lsst_meta"][f"{key}/nsf vs n_input_i"],
                                              "spearman_cond_vs_n_input_i": summary["spearman_vs_lsst_meta"][f"{key}/{PM} vs n_input_i"]}
            im = ax.scatter(su, sc_, s=3, c=meta["n_input_i"], cmap="viridis", alpha=0.6); plt.colorbar(im, ax=ax, label="n_input_i")
            ax.set_xlabel("unconditional NSF score  −log p(z)"); ax.set_ylabel(f"{PM.upper()} score  {PM_LABEL.get(PM, PM)}", fontsize=8.5)
            ax.set_title(f"{lab}: ρ(scores)={rho:+.2f}, top-1% overlap {len(tu & tc)}/{kk}\n"
                         f"ρ vs n_input_i: uncond {summary['cond_vs_uncond'][key]['spearman_uncond_vs_n_input_i']:+.2f} → cond "
                         f"{summary['cond_vs_uncond'][key]['spearman_cond_vs_n_input_i']:+.2f}", fontsize=9.5)
        fig.suptitle(f"{tag}: conditioning the flow on exposures per band — depth-driven anomalies should leave the ranking", fontsize=9.5)
        fig.tight_layout(); fig.savefig(figs / "cond_vs_uncond.png", dpi=130); plt.close(fig)

    if "ins/lsst_flat/cemp" in scores and "ins/lsst_flat/ccal" in scores:
        summary["ccal_vs_cemp"] = {key: {"spearman": float(spearmanr(scores[f"{key}/ccal"], scores[f"{key}/cemp"]).statistic),
                                         "top1pct_overlap": len(set(np.argsort(-scores[f"{key}/ccal"])[:max(1, int(0.01 * n))]) &
                                                                set(np.argsort(-scores[f"{key}/cemp"])[:max(1, int(0.01 * n))]))}
                                   for key in ("phys/lsst_mean", "ins/lsst_flat")}

    # 5c) depth calibration: does the score still track depth INSIDE the fit set?
    #     Median score per n_input_i bin over fit rows for every flow variant
    #     (raw conditional NLL slopes with depth because the conditional entropy
    #     does; the tail-probability score should be flat at −log10 0.5), plus
    #     the p-value histograms per group (uniform for fit rows if the flow is right).
    if len(flow_methods) > 1:
        groups_ = np.asarray(meta.get("group", np.array(["all"] * n)))
        edges = np.array([0, 5, 10, 20, 40, 60, 80, 100, 200]); fitm = meta["fit_mask"]
        b = np.clip(np.digitize(meta["n_input_i"], edges) - 1, 0, len(edges) - 2)
        centers = [f"{edges[i]}–{edges[i + 1]}" for i in range(len(edges) - 1)]
        ncol = len(flow_methods) + (1 if calib else 0)
        fig, axes = plt.subplots(2, ncol, figsize=(4.4 * ncol, 8), squeeze=False)
        summary["fit_rows_median_score_by_depth_bin"] = {}
        for r, (key, lab) in enumerate((("phys/lsst_mean", "physics (LSST image)"), ("ins/lsst_flat", "instrument (LSST image)"))):
            for c, m in enumerate(flow_methods):
                s = scores[f"{key}/{m}"]; ax = axes[r, c]
                meds, cnts = [], []
                for i in range(len(edges) - 1):
                    sel = fitm & (b == i)
                    meds.append(float(np.median(s[sel])) if sel.sum() >= 20 else np.nan); cnts.append(int(sel.sum()))
                summary["fit_rows_median_score_by_depth_bin"][f"{key}/{m}"] = dict(zip(centers, meds))
                ax.plot(range(len(meds)), meds, "o-", color="#2a78d6")
                for i, (mv, cn) in enumerate(zip(meds, cnts)):
                    if np.isfinite(mv): ax.annotate(f"n={cn:,}", (i, mv), xytext=(0, 6), textcoords="offset points", fontsize=6.5, ha="center")
                if m in ("ccal", "cemp"):   # honest y-range: flatness should look flat, not auto-zoomed noise
                    ax.axhline(-np.log10(0.5), color="0.5", ls="--", lw=1, label="−log10 0.5 (calibrated median)"); ax.legend(fontsize=7, frameon=False)
                    ax.set_ylim(0, max(0.8, 1.3 * np.nanmax(meds)))
                ax.set_xticks(range(len(centers))); ax.set_xticklabels(centers, rotation=45, fontsize=7)
                ax.set_xlabel("n_input_i bin (fit rows only)"); ax.set_ylabel(f"median {m.upper()} score"); ax.set_title(f"{lab}: {m}", fontsize=9.5)
            if calib and key in calib:
                ax = axes[r, ncol - 1]; p = calib[key]["p"]
                bins = np.linspace(0, 1, 21)
                for g in [g for g in GROUP_ORDER if (groups_ == g).any()] or ["all"]:
                    sel = (groups_ == g) if g != "all" else np.ones(n, bool)
                    ax.hist(p[sel], bins=bins, histtype="step", lw=1.5, density=True, color=GROUP_COLORS.get(g), label=f"{g} (n={int(sel.sum()):,})")
                ax.axhline(1, color="0.5", ls="--", lw=1); ax.set_ylim(0, None)
                ax.set_xlabel("tail probability p under p(z | c)"); ax.set_ylabel("density"); ax.set_title(f"{lab}: p-value calibration (uniform = calibrated)", fontsize=9); ax.legend(fontsize=7, frameon=False)
        fig.suptitle(f"{tag}: does the score still track depth inside the fit set? (raw conditional NLL follows the conditional entropy; ccal should be flat)", fontsize=9)
        fig.tight_layout(); fig.savefig(figs / "depth_calibration.png", dpi=130); plt.close(fig)

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
        # who fills the top 1 %? A group over-represented relative to its population share is either
        # genuinely unusual or (for a same-distribution held-out slice) a sign the flow memorised its fit rows
        kk = max(1, int(0.01 * n))
        summary["group_share"] = {g: float(np.mean(groups == g)) for g in gnames}
        summary["top1pct_group_share"] = {}
        for key in (f"phys/lsst_mean/{PM}", f"ins/lsst_flat/{PM}"):
            top = np.argsort(-scores[key])[:kk]
            for g in gnames:
                summary["top1pct_group_share"][f"{key} | {g}"] = float(np.mean(groups[top] == g))
        summary["fit_group"] = ref_group
        summary["group_median_nsf_minus_fit_group"] = {}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, (key, lab) in zip(axes, ((f"phys/lsst_mean/{PM}", "physics space (encoder_1, LSST image)"),
                                        (f"ins/lsst_flat/{PM}", "instrument space (encoder_2, LSST image)"))):
            s = scores[key]; ref = np.median(s[groups == ref_group])
            lo, hi = np.percentile(s, [0.5, 99.5]); bins = np.linspace(lo, hi, 80)
            for g in gnames:
                ax.hist(s[groups == g], bins=bins, histtype="step", lw=1.6, density=True, color=GROUP_COLORS.get(g), label=f"{g} (n={int((groups == g).sum()):,})")
                summary["group_median_nsf_minus_fit_group"][f"{key} | {g}"] = float(np.median(s[groups == g]) - ref)
            ax.set_xlabel(f"{PM.upper()} anomaly score  {PM_LABEL.get(PM, PM)}\n(fitted on {ref_group})", fontsize=8.5); ax.set_ylabel("density"); ax.set_title(lab, fontsize=10)
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f"{tag}: score distributions per group (detectors fitted on {ref_group})", fontsize=9.5)
        fig.tight_layout(); fig.savefig(figs / "group_score_distributions.png", dpi=130); plt.close(fig)
        # per-group galleries for the two LSST-view spaces
        for g in gnames:
            if g == ref_group:
                continue
            sel = np.where(groups == g)[0]
            for key in ("phys/lsst_mean", "ins/lsst_flat"):
                s = scores[f"{key}/{PM}"]; order = sel[np.argsort(-s[sel])]
                gallery(ds, meta, order, s, f"{tag}: top anomalies within {g} in {key} by {PM} (HSC | LSST)",
                        figs / f"top16_{g}_{key.replace('/', '_')}_{PM}.png")
    return summary


# --------------------------------------------------------------------- main

def finish(args, ds, meta, latents, scores, calib, nsf_nll, out, views, n, fit_mask, t0):
    """Report + UMAPs + summary.json + Discord. Shared by the full pipeline and
    --report-only (which re-reports saved latents/scores under a different
    primary score, e.g. the empirical depth-calibrated `cemp`)."""
    summary = report(ds, meta, scores, out, args.tag, views=views, calib=calib or None)
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
        parts = [f"{g}: physics {gm.get(f'phys/lsst_mean/{PM} | {g}', float('nan')):+.2f} | instrument {gm.get(f'ins/lsst_flat/{PM} | {g}', float('nan')):+.2f}"
                 for g in summary["group_sizes"] if g != ref]
        shift = f"median {PM.upper()} score shift vs {ref} — " + "; ".join(parts) + "\n"
        ts = summary.get("top1pct_group_share", {}); gs = summary.get("group_share", {})
        shift += ("top-1% composition, instrument ({}): ".format(PM) +
                  "; ".join(f"{g} {100 * ts.get(f'ins/lsst_flat/{PM} | {g}', 0):.0f}% (pop {100 * gs.get(g, 0):.0f}%)" for g in summary["group_sizes"]) + "\n")
    ovl = "; ".join(f"{sv.upper()} {ov[sv]['overlap']}/{ov[sv]['k']}" for sv in views)
    scope_txt = f"{args.scope}{'/' + args.field if args.field else ''}, views={'+'.join(views)}"
    cond_txt = ("" if PM == "nsf" else
                (f"**conditional flow p(z | log n_input g,r,i,z)**" + (f", **depth-calibrated tail probability** (K={args.calibrate} samples/row)" if PM == "ccal" else "")
                 + (", **empirical depth-calibrated percentile (cemp)**: rank of the out-of-fold conditional NLL among fit rows with the same exposure pattern" if PM == "cemp" else "")
                 + " is the primary score; unconditional kept for contrast\n"
                 f"Spearman vs n_input_i, UNconditional: physics {sp['phys/lsst_mean/nsf vs n_input_i']:+.3f} | instrument {sp['ins/lsst_flat/nsf vs n_input_i']:+.3f}\n")
                + (f"Spearman vs n_input_i, raw conditional NLL: physics {sp['phys/lsst_mean/cnsf vs n_input_i']:+.3f} | instrument {sp['ins/lsst_flat/cnsf vs n_input_i']:+.3f}\n" if PM in ("ccal", "cemp") else "")
                + (f"Spearman vs n_input_i, model-calibrated ccal: physics {sp['phys/lsst_mean/ccal vs n_input_i']:+.3f} | instrument {sp['ins/lsst_flat/ccal vs n_input_i']:+.3f}\n" if PM == "cemp" and "phys/lsst_mean/ccal vs n_input_i" in sp else ""))
    scope_txt += f", {args.folds}-fold out-of-fold flows" if args.folds > 1 else ""
    text = (f"**HSC × LSST anomaly stage — `{args.tag}`** ({scope_txt}) done in {summary['wall_min']} min, N={n:,} LSST rows, "
            f"detectors fitted on {int(fit_mask.sum()):,} rows ({summary.get('fit_group', 'fit set')})\n"
            f"ckpt: `{Path(args.checkpoint).parent.parent.name}/{Path(args.checkpoint).name}`\n" + cond_txt + shift +
            f"top-1% overlap physics∩instrument ({PM}): {ovl} (random ≈ {ov['lsst']['expected_random']:.1f})\n"
            f"Spearman vs n_input_i ({PM}): physics {sp[f'phys/lsst_mean/{PM} vs n_input_i']:+.3f} | instrument {sp[f'ins/lsst_flat/{PM} vs n_input_i']:+.3f}\n"
            f"Spearman vs psf_fwhm_i ({PM}): physics {sp[f'phys/lsst_mean/{PM} vs psf_fwhm_i']:+.3f} | instrument {sp[f'ins/lsst_flat/{PM} vs psf_fwhm_i']:+.3f}\n"
            f"outputs: `{out}`")
    batch1 = [figs / f"top16_phys_lsst_mean_{PM}.png", figs / f"top16_ins_lsst_flat_{PM}.png",
              figs / "phys_vs_ins_scatter.png", figs / "score_vs_depth_psf.png", figs / "cond_vs_uncond.png", figs / "depth_calibration.png"]
    batch2 = umap_paths + [figs / f"top16_phys_hsc_mean_{PM}.png", figs / f"top16_ins_hsc_flat_{PM}.png", figs / "score_correlation.png"]
    ok1 = discord_post(text, [str(p) for p in batch1 if Path(p).exists()])
    ok2 = discord_post(f"`{args.tag}` — UMAPs with the top-{TOP_K_UMAP} anomalies marked, HSC-view galleries, score correlations",
                       [str(p) for p in batch2 if Path(p).exists()][:10])
    ok3 = True
    if (figs / "group_score_distributions.png").exists():
        batch3 = [figs / "group_score_distributions.png"] + sorted(p for p in figs.glob(f"top16_*_{PM}.png")
                                                                    if p.name.split("_")[1] in ("deep23", "cosmos") and not p.name.startswith(("top16_phys", "top16_ins")))
        ok3 = discord_post(f"`{args.tag}` — score distributions per group (fitted on {summary.get('fit_group')}) and within-group galleries",
                           [str(p) for p in batch3 if Path(p).exists()][:10])
    print(f"[notify] discord: summary {'sent' if ok1 else 'FAILED'}, figures {'sent' if ok2 else 'FAILED'}, groups {'sent' if ok3 else 'FAILED'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None, help="model checkpoint (required unless --report-only)")
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
    ap.add_argument("--cond", choices=("none", "n_input", "n_input_psf"), default="none",
                    help="n_input: also fit a CONDITIONAL flow p(z | log n_input in g,r,i,z) and use it as the primary "
                         "score (<key>/cnsf) — objects anomalous for their own number of visits. n_input_psf adds "
                         "psf_fwhm per band to the context. The unconditional flow is still fitted for comparison.")
    ap.add_argument("--cond-epochs", type=int, default=None, help="epochs for the conditional flow (default: same as --nsf-epochs)")
    ap.add_argument("--calibrate", type=int, default=0, metavar="K",
                    help="with --cond: make the primary score the depth-CALIBRATED tail probability <key>/ccal, estimated "
                         "with K flow samples per context centroid (see calibrated_scores; also writes the empirical "
                         "within-cluster percentile <key>/cemp). Needed because −log p(z | c) still ranks shallow stamps "
                         "high — it tracks the conditional entropy at each depth, not how unusual the object is for its "
                         "depth. K=1024 gives a floor of p = 1/1025 (top ~0.1%%).")
    ap.add_argument("--calib-centroids", type=int, default=256, help="k-means clusters over the context for --calibrate")
    ap.add_argument("--folds", type=int, default=1,
                    help="out-of-fold scoring of both flows with this many seeded folds (see score_all): every row is "
                         "scored by a flow that did not see it, so memorisation cannot push held-out rows to the top. "
                         "1 = single in-sample fit (paper recipe).")
    ap.add_argument("--headline-only", action="store_true",
                    help="fit flows only for the headline representations (physics token-mean, instrument flat) of "
                         "the requested views; the others are still encoded and saved. Halves the flow time with --folds.")
    ap.add_argument("--balance", action="store_true",
                    help="weight the conditional flow's fit by the inverse density of the context (log n_input_i), so the "
                         "sparse shallow depths are learned rather than drowned by the 75-105 exposure bulk")
    ap.add_argument("--fit-on", choices=("cosmos_train", "both_fields"), default="cosmos_train",
                    help="--scope lsst_all only. cosmos_train: detectors fitted on the COSMOS train split (DEEP2-3 is an "
                         "OOD probe). both_fields: fitted on COSMOS train + a seeded 80%% slice of DEEP2-3, so the "
                         "conditioning variable is covered from 1 to 114 exposures; groups still reported separately.")
    ap.add_argument("--primary", choices=("auto", "nsf", "cnsf", "ccal", "cemp"), default="auto",
                    help="which flow score drives galleries / UMAP markers / overlaps / the Discord headline. auto = "
                         "ccal if --calibrate, else cnsf if --cond, else nsf. cemp (empirical within-context-cluster "
                         "percentile of the out-of-fold conditional NLL) is the recommended final ranking (D12).")
    ap.add_argument("--report-only", metavar="SRC_TAG", default=None,
                    help="skip encoding and scoring: load latents.h5 + scores.h5 of an existing run <SRC_TAG> and "
                         "re-run report/UMAP/Discord into --tag (a new dir), typically with --primary cemp. Needs no GPU.")
    args = ap.parse_args()
    if args.report_only is None and not args.checkpoint:
        ap.error("--checkpoint is required unless --report-only is given")
    if args.scope == "field" and not args.field:
        ap.error("--scope field needs --field COSMOS|DEEP2-3")
    views = ("hsc", "lsst") if args.views == "both" else ("lsst",)
    if args.calibrate and args.cond == "none":
        ap.error("--calibrate needs --cond n_input|n_input_psf (it calibrates the conditional flow)")
    global PM
    PM = "ccal" if (args.cond != "none" and args.calibrate) else ("cnsf" if args.cond != "none" else "nsf")
    if args.primary != "auto":
        PM = args.primary

    out = OUT_ROOT / args.tag
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pick another --tag or pass --force (no silent overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    if args.report_only:
        # ------------------------------------------------ re-report a finished run under another primary score
        srcdir = OUT_ROOT / args.report_only
        latents, meta, scores, calib = {}, {}, {}, {}
        with h5py.File(srcdir / "latents.h5") as f:
            def _lat(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if name.startswith("meta/"):
                        v = obj[:]; meta[name[5:]] = v.astype(str) if v.dtype.kind == "S" else v
                    else:
                        latents[name] = obj[:]
            f.visititems(_lat); lattrs = dict(f.attrs)
        with h5py.File(srcdir / "scores.h5") as f:
            def _sc(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if name.startswith("calib/"):
                        _, enc, rep_, what = name.split("/"); calib.setdefault(f"{enc}/{rep_}", {})[what] = obj[:]
                    elif name not in ("is_val", "context"):
                        scores[name] = obj[:]
            f.visititems(_sc); sattrs = dict(f.attrs)
        views = tuple(str(lattrs["views"]).split(","))
        n = len(meta["is_val"]); fit_mask = meta["fit_mask"].astype(bool); meta["is_val"] = meta["is_val"].astype(bool); meta["fit_mask"] = fit_mask
        # scope / field / recipe of the source run, so the Discord text and dataset match it
        args.scope = str(lattrs["scope"]); args.field = str(lattrs["field"]) or None
        args.checkpoint = str(lattrs["checkpoint"]); args.cond = str(sattrs.get("cond", "none"))
        args.calibrate = int(sattrs.get("calibrate_samples", 0)); args.folds = int(sattrs.get("folds", 1))
        if f"ins/lsst_flat/{PM}" not in scores:
            raise SystemExit(f"--primary {PM} not in {args.report_only}: available {sorted(k.split('/')[-1] for k in scores if k.startswith('ins/lsst_flat/'))}")
        try:
            nsf_nll = json.load(open(srcdir / "summary.json")).get("nsf_train_nll")
        except FileNotFoundError:
            nsf_nll = None
        if args.scope == "anchors":
            ds = HscLsstRAM48Dataset()
        elif args.scope == "lsst_all":
            ds = HscLsstRAM48Dataset(field=None, min_n_input=0)
        else:
            ds = HscLsstRAM48Dataset(field=args.field, min_n_input=0)
        assert np.array_equal(ds.anchor_rows[:n], meta["lsst_row"]), "dataset rows do not match the saved run"
        print(f"[report-only] {args.report_only} -> {out}: primary {PM}, {n:,} rows, views {views}, "
              f"scores {sorted(set(k.split('/')[-1] for k in scores))}", flush=True)
        with open(out / "REPORT_ONLY.txt", "w") as f:
            f.write(f"re-report of {srcdir} with --primary {PM}; latents/scores live there, not here\n")
        finish(args, ds, meta, latents, scores, calib, nsf_nll, out, views, n, fit_mask, t0)
        return

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
        if args.scope == "lsst_all" and args.fit_on == "both_fields":
            # pool a seeded 80 % slice of DEEP2-3 into the fit set; its other 20 % is a held-out group
            d_pos = np.where(field == "DEEP2-3")[0]
            d_held = train_val_positions(len(d_pos), val_ratio=0.2, seed=SPLIT_SEED)
            fit_mask[d_pos[~d_held]] = True
            group = np.array(group, dtype=object)
            group[d_pos[~d_held]] = "deep23_fit"; group[d_pos[d_held]] = "deep23_held"
            group = group.astype(str)
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

    ctx = build_context(meta, fit_mask, with_psf=(args.cond == "n_input_psf")) if args.cond != "none" else None
    if args.headline_only:
        latents = {k: v for k, v in latents.items() if k in gallery_keys(views) or k.split("/")[1].startswith("joint")}
        print(f"[score] --headline-only: flows for {sorted(latents)}", flush=True)
    scores, nsf_nll, calib = score_all(latents, fit_mask, device, args.nsf_epochs, ctx=ctx,
                                       cond_epochs=args.cond_epochs, balance=args.balance, calibrate=args.calibrate,
                                       calib_centroids=args.calib_centroids, folds=args.folds)
    with h5py.File(out / "scores.h5", "w") as f:
        for k, v in scores.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        f.create_dataset("is_val", data=is_val)
        if ctx is not None:
            f.create_dataset("context", data=ctx)
        for k, v in calib.items():                       # tail probabilities, z-scores, context clusters behind <key>/ccal and /cemp
            for kk, vv in v.items():
                f.create_dataset(f"calib/{k}/{kk}", data=vv)
        f.attrs.update({"checkpoint": args.checkpoint, "tag": args.tag, "nsf_epochs": args.nsf_epochs,
                        "methods": list(METHODS) + (["cnsf"] if ctx is not None else []) + (["ccal", "cemp"] if calib else []), "primary_method": PM,
                        "cond": args.cond, "fit_on": args.fit_on, "cond_epochs": args.cond_epochs or args.nsf_epochs,
                        "context_balanced": args.balance, "calibrate_samples": args.calibrate, "calib_centroids": args.calib_centroids,
                        "folds": args.folds, "headline_only": args.headline_only,
                        "context_desc": ("standardised log1p(n_input) i,g,r,z" + (" + psf_fwhm g,r,i,z" if args.cond == "n_input_psf" else "")) if ctx is not None else ""})
    df = pd.DataFrame({k.replace("/", "__"): v for k, v in scores.items()})
    for k in ("object_id_hsc", "lsst_row", "parent_row", "is_val", "fit_mask", "field", "group", "n_input_i", "psf_fwhm_i"):
        df[k] = meta[k]
    df.to_parquet(out / "scores.parquet", index=False)

    finish(args, ds, meta, latents, scores, calib, nsf_nll, out, views, n, fit_mask, t0)

if __name__ == "__main__":
    main()
