"""Post-training anomaly stage for the LSST models (AE / FM-AE / JEPA).

Encodes every grizy object (64,918), then ranks anomalies:
  1. NSF        — zuko neural spline flow on latents (the run_flow_only.py
                  recipe), fit on TRAIN latents only; score = -log p.
  2. recon      — per-object reconstruction MSE (only for models whose
                  forward(x) returns (xhat, z); skipped for JEPA/FM-AE).
  3. iforest    — sklearn IsolationForest on latents, fit on train.

Generalizations for the experiment matrix (2026-09-02):
  - model.encode(x) is used when present (JEPA/FM-AE); else forward.
  - train_field: NSF/IForest are fit on train-split latents of the model's
    own training field, and rankings/galleries are restricted to that field
    (scores for ALL objects are still written to scores.parquet).
  - gallery_recon_fn(model, x) -> xhat: optional expensive reconstruction
    (e.g. FM ODE sampling) used only for the top-K gallery rows.

Writes latents.h5, scores.parquet, and a top-16 gallery PNG per method into
the run dir, then posts a summary + galleries to Discord.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import FIELDS, LsstGrizyDataset
from .notify import discord_post

TOPK_GALLERY = 16
TOPK_OVERLAP = 100


@torch.no_grad()
def encode_all(model, device, batch_size=512, workers=8):
    ds = LsstGrizyDataset("all")
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                        shuffle=False)
    has_recon = not hasattr(model, "encode")
    zs, errs = [], []
    model.eval().to(device)
    for batch in loader:
        x = batch["x"].to(device)
        if has_recon:
            xhat, z = model(x)
            errs.append(torch.nn.functional.mse_loss(
                xhat, x, reduction="none").mean((1, 2, 3)).float().cpu())
        else:
            z = model.encode(x)
        zs.append(z.float().cpu())
    return ds, torch.cat(zs).numpy(), (torch.cat(errs).numpy()
                                       if has_recon else None)


def fit_nsf(z_train, z_all, device, steps=3000, batch=1024):
    import zuko
    flow = zuko.flows.NSF(features=z_all.shape[1], context=0, transforms=6,
                          hidden_features=[64, 64]).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    zt = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    for step in range(steps):
        idx = torch.randint(0, zt.shape[0], (batch,), device=device)
        loss = -flow().log_prob(zt[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"[nsf] step {step} nll {loss.item():.2f}", flush=True)
    nll = []
    za = torch.as_tensor(z_all, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(0, za.shape[0], 4096):
            nll.append((-flow().log_prob(za[i:i + 4096])).cpu())
    return torch.cat(nll).numpy()


def gallery(ds, model, device, order, scores, title, path,
            k=TOPK_GALLERY, recon_fn=None):
    """recon_fn: None -> use model(x) if the model reconstructs, else
    inputs-only gallery; callable -> recon_fn(model, x_batch) -> xhat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rgb(x5):
        r = np.stack([x5[2], x5[1], x5[0]], -1)
        return np.clip((r + 1.0) / 5.0, 0, 1)

    if recon_fn is None and not hasattr(model, "encode"):
        recon_fn = lambda m, x: m(x)[0]                     # plain AE
    per = 2 if recon_fn else 1
    cols = 8
    rows = int(np.ceil(k / cols)) * per
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.4 * rows),
                             layout="constrained", squeeze=False)
    model.eval().to(device)
    with torch.no_grad():
        idx = [int(i) for i in order[:k]]
        xs = torch.stack([ds[i]["x"] for i in idx])
        xhat = recon_fn(model, xs.to(device)).float().cpu() if recon_fn \
            else None
        for j, i in enumerate(idx):
            r0 = (j // cols) * per
            c = j % cols
            axes[r0, c].imshow(rgb(xs[j].numpy()), origin="lower")
            axes[r0, c].set_title(
                f"{FIELDS[ds.field[i]]}\n{ds.ids[i]}\n"
                f"score={scores[i]:.3g}", fontsize=6)
            if xhat is not None:
                axes[r0 + 1, c].imshow(rgb(xhat[j].numpy()), origin="lower")
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    sub = " (row pairs: input / recon)" if per == 2 else ""
    fig.suptitle(f"{title} — top {k}{sub}", fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_anomaly_stage(model, run_dir, device="cuda", train_field=None,
                      gallery_recon_fn=None):
    print("[anomaly] encoding all objects...", flush=True)
    ds, z, recon = encode_all(model, device)
    train_mask = ~ds.is_val
    own = (ds.field == FIELDS.index(train_field)) if train_field \
        else np.ones(len(ds), bool)
    fit_mask = train_mask & own

    mu, sd = z[fit_mask].mean(0), z[fit_mask].std(0) + 1e-8
    zs = (z - mu) / sd

    print(f"[anomaly] fitting NSF on {fit_mask.sum()} train latents"
          f" (field={train_field or 'all'})...", flush=True)
    nsf_nll = fit_nsf(zs[fit_mask], zs, device)

    print("[anomaly] fitting IsolationForest...", flush=True)
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=300, random_state=0, n_jobs=4)
    iso.fit(zs[fit_mask])
    iso_score = -iso.score_samples(zs)              # higher = more anomalous

    import h5py
    with h5py.File(os.path.join(run_dir, "latents.h5"), "w") as f:
        f["z"] = z
        f["object_id_hsc"] = ds.ids
        f["is_val"] = ds.is_val
        f["field"] = ds.field

    cols = {"object_id_hsc": ds.ids,
            "field": [FIELDS[i] for i in ds.field],
            "is_val": ds.is_val, "nsf_nll": nsf_nll, "iforest": iso_score}
    if recon is not None:
        cols["recon_mse"] = recon
    df = pd.DataFrame(cols)
    score_keys = (["recon_mse"] if recon is not None else []) \
        + ["nsf_nll", "iforest"]
    for col in score_keys:
        df[f"rank_{col}"] = (-df[col]).argsort().argsort() + 1
    df.attrs["train_field"] = train_field or "all"
    df.to_parquet(os.path.join(run_dir, "scores.parquet"), index=False)

    methods = {"NSF (latent log-prob)": ("nsf_nll", nsf_nll),
               "IsolationForest (latents)": ("iforest", iso_score)}
    if recon is not None:
        methods["Reconstruction MSE"] = ("recon_mse", recon)
    pngs, tops = [], {}
    own_idx = np.where(own)[0]
    for title, (key, s) in methods.items():
        order = own_idx[np.argsort(-s[own])]        # own-field ranking
        tops[key] = set(order[:TOPK_OVERLAP])
        path = os.path.join(run_dir, f"anomalies_{key}.png")
        gallery(ds, model, device, order, s, title, path,
                recon_fn=gallery_recon_fn)
        pngs.append(path)

    keys = list(tops)
    olap = {f"{a.split('_')[0]}∩{b.split('_')[0]}":
            len(tops[a] & tops[b]) for i, a in enumerate(keys)
            for b in keys[i + 1:]}
    lines = [f"🧪 **anomaly report** ({os.path.basename(run_dir)}, "
             f"train_field={train_field or 'all'})"]
    if recon is not None:                           # per-field val recon —
        per_field = df[df.is_val].groupby("field")[  # cross-field matrix row
            "recon_mse"].mean()
        lines.append("val recon MSE (all fields encoded): " + ", ".join(
            f"{k} {v:.4f}" for k, v in per_field.items()))
    lines.append(f"top-{TOPK_OVERLAP} overlaps (own-field): {olap}")
    for title, (key, s) in methods.items():
        ids = ds.ids[own_idx[np.argsort(-s[own])][:10]]
        lines.append(f"**{title}** top-10: " +
                     ", ".join(str(i) for i in ids))
    discord_post("\n".join(lines), pngs)
    print("[anomaly] done; report posted.", flush=True)
