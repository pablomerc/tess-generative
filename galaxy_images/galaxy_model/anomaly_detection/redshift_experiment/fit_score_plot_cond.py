"""
Job B — conditional NSF anomaly detection: p(physics latent | redshift).

v2: the v1 cond NSF was confirmed to be ignoring its context (Fix 4a diagnostic:
Spearman with the unconditional flow = 0.94 / top-32 overlap = 29/32 despite
Ridge R²(z | latent) = 0.59). v2 uses the WIDE profile (transforms=8,
hidden_features=[128,128]) and a 2-D context [z_std, z_std**2] with 100 epochs +
cosine LR. The bare-z + small-hyper-net combo from v1 wasn't enough.

Posts to Discord:
  B1: top-32 conditional anomalies (display = 160×160 paper-figure path)
  B2: 5-panel physics-property distributions with top-32 rug
  B3: middle-rank 32 sanity-check grid
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import common  # noqa: E402

_HERE = Path(__file__).resolve().parent
OUTPUT_DIR = _HERE / "outputs"


def _load_props(h5, working_keys):
    out = {}
    if "props" in h5:
        for k in working_keys:
            if k in h5["props"]:
                out[k] = h5["props"][k][:]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, default=OUTPUT_DIR / "latents_redshift.h5")
    p.add_argument("--images-bin", type=Path, default=common.DEFAULT_IMAGES_BIN)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--top-n", type=int, default=32)
    p.add_argument("--nsf-epochs", type=int, default=100,
                   help="More epochs than v1 (50); wide cond flow benefits from it.")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--profile", default="wide", choices=list(common.NSF_PROFILES.keys()),
                   help="NSF capacity profile (default=wide for the cond flow).")
    p.add_argument("--z-context", default="z_z2", choices=["z", "z_z2"],
                   help="z-context dimensionality. v1 used 'z'; v2 default 'z_z2' restores coupling.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--webhook", default=None)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.latents, "r") as f:
        hsc_mean = f["hsc_mean"][:]
        desi_z = f["desi_z"][:]
        record_idx = f["record_idx"][:]
        prop_keys = list(common.PROPERTY_AXES.keys() - {"desi_z"})
        props = _load_props(f, prop_keys)
    props["desi_z"] = desi_z
    n, d = hsc_mean.shape
    print(f"Loaded latents N={n} D={d}  props keys: {list(props.keys())}")

    # Train/test split; build (N, C) z-context using train-split statistics.
    train_idx, _ = common.train_test_split(n, args.train_frac)
    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    z_ctx = common.make_z_context(desi_z, train_mask, mode=args.z_context)

    print(f"Fitting conditional NSF (profile={args.profile}, ctx={z_ctx.shape[1]}, "
          f"epochs={args.nsf_epochs}, n_train={len(train_idx)})...")
    scores = common.score_nsf(
        hsc_mean[train_idx], hsc_mean, args.nsf_epochs, args.device,
        train_c=z_ctx[train_idx], all_c=z_ctx,
        profile=args.profile, cosine_lr=True)
    np.save(args.out_dir / "scores_cond.npy", scores)

    tag = (f"Ours-Physics NSF (conditional on z; profile={args.profile}, ctx={args.z_context}, "
           f"epochs={args.nsf_epochs})")

    # --- (B1) top-N grid ---
    order, _, top_pcts = common.top_n_with_percentiles(scores, args.top_n)
    imgs = common.load_hsc_images_for_display(args.images_bin, record_idx[order])
    grid_path = common.plot_anomaly_grid(
        imgs, ranks=np.arange(1, len(order) + 1), z_vals=desi_z[order], pcts=top_pcts,
        title=f"Top {len(order)} anomalies — {tag}", out_path=args.out_dir / "B1_top_anomalies.png")
    common.discord_notify(args.webhook,
                          f"🅱️ **Job B — {tag}**\nB1: top {len(order)} anomalies "
                          f"given their redshift.",
                          file_path=grid_path)

    # --- (B2) physics-property distributions with cond top-K ---
    prop_path = common.plot_property_distributions(
        props, top_idx=order, title=f"Physics-property distributions — {tag}",
        top_k_label="top cond anomalies",
        out_path=args.out_dir / "B2_property_distributions.png")
    common.discord_notify(args.webhook,
                          f"🅱️ B2: full-sample distributions across desi_z / logMstar / t_age / "
                          f"Z_met / SFR (top {len(order)} CONDITIONAL anomalies overplotted).",
                          file_path=prop_path)

    # --- (B3) middle-rank sanity grid ---
    finite_mask = np.isfinite(scores)
    full_order = np.argsort(np.where(finite_mask, scores, -np.inf))[::-1]
    n_fin = int(finite_mask.sum())
    mid_start = max(0, n_fin // 2 - args.top_n // 2)
    mid_order = full_order[mid_start: mid_start + args.top_n]
    mid_ranks = np.arange(mid_start + 1, mid_start + 1 + len(mid_order))
    sorted_finite = np.sort(scores[finite_mask])
    mid_pcts = np.array([np.searchsorted(sorted_finite, scores[i], side="left") /
                         len(sorted_finite) * 100.0 for i in mid_order])
    imgs_m = common.load_hsc_images_for_display(args.images_bin, record_idx[mid_order])
    mid_path = common.plot_anomaly_grid(
        imgs_m, ranks=mid_ranks, z_vals=desi_z[mid_order], pcts=mid_pcts,
        title=f"Middle-rank {len(mid_order)} sanity (ranks {mid_start+1}–{mid_start+len(mid_order)}) — {tag}",
        out_path=args.out_dir / "B3_mid_rank_anomalies.png")
    common.discord_notify(args.webhook,
                          f"🅱️ B3: middle-rank sanity check (ranks {mid_start+1}–"
                          f"{mid_start+len(mid_order)}) — should look TYPICAL.",
                          file_path=mid_path)

    # --- (B4) bottom-rank-32 sanity (lowest NLL = MOST typical) ---
    bot_order, _, bot_pcts = common.bottom_n_with_percentiles(scores, args.top_n)
    bot_ranks = np.arange(1, len(bot_order) + 1)
    imgs_b = common.load_hsc_images_for_display(args.images_bin, record_idx[bot_order])
    bot_path = common.plot_anomaly_grid(
        imgs_b, ranks=bot_ranks, z_vals=desi_z[bot_order], pcts=bot_pcts,
        title=f"Bottom-{len(bot_order)} cond NLL (most-typical given z) — {tag}",
        out_path=args.out_dir / "B4_bottom_rank_anomalies.png")
    common.discord_notify(args.webhook,
                          f"🅱️ B4: bottom-{len(bot_order)} (lowest cond NLL = most TYPICAL given z).",
                          file_path=bot_path)

    print("Job B done.")


if __name__ == "__main__":
    main()
