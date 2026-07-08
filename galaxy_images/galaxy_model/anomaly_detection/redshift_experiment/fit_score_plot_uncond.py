"""
Job A — unconditional NSF anomaly detection on the Ours-Physics latent.

Reads outputs/latents_redshift.h5, fits an unconditional NSF (redshift-agnostic),
scores every galaxy (-log_prob), and produces four figures posted to Discord:
  A1: top-32 anomaly grid (paper-figure display path: 160x160 arcsinh-only)
  A2: 5-panel physics-property distributions with top-32 rug
  A3: top-32 after dropping lowest-25% redshift (filter-then-rank)
  A4: middle-rank 32 sanity-check grid (median-anomaly cohort)

Run:
  python anomaly_detection/redshift_experiment/fit_score_plot_uncond.py \
    [--latents .../outputs/latents_redshift.h5] [--top-n 32] [--nsf-epochs 50] \
    [--webhook URL] [--device cuda]
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
    """Return {colname: (N,) array} for whichever props/* keys actually exist."""
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
    p.add_argument("--nsf-epochs", type=int, default=50)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--webhook", default=None)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.latents, "r") as f:
        hsc_mean = f["hsc_mean"][:]
        desi_z = f["desi_z"][:]
        record_idx = f["record_idx"][:]
        prop_keys = list(common.PROPERTY_AXES.keys() - {"desi_z"})  # other 4
        props = _load_props(f, prop_keys)
    props["desi_z"] = desi_z  # include redshift in the multi-property panel
    n, d = hsc_mean.shape
    print(f"Loaded latents N={n} D={d}  props keys: {list(props.keys())}")

    # --- fit unconditional NSF, score everyone ---
    train_idx, _ = common.train_test_split(n, args.train_frac)
    print(f"Fitting unconditional NSF (epochs={args.nsf_epochs}, n_train={len(train_idx)})...")
    scores = common.score_nsf(hsc_mean[train_idx], hsc_mean, args.nsf_epochs, args.device)
    np.save(args.out_dir / "scores_uncond.npy", scores)

    tag = "Ours-Physics NSF (unconditional)"

    # --- (A1) top-N anomaly grid using the paper-figure display path ---
    order, _, top_pcts = common.top_n_with_percentiles(scores, args.top_n)
    top_records = record_idx[order]
    imgs = common.load_hsc_images_for_display(args.images_bin, top_records)
    grid_path = common.plot_anomaly_grid(
        imgs, ranks=np.arange(1, len(order) + 1), z_vals=desi_z[order], pcts=top_pcts,
        title=f"Top {len(order)} anomalies — {tag}", out_path=args.out_dir / "A1_top_anomalies.png")
    common.discord_notify(args.webhook,
                          f"🅰️ **Job A — {tag}**\nA1: top {len(order)} anomalies "
                          f"(label: rank / z / NLL percentile; display = 160×160 paper-figure path).",
                          file_path=grid_path)

    # --- (A2) 5-panel physics-property distributions ---
    prop_path = common.plot_property_distributions(
        props, top_idx=order, title=f"Physics-property distributions — {tag}",
        out_path=args.out_dir / "A2_property_distributions.png")
    common.discord_notify(args.webhook,
                          f"🅰️ A2: full-sample distributions for desi_z / logMstar / t_age / "
                          f"Z_met / SFR (top {len(order)} anomalies overplotted as red rug).",
                          file_path=prop_path)

    # --- (A3) drop lowest-25% redshift, re-rank by the same scores ---
    z25 = float(np.percentile(desi_z, 25))
    keep = desi_z >= z25
    kept_pos = np.where(keep)[0]
    sub_order_within, _, sub_pcts = common.top_n_with_percentiles(scores[kept_pos], args.top_n)
    global_pos = kept_pos[sub_order_within]
    imgs_f = common.load_hsc_images_for_display(args.images_bin, record_idx[global_pos])
    filt_path = common.plot_anomaly_grid(
        imgs_f, ranks=np.arange(1, len(global_pos) + 1), z_vals=desi_z[global_pos], pcts=sub_pcts,
        title=f"Top {len(global_pos)} anomalies, z≥{z25:.3f} (lowest-25% removed) — {tag}",
        out_path=args.out_dir / "A3_top_anomalies_zfiltered.png")
    common.discord_notify(args.webhook,
                          f"🅰️ A3: top {len(global_pos)} anomalies after removing the lowest-25% "
                          f"redshift (z < {z25:.3f}); {int(keep.sum())} galaxies remain.",
                          file_path=filt_path)

    # --- (A4) middle-rank-32 sanity-check grid ---
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
        out_path=args.out_dir / "A4_mid_rank_anomalies.png")
    common.discord_notify(args.webhook,
                          f"🅰️ A4: middle-rank sanity check (ranks {mid_start+1}–"
                          f"{mid_start+len(mid_order)}) — should look TYPICAL, not anomalous.",
                          file_path=mid_path)

    # --- (A5) bottom-rank-32 sanity (lowest NLL = MOST typical) ---
    bot_order, _, bot_pcts = common.bottom_n_with_percentiles(scores, args.top_n)
    bot_ranks = np.arange(1, len(bot_order) + 1)  # 1 = most-typical
    imgs_b = common.load_hsc_images_for_display(args.images_bin, record_idx[bot_order])
    bot_path = common.plot_anomaly_grid(
        imgs_b, ranks=bot_ranks, z_vals=desi_z[bot_order], pcts=bot_pcts,
        title=f"Bottom-{len(bot_order)} (LOWEST NLL = most-typical) — {tag}",
        out_path=args.out_dir / "A5_bottom_rank_anomalies.png")
    common.discord_notify(args.webhook,
                          f"🅰️ A5: bottom-{len(bot_order)} (lowest NLL = MOST TYPICAL galaxies). "
                          f"Should look like the bulk of HSC galaxies (small/faint).",
                          file_path=bot_path)

    print("Job A done.")


if __name__ == "__main__":
    main()
