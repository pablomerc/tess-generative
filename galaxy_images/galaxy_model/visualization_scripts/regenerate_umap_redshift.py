"""Encode the HSC∩Legacy redshift-overlap dataset through the base snapshot's
encoder_1 / encoder_2, run UMAP, save coords + redshift, and push a 2x2 figure
to Discord (rows = encoder, columns = colored-by-survey / colored-by-redshift).

Dataset: /work1/jeroenaudenaert/pablomer/data/umap_withredshift/overlap_dataset.h5
  hsc_flux       (N, 5, 160, 160) float16  bands g,r,i,z,y
  legacy_flux    (N, 4, 160, 160) float16  bands g,r,i,z
  redshift       (N,)             float32  DESI z
  ...
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
GM_DIR = SCRIPT_DIR.parent
REPO_ROOT = GM_DIR.parent.parent
LATENT_SPACE_DIR = SCRIPT_DIR / "neighbors_visualization" / "latent_space"
for p in (str(REPO_ROOT), str(GM_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from galaxy_images.image_preprocessing import preprocess_image_v2
from galaxy_images.galaxy_model.data import zoom_legacy_image
from galaxy_images.galaxy_model.neighbors import NORM_DICT
from double_train_fm_neighbors import ConditionalFlowMatchingModule
from discord_notify import notify
import plot_umap_from_file as plotter


def _pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    for gpu_id in range(torch.cuda.device_count()):
        try:
            t = torch.tensor([1.0], device=f"cuda:{gpu_id}")
            del t
            torch.cuda.empty_cache()
            return torch.device(f"cuda:{gpu_id}")
        except RuntimeError:
            continue
    return torch.device("cpu")


def _preprocess(raw_chw: np.ndarray, survey: str, crop_size: int) -> torch.Tensor:
    """Apply the same crop/zoom/normalize chain as NeighborsEfficientDataset."""
    img = torch.from_numpy(raw_chw.astype(np.float32))
    img = preprocess_image_v2(img, crop_size=crop_size, survey=survey)
    if survey == "legacy":
        img = zoom_legacy_image(img)
        mean, std = NORM_DICT["legacy_zoom"]
    else:
        mean, std = NORM_DICT["hsc"]
    return (img - mean) / std


def _encode(model, images: torch.Tensor, encoder, device, bs: int) -> torch.Tensor:
    out = []
    for i in range(0, images.shape[0], bs):
        chunk = images[i:i + bs].to(device, non_blocking=True)
        emb = encoder(chunk)
        out.append(emb.detach().cpu())
    return torch.cat(out, dim=0)


def _scatter_redshift(ax, xy_hsc, xy_leg, z, title, vmin, vmax, point_size, alpha):
    xy = np.concatenate([xy_hsc, xy_leg], axis=0)
    zz = np.concatenate([z, z], axis=0)  # same redshift for both surveys (paired)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=zz, cmap="viridis", vmin=vmin, vmax=vmax,
                    s=point_size, alpha=alpha)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    return sc


def _scatter_survey(ax, xy_hsc, xy_leg, title, point_size, alpha):
    ax.scatter(xy_hsc[:, 0], xy_hsc[:, 1], s=point_size, c=plotter.COLOR_HSC,
               alpha=alpha, label="HSC")
    ax.scatter(xy_leg[:, 0], xy_leg[:, 1], s=point_size, c=plotter.COLOR_LEGACY,
               alpha=alpha, label="Legacy")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plotter.COLOR_HSC,
               markeredgecolor='black', markersize=8, label='HSC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plotter.COLOR_LEGACY,
               markeredgecolor='black', markersize=8, label='Legacy'),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt")
    parser.add_argument("--h5", type=Path,
                        default=Path("/work1/jeroenaudenaert/pablomer/data/umap_withredshift/overlap_dataset.h5"))
    parser.add_argument("--tag", type=str, default="overlap_redshift")
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--umap-seed", type=int, default=314)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--point-size", type=float, default=14)
    parser.add_argument("--webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "<no-slurm>")
    host = os.environ.get("SLURMD_NODENAME", socket.gethostname())
    t0 = time.perf_counter()

    notify(args.webhook,
           f"▶️ **UMAP redshift-overlap** start `{args.tag}`  host=`{host}`  jobid=`{job_id}`")

    try:
        device = _pick_device()
        print(f"[umap-z] device={device}")

        # --- Load model ---
        model = ConditionalFlowMatchingModule.load_from_checkpoint(
            str(args.checkpoint), map_location="cpu")
        model.eval(); torch.set_grad_enabled(False)
        model = model.to(device)
        dim = int(model.hparams.cross_attention_dim)

        # --- Load + preprocess H5 ---
        with h5py.File(args.h5, "r") as f:
            hsc_raw = f["hsc_flux"][:]      # (N,5,160,160) float16
            leg_raw = f["legacy_flux"][:]   # (N,4,160,160) float16
            redshift = f["redshift"][:].astype(np.float32)
            try:
                logmstar = f["physics/provabgs_logmstar"][:].astype(np.float32)
            except KeyError:
                logmstar = None
            try:
                ra = f["meta/ra"][:]; dec = f["meta/dec"][:]
            except KeyError:
                ra = dec = None
        n = redshift.shape[0]
        print(f"[umap-z] N={n}  hsc_raw={hsc_raw.shape}  leg_raw={leg_raw.shape}")

        # Vectorized preprocessing (same per-image chain).
        t = time.perf_counter()
        hsc_proc = torch.stack([_preprocess(hsc_raw[i], "hsc", args.crop_size)[:4]
                                for i in range(n)], dim=0)
        leg_proc = torch.stack([_preprocess(leg_raw[i], "legacy", args.crop_size)
                                for i in range(n)], dim=0)
        prep_time = time.perf_counter() - t
        print(f"[umap-z] preprocessed in {prep_time:.1f}s  hsc={tuple(hsc_proc.shape)}  leg={tuple(leg_proc.shape)}")

        # --- Encode ---
        t = time.perf_counter()
        hsc_e1 = _encode(model, hsc_proc, model.encoder_1, device, args.batch_size)
        leg_e1 = _encode(model, leg_proc, model.encoder_1, device, args.batch_size)
        hsc_e2 = _encode(model, hsc_proc, model.encoder_2, device, args.batch_size)
        leg_e2 = _encode(model, leg_proc, model.encoder_2, device, args.batch_size)
        encode_time = time.perf_counter() - t
        print(f"[umap-z] encoded in {encode_time:.1f}s")

        all_e1 = torch.cat([hsc_e1, leg_e1], 0).flatten(1).numpy()
        all_e2 = torch.cat([hsc_e2, leg_e2], 0).flatten(1).numpy()
        umap_params = dict(n_neighbors=15, min_dist=0.1, n_components=2,
                           metric="euclidean", random_state=int(args.umap_seed))
        emb1 = umap.UMAP(**umap_params).fit_transform(all_e1)
        emb2 = umap.UMAP(**umap_params).fit_transform(all_e2)
        hsc_u1, leg_u1 = emb1[:n], emb1[n:]
        hsc_u2, leg_u2 = emb2[:n], emb2[n:]

        # --- Persist coords + redshift ---
        LATENT_SPACE_DIR.mkdir(parents=True, exist_ok=True)
        npz_path = LATENT_SPACE_DIR / f"umap_redshift_zdim{dim}_{args.tag}_data.npz"
        if npz_path.exists():
            raise FileExistsError(f"Refusing to overwrite {npz_path}")
        np.savez_compressed(
            npz_path,
            hsc_umap_1=hsc_u1.astype(np.float32),
            legacy_umap_1=leg_u1.astype(np.float32),
            hsc_umap_2=hsc_u2.astype(np.float32),
            legacy_umap_2=leg_u2.astype(np.float32),
            redshift=redshift,
            **({"logmstar": logmstar} if logmstar is not None else {}),
            **({"ra": ra, "dec": dec} if ra is not None else {}),
        )
        meta = {
            "tag": args.tag, "n": int(n), "dim": dim,
            "checkpoint": str(args.checkpoint), "h5": str(args.h5),
            "umap_params": umap_params, "umap_seed": int(args.umap_seed),
            "redshift_stats": {
                "min": float(redshift.min()), "max": float(redshift.max()),
                "median": float(np.median(redshift)),
                "p5": float(np.percentile(redshift, 5)),
                "p95": float(np.percentile(redshift, 95)),
            },
            "host": host, "slurm_job_id": job_id,
        }
        (LATENT_SPACE_DIR / f"umap_redshift_zdim{dim}_{args.tag}_metadata.json").write_text(
            json.dumps(meta, indent=2))
        print(f"[umap-z] saved {npz_path}")

        # --- Plot 2x2: rows = encoder, cols = survey / redshift ---
        z_lo = float(np.percentile(redshift, 1))
        z_hi = float(np.percentile(redshift, 99))
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        _scatter_survey(axes[0, 0], hsc_u1, leg_u1,
                        "Physics latent — colored by survey",
                        args.point_size, args.alpha)
        sc1 = _scatter_redshift(axes[0, 1], hsc_u1, leg_u1, redshift,
                                "Physics latent — colored by redshift",
                                z_lo, z_hi, args.point_size, args.alpha)
        plt.colorbar(sc1, ax=axes[0, 1], label="DESI redshift")

        _scatter_survey(axes[1, 0], hsc_u2, leg_u2,
                        "Instrument latent — colored by survey",
                        args.point_size, args.alpha)
        sc2 = _scatter_redshift(axes[1, 1], hsc_u2, leg_u2, redshift,
                                "Instrument latent — colored by redshift",
                                z_lo, z_hi, args.point_size, args.alpha)
        plt.colorbar(sc2, ax=axes[1, 1], label="DESI redshift")

        plt.suptitle(f"base ckpt — UMAP on overlap dataset (N={n})  "
                     f"z range=[{redshift.min():.3f}, {redshift.max():.3f}]  "
                     f"median={np.median(redshift):.3f}",
                     fontsize=14)
        plt.tight_layout()
        png_path = LATENT_SPACE_DIR / f"umap_redshift_zdim{dim}_{args.tag}.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[umap-z] saved {png_path}")

        total = time.perf_counter() - t0
        notify(args.webhook,
               f"✅ `{args.tag}` done in {total:.1f}s — N={n}, "
               f"z=[{redshift.min():.3f}, {redshift.max():.3f}], median={np.median(redshift):.3f}",
               file_path=png_path)

    except Exception as e:
        tb = traceback.format_exc()
        notify(args.webhook,
               f"❌ `{args.tag}` failed: `{type(e).__name__}: {e}`\n```\n{tb[-1500:]}\n```")
        raise


if __name__ == "__main__":
    main()
