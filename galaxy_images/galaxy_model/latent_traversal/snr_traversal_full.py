"""
Batch SNR-conditioned latent traversal.

Loops over a list of target stats_idxs and a list of modes, loading the model
ONCE. Per (target, mode):
  - selects K=n_select HSC neighbors per SNR bucket (5 buckets: p95→p5, noisy→clean)
  - repeats the single nearest neighbor n_pass times if mode is k1xN
  - generates one HSC reconstruction per bucket
  - saves the rowscale PNG to snr_traversal_results/target_<idx>/
  - posts the rowscale PNG to Discord
  - appends all arrays (target HSC/Legacy, conditioning images, generated images,
    metadata) into one HDF5 sidecar so visualization can be redone offline

Modes:
  k5     : 5 distinct neighbors                 → (5, 5, 4,48,48) sameins
  k10    : 10 distinct neighbors                → (5,10, 4,48,48)
  k15    : 15 distinct neighbors                → (5,15, 4,48,48)
  k1x5   : 1 nearest neighbor repeated 5 times  → (5, 5, 4,48,48), all 5 identical per bucket
  k1x10  : 1 nearest neighbor repeated 10 times → (5,10, 4,48,48)
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_script_dir))

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
torch.backends.cuda.preferred_blas_library("hipblas")

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.neighbors import preprocess_raw_image
from discord_notify import notify as _notify

# ============= CONFIG =============

MODEL_CHECKPOINT = (
    "/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/"
    "outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt"
)
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
METRICS_HDF5   = Path("/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5")
STATS_HDF5     = Path("/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5")
OUTPUT_DIR     = Path(__file__).resolve().parent / "snr_traversal_results"
ARRAYS_FILE    = OUTPUT_DIR / "snr_traversal_arrays.h5"

DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)

CROP_SIZE = 48
SNR_LEVELS = [
    ("Noisy neighbors (p95)", 95),
    ("p75",                   75),
    ("p50",                   50),
    ("p25",                   25),
    ("Clean neighbors (p5)",   5),
]

MODE_CONFIG = {
    "k5":    dict(n_select=5,  n_pass=5,  repeat_one=False),
    "k10":   dict(n_select=10, n_pass=10, repeat_one=False),
    "k15":   dict(n_select=15, n_pass=15, repeat_one=False),
    "k1x5":  dict(n_select=1,  n_pass=5,  repeat_one=True),
    "k1x10": dict(n_select=1,  n_pass=10, repeat_one=True),
}

# ============= HELPERS =============

def determine_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            try:
                torch.tensor([1.0], device=f"cuda:{gpu_id}")
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                break
            except RuntimeError:
                continue
    return device


def vis_rowscale(img, vmin, vmax):
    rgb = img[:3].copy()
    rgb = (rgb - vmin[:, None, None]) / (vmax[:, None, None] - vmin[:, None, None] + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def vis_indep(img):
    rgb = img[:3].copy()
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))


def send_to_discord(file_path, message):
    with open(file_path, "rb") as fh:
        data = fh.read()
    try:
        resp = requests.post(
            DISCORD_WEBHOOK,
            data={"content": message},
            files={"file": (file_path.name, io.BytesIO(data), "image/png")},
            timeout=60,
        )
        if resp.status_code in (200, 204):
            print(f"  Sent {file_path.name}")
        else:
            print(f"  WARNING Discord {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  WARNING Discord post failed: {e}")


def load_legacy(hdf5_path, raw_row):
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f["images_legacy"][raw_row], dtype=np.float32)
    return preprocess_raw_image(img_raw, survey="legacy", crop_size=CROP_SIZE)


def load_hsc(hdf5_path, raw_row):
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f["images_hsc"][raw_row], dtype=np.float32)
    return preprocess_raw_image(img_raw, survey="hsc", crop_size=CROP_SIZE)[:4]


def find_bucket_idxs(score, valid_mask, percentile, n):
    valid_score = score[valid_mask]
    target_val = np.nanpercentile(valid_score, percentile)
    distances = np.where(valid_mask, np.abs(score - target_val), np.inf)
    nearest = np.argsort(distances)[:n]
    return nearest, float(target_val)


# ============= PLOT (rowscale only) =============

def plot_grid(
    target_hsc_np, target_legacy_np,
    target_meta, bucket_data, generated_images,
    target_idx, mode, output_path, vis_method, vmin=None, vmax=None,
):
    n_levels = len(SNR_LEVELS)
    n_cols = 2 + n_levels
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.5, 2 * 2.8))

    def apply_vis(img):
        if vis_method == "rowscale":
            return vis_rowscale(img, vmin, vmax)
        if vis_method == "indep":
            return vis_indep(img)
        raise ValueError(vis_method)

    def show(ax, img, title=None):
        ax.imshow(apply_vis(img)); ax.set_axis_off()
        if title: ax.set_title(title, fontsize=8)

    show(axes[0, 0], target_hsc_np,
         f"Target HSC\nSNR={target_meta['snr']:.1f}\n"
         f"ivar={target_meta['ivar']:.0f} PSF={target_meta['psf']:.2f}\"")
    show(axes[0, 1], target_legacy_np, "Legacy source\n(same galaxy)")
    for col, b in enumerate(bucket_data):
        show(axes[0, 2 + col], b["cond_show"],
             f"{b['label']}\nSNR={b['snr_avg']:.1f}\n"
             f"ivar={b['ivar_avg']:.0f} PSF={b['psf_avg']:.2f}\"")
    axes[1, 0].set_axis_off(); axes[1, 1].set_axis_off()
    for col, gen in enumerate(generated_images):
        show(axes[1, 2 + col], gen, f"Generated\n({bucket_data[col]['label']})")

    for row_i, row_lbl in enumerate(["Conditioning", "Generated"]):
        ax0 = axes[row_i, 0]
        ax0.set_axis_on(); ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values(): sp.set_visible(False)
        ax0.set_ylabel(row_lbl, fontsize=11, fontweight="bold", labelpad=6)

    vis_label = "Row-scaled to target" if vis_method == "rowscale" else "Per-image min-max (indep)"
    fig.suptitle(
        f"SNR traversal — cmodel flux/fluxerr (4-band avg)  "
        f"(target stats_idx={target_idx}, mode={mode})  {vis_label}",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============= ARRAY STORAGE =============

def save_target_mode_arrays(arr_path: Path, target_idx, mode, target_meta,
                             target_hsc_np, target_legacy_np,
                             bucket_data, generated_images, n_select, n_pass, repeat_one):
    """Append one (target, mode) record into a master HDF5 file."""
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    grp_name = f"target_{target_idx:05d}/{mode}"
    with h5py.File(arr_path, "a") as f:
        if grp_name in f:
            del f[grp_name]
        g = f.create_group(grp_name)

        # target metadata + images
        for k, v in target_meta.items():
            g.attrs[f"target_{k}"] = v
        g.create_dataset("target_hsc", data=target_hsc_np.astype(np.float32))
        g.create_dataset("target_legacy", data=target_legacy_np.astype(np.float32))

        # mode info
        g.attrs["mode"]        = mode
        g.attrs["n_select"]    = n_select
        g.attrs["n_pass"]      = n_pass
        g.attrs["repeat_one"]  = repeat_one

        # buckets (5 of them)
        labels = np.array([b["label"] for b in bucket_data], dtype="S64")
        pcts   = np.array([b["pct"]   for b in bucket_data], dtype=np.float32)
        sel_si = np.stack([b["sel_stats_idx"] for b in bucket_data]).astype(np.int32)
        sel_hr = np.stack([b["sel_hdf5_row"]  for b in bucket_data]).astype(np.int32)
        snr_avg = np.array([b["snr_avg"] for b in bucket_data], dtype=np.float32)
        iv_avg  = np.array([b["ivar_avg"] for b in bucket_data], dtype=np.float32)
        ps_avg  = np.array([b["psf_avg"] for b in bucket_data], dtype=np.float32)
        cond_unique = np.stack([b["cond_unique"] for b in bucket_data]).astype(np.float32)

        bg = g.create_group("buckets")
        bg.create_dataset("labels",                data=labels)
        bg.create_dataset("percentiles",           data=pcts)
        bg.create_dataset("selected_stats_idxs",   data=sel_si)
        bg.create_dataset("selected_hdf5_rows",    data=sel_hr)
        bg.create_dataset("snr_avgs",              data=snr_avg)
        bg.create_dataset("ivar_avgs",             data=iv_avg)
        bg.create_dataset("psf_avgs",              data=ps_avg)
        bg.create_dataset("cond_images_unique",    data=cond_unique)

        gen = np.stack(generated_images).astype(np.float32)
        g.create_dataset("generated", data=gen)


# ============= CORE: process one (target, mode) =============

def run_one(model, device, hdf5_path, target_idx, target_hsc, target_legacy,
            target_meta, mode, snr_neg, snr_pos, valid, ivar_all, psf_all,
            hdf5_row_idx, output_dir, arrays_file, post_discord, fixed_noise=False):
    cfg = MODE_CONFIG[mode]
    n_select, n_pass, repeat_one = cfg["n_select"], cfg["n_pass"], cfg["repeat_one"]

    # Build buckets + load conditioning HSC images
    bucket_data = []
    print(f"  [{mode}] building 5 buckets (n_select={n_select}, n_pass={n_pass}, repeat_one={repeat_one})")
    for label, neg_pct in SNR_LEVELS:
        idxs, target_neg = find_bucket_idxs(snr_neg, valid, neg_pct, n_select)
        cond_unique = np.stack(
            [load_hsc(hdf5_path, int(hdf5_row_idx[bi])).numpy() for bi in idxs]
        )  # (n_select, 4, 48, 48)
        snr_avg = float(snr_pos[idxs].mean())
        iv_avg  = float(ivar_all[idxs].mean())
        ps_avg  = float(psf_all[idxs].mean())
        bucket_data.append(dict(
            label=label,
            pct=neg_pct,
            sel_stats_idx=idxs.astype(np.int32),
            sel_hdf5_row=hdf5_row_idx[idxs].astype(np.int32),
            cond_unique=cond_unique,
            snr_avg=snr_avg, ivar_avg=iv_avg, psf_avg=ps_avg,
            cond_show=cond_unique[0],  # display first
        ))

    # Generate
    samegal  = target_legacy.unsqueeze(0).to(device)
    masks_kb = torch.ones(1, n_pass, dtype=torch.bool, device=device)

    # Optionally share one noise draw across all SNR buckets so reconstructions
    # differ only in conditioning, not in the initial noise realization.
    x_noise_shared = None
    if fixed_noise:
        x_noise_shared = torch.randn(
            1, model.in_channels, model.image_size, model.image_size,
            device=device,
        )

    generated = []
    for b in bucket_data:
        cond_t = torch.from_numpy(b["cond_unique"])  # (n_select, 4, 48, 48)
        if repeat_one:
            cond_t = cond_t[:1].repeat(n_pass, 1, 1, 1)  # (n_pass, 4, 48, 48)
        sameins = cond_t.unsqueeze(0).to(device)  # (1, n_pass, 4, 48, 48)
        gen = model.sample(samegal, sameins, masks=masks_kb, x_noise=x_noise_shared)
        generated.append(gen.squeeze(0).cpu().numpy())

    # Plot rowscale (main) + indep (separate subdir)
    target_dir = output_dir / f"target_{target_idx:05d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    indep_target_dir = output_dir / "indep" / f"target_{target_idx:05d}"
    indep_target_dir.mkdir(parents=True, exist_ok=True)

    rowscale_ref = target_hsc.numpy()[:3]
    vmin = rowscale_ref.min(axis=(1, 2))
    vmax = rowscale_ref.max(axis=(1, 2))

    fname_row   = f"snr_traversal_{target_idx:05d}_{mode}_rowscale.png"
    fname_indep = f"snr_traversal_{target_idx:05d}_{mode}_indep.png"
    out_row   = target_dir / fname_row
    out_indep = indep_target_dir / fname_indep

    plot_grid(
        target_hsc.numpy(), target_legacy.numpy(),
        target_meta, bucket_data, generated,
        target_idx, mode, out_row, vis_method="rowscale", vmin=vmin, vmax=vmax,
    )
    plot_grid(
        target_hsc.numpy(), target_legacy.numpy(),
        target_meta, bucket_data, generated,
        target_idx, mode, out_indep, vis_method="indep",
    )
    print(f"  Saved {out_row} + {out_indep}")

    # Gallery copy (rowscale only)
    gallery = output_dir / "gallery_rowscale"
    gallery.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(out_row, gallery / fname_row)

    # Save arrays
    save_target_mode_arrays(
        arrays_file, target_idx, mode, target_meta,
        target_hsc.numpy(), target_legacy.numpy(),
        bucket_data, generated, n_select, n_pass, repeat_one,
    )

    # Discord (rowscale only)
    if post_discord:
        send_to_discord(
            out_row,
            f"**SNR traversal [{mode}]** target={target_idx} | target SNR={target_meta['snr']:.1f}",
        )


# ============= MAIN =============

def parse_csv_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_csv_str(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-idxs", type=parse_csv_int, required=True,
                        help="Comma-separated stats_idx values, e.g. '92198,53601,91614'")
    parser.add_argument("--modes", type=parse_csv_str,
                        default="k5,k10,k15,k1x5,k1x10",
                        help="Comma-separated modes from {k5,k10,k15,k1x5,k1x10}")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_HDF5)
    parser.add_argument("--stats-path",   type=Path, default=STATS_HDF5)
    parser.add_argument("--hdf5-path",    type=str, default=NEIGHBORS_HDF5)
    parser.add_argument("--checkpoint",   type=str, default=MODEL_CHECKPOINT)
    parser.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    parser.add_argument("--arrays-file",  type=Path, default=ARRAYS_FILE)
    parser.add_argument("--no-discord",   action="store_true")
    parser.add_argument("--fixed-noise",  action="store_true",
                        help="Use a single shared noise draw for all SNR bucket reconstructions "
                             "so differences are due to conditioning only, not noise lottery.")
    args = parser.parse_args()

    for m in args.modes:
        if m not in MODE_CONFIG:
            raise ValueError(f"Unknown mode: {m}; valid: {list(MODE_CONFIG)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = determine_device()
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    def notify(msg):
        if not args.no_discord:
            _notify(msg)

    # ---- Load metrics + context ----
    print(f"Loading metrics from {args.metrics_path}")
    with h5py.File(args.metrics_path, "r") as mf:
        rows_metrics = np.array(mf["hdf5_row_idx"], dtype=np.int64)
        snr_neg = np.array(mf["snr_neg_4band"], dtype=np.float64)
    with h5py.File(args.stats_path, "r") as sf:
        rows_stats = np.array(sf["hdf5_row_idx"], dtype=np.int64)
        ivar_all = np.array(sf["hsc_mean_ivar"], dtype=np.float32)
        psf_all  = np.array(sf["hsc_psf_fwhm_avg"], dtype=np.float32)
    if not np.array_equal(rows_metrics, rows_stats):
        raise RuntimeError("metrics/stats row mismatch")
    snr_pos = -snr_neg
    valid = np.isfinite(snr_neg)
    hdf5_row_idx = rows_metrics
    print(f"  N={len(rows_metrics):,}  finite SNR={valid.sum():,}")

    # ---- Load model ONCE ----
    notify(
        f"**SNR traversal full** | job={job_id} | "
        f"{len(args.target_idxs)} targets × {len(args.modes)} modes = "
        f"{len(args.target_idxs) * len(args.modes)} runs"
    )
    print(f"\nLoading model {args.checkpoint}")
    t0 = time.time()
    model = ConditionalFlowMatchingModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # ---- Loop ----
    n_total = len(args.target_idxs) * len(args.modes)
    done = 0
    for target_idx in args.target_idxs:
        if not valid[target_idx]:
            print(f"!! target {target_idx} has invalid SNR; skipping")
            continue

        target_raw_row = int(hdf5_row_idx[target_idx])
        target_meta = dict(
            stats_idx=int(target_idx),
            hdf5_row=target_raw_row,
            snr=float(snr_pos[target_idx]),
            ivar=float(ivar_all[target_idx]),
            psf=float(psf_all[target_idx]),
        )
        print(f"\n=== Target stats_idx={target_idx} (hdf5_row={target_raw_row}) "
              f"SNR={target_meta['snr']:.1f} ivar={target_meta['ivar']:.0f} "
              f"PSF={target_meta['psf']:.2f}\" ===")
        notify(f"🎯 Target stats_idx={target_idx} (SNR={target_meta['snr']:.1f}) starting")

        target_hsc    = load_hsc(args.hdf5_path, target_raw_row)
        target_legacy = load_legacy(args.hdf5_path, target_raw_row)

        for mode in args.modes:
            t0 = time.time()
            run_one(
                model, device, args.hdf5_path, target_idx,
                target_hsc, target_legacy, target_meta, mode,
                snr_neg, snr_pos, valid, ivar_all, psf_all,
                hdf5_row_idx, args.output_dir, args.arrays_file,
                post_discord=not args.no_discord,
                fixed_noise=args.fixed_noise,
            )
            done += 1
            print(f"  [{done}/{n_total}] mode={mode} done in {time.time()-t0:.1f}s")

    notify(f"✅ SNR traversal full: {done}/{n_total} runs complete | arrays → {args.arrays_file.name}")
    print(f"\nDone. {done}/{n_total} runs.")
    print(f"Arrays saved to {args.arrays_file}")


if __name__ == "__main__":
    main()
