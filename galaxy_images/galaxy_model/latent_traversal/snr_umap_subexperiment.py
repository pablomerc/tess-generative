"""
SNR UMAP subexperiment.

Question: does the model's latent space have a coherent "SNR direction", or do
HSC images cluster by SNR percentile?

Procedure:
  1. Sample 8000 random MMU pairs (HSC + Legacy) via NeighborsSimpleDataset.
  2. Pick 8 HSC images closest to each of SNR percentiles {5, 25, 50, 75, 95}
     using hsc_noise_metrics.h5 (40 extra HSC images).
  3. Encode all images through encoder_1 (physics) and encoder_2 (instrument).
  4. Fit UMAP on the joint set per encoder.
  5. Plot both panels with the random pool faded and the 5 SNR groups overlaid
     with distinct colors + star markers, legend included.
  6. Save figure + embeddings.npz, post the figure to a dedicated Discord
     webhook (separate from the per-job ping channel).
"""

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
import umap
from torch.utils.data import DataLoader, Subset

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
from galaxy_images.galaxy_model.neighbors import NeighborsSimpleDataset, preprocess_raw_image
from discord_notify import notify as _notify

# ============= CONFIG =============

MODEL_CHECKPOINT = (
    "/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/"
    "outputs/neighbors_all_attn/checkpoints/best-epoch=228-step=87000.ckpt"
)
NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
METRICS_HDF5   = Path("/work1/jeroenaudenaert/pablomer/data/hsc_noise_metrics.h5")

OUTPUT_DIR    = _script_dir / "snr_umap_subexperiment"
FIG_DIR       = OUTPUT_DIR / "figures"
EMB_PATH      = OUTPUT_DIR / "embeddings.npz"

# Webhook the user supplied for THIS subexperiment's figure (separate from the
# discord_notify.py one used for SLURM start/stop pings).
FIGURE_WEBHOOK = (
    "https://discord.com/api/webhooks/1497979390079008955/"
    "MPr8CeORWZztX1LEu1XGU76QdIiiKVyHvxHZ03i9H20CeB3VLVEwJGlm1z3M619NYk9r"
)

N_RANDOM_PAIRS = 8000
SNR_PERCENTILES = [5, 25, 50, 75, 95]
N_PER_GROUP = 8
RANDOM_SEED = 42

BATCH_SIZE = 256
CROP_SIZE = 48

UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "euclidean",
    "random_state": 42,
}

COLOR_HSC = "#e8c4a0"
COLOR_LEGACY = "#8eb8e8"


# ============= HELPERS =============

def determine_device():
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            try:
                torch.tensor([1.0], device=f"cuda:{gpu_id}")
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                return torch.device(f"cuda:{gpu_id}")
            except RuntimeError:
                continue
    print("No GPU; using CPU")
    return torch.device("cpu")


def collate_simple(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    return hsc, leg


def find_bucket_idxs(score, valid_mask, percentile, n):
    """Pick the n indices whose `score` is closest to its percentile (over valid)."""
    valid_score = score[valid_mask]
    target_val = np.nanpercentile(valid_score, percentile)
    distances = np.where(valid_mask, np.abs(score - target_val), np.inf)
    return np.argsort(distances)[:n], float(target_val)


def load_hsc_image(hdf5_path, raw_row):
    with h5py.File(hdf5_path, "r") as f:
        img_raw = np.asarray(f["images_hsc"][raw_row], dtype=np.float32)
    return preprocess_raw_image(img_raw, survey="hsc", crop_size=CROP_SIZE)[:4]


def post_figure_to_webhook(file_path, message, webhook):
    with open(file_path, "rb") as fh:
        data = fh.read()
    try:
        resp = requests.post(
            webhook,
            data={"content": message[:1900]},
            files={"file": (file_path.name, io.BytesIO(data), "image/png")},
            timeout=120,
        )
        if resp.status_code in (200, 204):
            print(f"  Posted {file_path.name} to figure webhook")
        else:
            print(f"  WARNING webhook {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  WARNING webhook post failed: {e}")


def post_text_to_webhook(message, webhook):
    try:
        requests.post(webhook, json={"content": message[:1900]}, timeout=20)
    except Exception as e:
        print(f"  WARNING webhook text post failed: {e}")


# ============= ENCODING =============

def encode_images(model, images_chw, device, label):
    """Encode (N, C, H, W) tensor through encoder_1 and encoder_2; returns
    flattened np arrays (N, D1) and (N, D2)."""
    e1_list, e2_list = [], []
    n = images_chw.shape[0]
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk = images_chw[start:end].to(device)
            e1_list.append(model.encoder_1(chunk).cpu())
            e2_list.append(model.encoder_2(chunk).cpu())
            print(f"    [{label}] {end}/{n}")
    e1 = torch.cat(e1_list).flatten(start_dim=1).numpy()
    e2 = torch.cat(e2_list).flatten(start_dim=1).numpy()
    return e1, e2


# ============= PLOTTING =============

def plot_umap(umap_e1, umap_e2, slices, snr_groups_meta, output_path):
    """slices = dict of name -> (start, end). snr_groups_meta = list of dicts
    with keys {label, percentile, start, end, color}."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, umap_xy, title in [
        (axes[0], umap_e1, "Encoder 1 — Physics"),
        (axes[1], umap_e2, "Encoder 2 — Instrument"),
    ]:
        s, e = slices["random_hsc"]
        ax.scatter(umap_xy[s:e, 0], umap_xy[s:e, 1],
                   s=4, alpha=0.25, c=COLOR_HSC, label="Random HSC", linewidths=0)
        s, e = slices["random_legacy"]
        ax.scatter(umap_xy[s:e, 0], umap_xy[s:e, 1],
                   s=4, alpha=0.25, c=COLOR_LEGACY, label="Random Legacy", linewidths=0)

        for g in snr_groups_meta:
            s, e = g["start"], g["end"]
            ax.scatter(
                umap_xy[s:e, 0], umap_xy[s:e, 1],
                s=160, c=g["color"], marker="*",
                edgecolors="black", linewidths=0.8,
                label=f"SNR p{g['percentile']} (HSC, n={e - s})",
                zorder=5,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9, markerscale=0.9)

    n_random = slices["random_hsc"][1] - slices["random_hsc"][0]
    fig.suptitle(
        f"SNR-overlay UMAP — {n_random} random HSC + {n_random} random Legacy + "
        f"5×{N_PER_GROUP} HSC at SNR p5/p25/p50/p75/p95",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ============= MAIN =============

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    device = determine_device()

    # ---- Load model ----
    print(f"\nLoading model from {MODEL_CHECKPOINT}")
    t0 = time.time()
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        MODEL_CHECKPOINT, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ---- Load random pool ----
    print(f"\nOpening NeighborsSimpleDataset from {NEIGHBORS_HDF5}")
    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n_total = len(full_dataset)
    n_use = min(N_RANDOM_PAIRS, n_total)
    rng = np.random.default_rng(RANDOM_SEED)
    random_indices = rng.choice(n_total, size=n_use, replace=False)
    random_indices = np.sort(random_indices)
    print(f"  Sampled {n_use} random MMU indices (dataset total={n_total})")

    subset = Subset(full_dataset, random_indices.tolist())
    loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=collate_simple,
    )

    print("\nLoading random HSC + Legacy tensors")
    hsc_chunks, leg_chunks = [], []
    for bi, (h, l) in enumerate(loader):
        hsc_chunks.append(h)
        leg_chunks.append(l)
        print(f"  loaded batch {bi + 1}, total so far: {(bi + 1) * BATCH_SIZE}")
    random_hsc = torch.cat(hsc_chunks)  # (N, 4, 48, 48)
    random_legacy = torch.cat(leg_chunks)
    print(f"  random_hsc={tuple(random_hsc.shape)} random_legacy={tuple(random_legacy.shape)}")

    # ---- Pick SNR group examples ----
    print(f"\nLoading SNR metrics from {METRICS_HDF5}")
    with h5py.File(METRICS_HDF5, "r") as mf:
        rows_metrics = np.array(mf["hdf5_row_idx"], dtype=np.int64)
        snr_neg = np.array(mf["snr_neg_4band"], dtype=np.float64)
    snr_pos = -snr_neg
    valid = np.isfinite(snr_neg)
    print(f"  N_metrics={len(rows_metrics):,}  finite SNR={valid.sum():,}")

    snr_group_records = []  # ordered list of dicts
    snr_hsc_imgs = []        # list of tensors (4, 48, 48)
    cursor = 0
    for pct in SNR_PERCENTILES:
        idxs, target_val = find_bucket_idxs(snr_neg, valid, pct, N_PER_GROUP)
        rows = rows_metrics[idxs]
        snr_at = snr_pos[idxs]
        print(f"  p{pct:>2d} target_neg={target_val:+.4f}  "
              f"SNR={snr_at.mean():.2f}  (rows={rows.tolist()})")
        for stat_idx, raw_row, snr_val in zip(idxs, rows, snr_at):
            img = load_hsc_image(NEIGHBORS_HDF5, int(raw_row))
            snr_hsc_imgs.append(img)
        snr_group_records.append(dict(
            percentile=pct,
            stats_idxs=idxs.astype(np.int64),
            hdf5_rows=rows.astype(np.int64),
            snr_pos=snr_at.astype(np.float64),
            start=cursor,
            end=cursor + N_PER_GROUP,
        ))
        cursor += N_PER_GROUP

    snr_hsc = torch.stack(snr_hsc_imgs)  # (40, 4, 48, 48)
    print(f"  snr_hsc={tuple(snr_hsc.shape)}")

    # ---- Encode all ----
    print("\nEncoding random HSC")
    rh_e1, rh_e2 = encode_images(model, random_hsc, device, "random_hsc")
    print("Encoding random Legacy")
    rl_e1, rl_e2 = encode_images(model, random_legacy, device, "random_legacy")
    print("Encoding SNR HSC groups")
    sh_e1, sh_e2 = encode_images(model, snr_hsc, device, "snr_hsc")

    n_random = rh_e1.shape[0]
    n_snr = sh_e1.shape[0]
    slices = {
        "random_hsc":     (0, n_random),
        "random_legacy":  (n_random, 2 * n_random),
        "snr_hsc":        (2 * n_random, 2 * n_random + n_snr),
    }
    # Re-index group start/end into the joint matrix
    snr_off = slices["snr_hsc"][0]
    for g in snr_group_records:
        g["start"] += snr_off
        g["end"] += snr_off

    all_e1 = np.concatenate([rh_e1, rl_e1, sh_e1], axis=0)
    all_e2 = np.concatenate([rh_e2, rl_e2, sh_e2], axis=0)
    print(f"  Joint matrix shapes: e1={all_e1.shape} e2={all_e2.shape}")

    # ---- UMAP ----
    print("\nRunning UMAP — encoder 1 (physics)")
    t0 = time.time()
    reducer_e1 = umap.UMAP(**UMAP_PARAMS)
    umap_e1 = reducer_e1.fit_transform(all_e1)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Running UMAP — encoder 2 (instrument)")
    t0 = time.time()
    reducer_e2 = umap.UMAP(**UMAP_PARAMS)
    umap_e2 = reducer_e2.fit_transform(all_e2)
    print(f"  done in {time.time() - t0:.1f}s")

    # ---- Color map for SNR groups (clean=cool, noisy=warm) ----
    cmap = plt.get_cmap("viridis")
    snr_groups_meta = []
    for i, g in enumerate(snr_group_records):
        color = cmap(i / max(len(snr_group_records) - 1, 1))
        snr_groups_meta.append(dict(
            label=f"SNR p{g['percentile']}",
            percentile=g["percentile"],
            start=g["start"],
            end=g["end"],
            color=color,
        ))

    # ---- Plot ----
    print("\nPlotting")
    out_png = FIG_DIR / "umap_snr_groups.png"
    plot_umap(umap_e1, umap_e2, slices, snr_groups_meta, out_png)

    # ---- Save embeddings ----
    np.savez_compressed(
        EMB_PATH,
        umap_e1=umap_e1.astype(np.float32),
        umap_e2=umap_e2.astype(np.float32),
        random_indices=random_indices.astype(np.int64),
        slice_random_hsc=np.array(slices["random_hsc"], dtype=np.int64),
        slice_random_legacy=np.array(slices["random_legacy"], dtype=np.int64),
        slice_snr_hsc=np.array(slices["snr_hsc"], dtype=np.int64),
        snr_group_percentiles=np.array(SNR_PERCENTILES, dtype=np.int64),
        snr_group_stats_idxs=np.stack([g["stats_idxs"] for g in snr_group_records]),
        snr_group_hdf5_rows=np.stack([g["hdf5_rows"] for g in snr_group_records]),
        snr_group_snr_pos=np.stack([g["snr_pos"] for g in snr_group_records]),
    )
    print(f"  Saved {EMB_PATH}")

    # ---- Discord post (figure webhook) ----
    summary = (
        f"**SNR UMAP subexperiment** | job={job_id}\n"
        f"{n_random} random HSC + {n_random} random Legacy + "
        f"5 SNR groups × {N_PER_GROUP} HSC = {n_random * 2 + n_snr} latent points.\n"
        f"Per-group mean SNR: "
        + " | ".join(
            f"p{g['percentile']}={g['snr_pos'].mean():.1f}"
            for g in snr_group_records
        )
    )
    post_figure_to_webhook(out_png, summary, FIGURE_WEBHOOK)
    _notify(f"✅ SNR UMAP subexp complete (job={job_id}) → figure posted to dedicated webhook")


if __name__ == "__main__":
    main()
