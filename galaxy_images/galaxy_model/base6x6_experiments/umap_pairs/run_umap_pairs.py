"""UMAP of HSC/Legacy pairs for the base6x6 / base6x6-15NB checkpoints.

Loads ConditionalFlowMatchingModule, encodes pairs via encoder_1 (physics)
and encoder_2 (instrument), runs UMAP, and produces both a scatter plot and
a pairs-overlay plot (same-galaxy pairs connected by lines).

Usage:
  python -m galaxy_images.galaxy_model.base6x6_experiments.umap_pairs.run_umap_pairs \
      --variant spatial_pooled --n 4096 \
      --ckpt galaxy_images/galaxy_model/checkpoints/base6x6/best-epoch=216-step=83000.ckpt \
      --out-dir galaxy_images/galaxy_model/base6x6_experiments/umap_pairs/outputs/base6x6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from tqdm import tqdm

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[4]  # .../tess-generative
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")

from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.data_loaders import (
    NEIGHBORS_HDF5_DEFAULT, make_loader, make_pair_dataset,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import (
    post_image, post_text,
)


VARIANTS = ("spatial_pooled", "spatial_flat")

UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, n_components=2,
                   metric="euclidean", random_state=42)

PAIR_PALETTE = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def load_model(ckpt_path: str | Path, device: torch.device):
    from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
    model = ConditionalFlowMatchingModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


@torch.no_grad()
def extract_physics(model, image: torch.Tensor, variant: str) -> torch.Tensor:
    tokens = model.encoder_1(image)  # (B, seq_len, cross_attention_dim)
    if variant == "spatial_pooled":
        return tokens.mean(dim=1).float()
    if variant == "spatial_flat":
        return tokens.reshape(tokens.shape[0], -1).float()
    raise ValueError(f"Unknown variant: {variant!r}")


@torch.no_grad()
def extract_instrument(model, image: torch.Tensor) -> torch.Tensor:
    tokens = model.encoder_2(image)  # (B, seq_len, cross_attention_dim)
    return tokens.mean(dim=1).float()


def encode_pairs(model, dataset, variant: str, batch_size: int, device: torch.device):
    loader = make_loader(dataset, batch_size=batch_size, num_workers=2,
                         pin_memory=(device.type == "cuda"))
    hsc_phys, leg_phys, hsc_inst, leg_inst = [], [], [], []
    for hsc, leg, _, _ in tqdm(loader, desc=f"encode[{variant}]"):
        hsc = hsc.to(device, non_blocking=True)
        leg = leg.to(device, non_blocking=True)
        hsc_phys.append(extract_physics(model, hsc, variant).cpu().numpy())
        leg_phys.append(extract_physics(model, leg, variant).cpu().numpy())
        hsc_inst.append(extract_instrument(model, hsc).cpu().numpy())
        leg_inst.append(extract_instrument(model, leg).cpu().numpy())
    return (np.concatenate(hsc_phys), np.concatenate(leg_phys),
            np.concatenate(hsc_inst), np.concatenate(leg_inst))


def _plot_overlay(emb_combined: np.ndarray, n_hsc: int, pair_idx: np.ndarray,
                  ax, title: str) -> None:
    ax.scatter(emb_combined[:n_hsc, 0], emb_combined[:n_hsc, 1],
               s=4, c="#8eb8e8", alpha=0.35, label="HSC", rasterized=True)
    ax.scatter(emb_combined[n_hsc:, 0], emb_combined[n_hsc:, 1],
               s=4, c="#e8c4a0", alpha=0.35, label="Legacy", rasterized=True)
    for i, p in enumerate(pair_idx):
        color = PAIR_PALETTE[i % len(PAIR_PALETTE)]
        x_h, y_h = emb_combined[p]
        x_l, y_l = emb_combined[n_hsc + p]
        ax.plot([x_h, x_l], [y_h, y_l], "-", color=color, alpha=0.8, linewidth=1.2)
        ax.scatter([x_h], [y_h], s=140, c=color, marker="o",
                   edgecolors="black", linewidths=0.8, zorder=3)
        ax.scatter([x_l], [y_l], s=140, c=color, marker="s",
                   edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(str(i + 1), (x_h, y_h), color="black", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=4)
        ax.annotate(str(i + 1), (x_l, y_l), color="black", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=4)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def run(variant: str, n: int, batch_size: int, ckpt: str, hdf5_path: str,
        out_dir: Path, post_discord: bool, overlay_k: int = 8, overlay_seed: int = 0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(ckpt).name

    if post_discord:
        post_text(f"🟢 [base6x6/umap_pairs/{variant}] starting — n={n}, ckpt={ckpt_name}")

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, variant={variant}, n={n}")

    model = load_model(ckpt, device=device)
    dataset = make_pair_dataset(hdf5_path=hdf5_path, n=n, shuffle=False)
    print(f"dataset size requested={n}, available={len(dataset)}")

    hsc_p, leg_p, hsc_i, leg_i = encode_pairs(model, dataset, variant, batch_size, device)
    print(f"physics: hsc={hsc_p.shape}, leg={leg_p.shape} | instrument: hsc={hsc_i.shape}")

    print("UMAP physics …")
    emb_p = umap.UMAP(**UMAP_PARAMS).fit_transform(np.concatenate([hsc_p, leg_p], axis=0))
    print("UMAP instrument …")
    emb_i = umap.UMAP(**UMAP_PARAMS).fit_transform(np.concatenate([hsc_i, leg_i], axis=0))

    n_hsc = hsc_p.shape[0]

    # ── basic UMAP scatter ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.scatter(emb_p[:n_hsc, 0], emb_p[:n_hsc, 1],
                s=5, c="blue", alpha=0.55, label="HSC", rasterized=True)
    ax1.scatter(emb_p[n_hsc:, 0], emb_p[n_hsc:, 1],
                s=5, c="orange", alpha=0.55, label="Legacy", rasterized=True)
    ax1.set_title(f"Physics encoder_1 — {variant}\n(dim={hsc_p.shape[1]})")
    ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.scatter(emb_i[:n_hsc, 0], emb_i[:n_hsc, 1],
                s=5, c="blue", alpha=0.55, label="HSC", rasterized=True)
    ax2.scatter(emb_i[n_hsc:, 0], emb_i[n_hsc:, 1],
                s=5, c="orange", alpha=0.55, label="Legacy", rasterized=True)
    ax2.set_title(f"Instrument encoder_2 — spatial_pooled\n(dim={hsc_i.shape[1]})")
    ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"UMAP HSC/Legacy pairs — base6x6/{variant} — N={n_hsc} pairs\n{ckpt_name}",
        fontsize=13,
    )
    plt.tight_layout()
    png_path = out_dir / f"umap_pairs_{variant}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path}")

    # ── save embeddings cache ──────────────────────────────────────────────────
    npz_path = out_dir / f"umap_pairs_{variant}_data.npz"
    np.savez_compressed(
        npz_path,
        hsc_phys=hsc_p, leg_phys=leg_p,
        hsc_inst=hsc_i, leg_inst=leg_i,
        emb_p=emb_p, emb_i=emb_i,
        n_hsc=np.int64(n_hsc),
    )
    print(f"saved {npz_path}")

    if post_discord:
        post_image(png_path, message=(
            f"📊 [base6x6/umap_pairs/{variant}] N={n_hsc} pairs, "
            f"phys_dim={hsc_p.shape[1]}, inst_dim={hsc_i.shape[1]}, ckpt={ckpt_name}"
        ))

    # ── pairs overlay ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(overlay_seed)
    pair_idx = rng.choice(n_hsc, size=min(overlay_k, n_hsc), replace=False)
    pair_idx.sort()
    print(f"overlay pair indices: {pair_idx.tolist()}")

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 9))
    _plot_overlay(emb_p, n_hsc, pair_idx, ax3,
                  f"Physics encoder_1 — {variant} — {len(pair_idx)} pairs")
    _plot_overlay(emb_i, n_hsc, pair_idx, ax4,
                  f"Instrument encoder_2 — {len(pair_idx)} pairs")
    fig2.suptitle(
        f"UMAP pairs — base6x6/{variant} — circles=HSC, squares=Legacy\n{ckpt_name}",
        fontsize=13,
    )
    plt.tight_layout()
    overlay_path = out_dir / f"umap_pairs_{variant}_overlay.png"
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {overlay_path}")

    elapsed = round(time.time() - t0, 1)
    meta = {
        "variant": variant, "n_pairs": int(n_hsc),
        "physics_dim": int(hsc_p.shape[1]), "instrument_dim": int(hsc_i.shape[1]),
        "ckpt": str(ckpt), "hdf5": str(hdf5_path),
        "elapsed_sec": elapsed,
        "data_npz": npz_path.name,
    }
    (out_dir / f"umap_pairs_{variant}.json").write_text(json.dumps(meta, indent=2))

    if post_discord:
        post_image(overlay_path, message=(
            f"📊 [base6x6/umap_pairs/{variant}] pairs overlay ({len(pair_idx)} pairs)"
        ))
        post_text(f"✅ [base6x6/umap_pairs/{variant}] done in {elapsed:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hdf5", default=NEIGHBORS_HDF5_DEFAULT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-discord", action="store_true")
    ap.add_argument("--overlay-k", type=int, default=8)
    ap.add_argument("--overlay-seed", type=int, default=0)
    args = ap.parse_args()

    try:
        run(args.variant, args.n, args.batch_size, args.ckpt, args.hdf5,
            args.out_dir, post_discord=not args.no_discord,
            overlay_k=args.overlay_k, overlay_seed=args.overlay_seed)
    except Exception as exc:
        if not args.no_discord:
            post_text(f"❌ [base6x6/umap_pairs/{args.variant}] FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
