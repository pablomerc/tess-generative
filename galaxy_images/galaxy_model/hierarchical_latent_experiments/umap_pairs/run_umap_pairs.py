"""UMAP of HSC/Legacy pairs in a given physics latent + the instrument latent.

Usage:
  python -m galaxy_images.galaxy_model.hierarchical_latent_experiments.umap_pairs.run_umap_pairs \
      --variant global_vec --n 4096 --ckpt <path> --out-dir outputs/global_vec
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

from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.ckpt import (
    DEFAULT_CKPT, load_trained,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.data_loaders import (
    NEIGHBORS_HDF5_DEFAULT, make_loader, make_pair_dataset,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import (
    post_image, post_text,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.latents import (
    VARIANTS, extract_both,
)


UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, n_components=2,
                   metric="euclidean", random_state=42)


def encode_pairs(model, dataset, variant: str, batch_size: int, device: torch.device):
    loader = make_loader(dataset, batch_size=batch_size, num_workers=2, pin_memory=(device.type == "cuda"))
    hsc_phys, leg_phys, hsc_inst, leg_inst = [], [], [], []
    for hsc, leg, _, _ in tqdm(loader, desc=f"encode[{variant}]"):
        hsc = hsc.to(device, non_blocking=True)
        leg = leg.to(device, non_blocking=True)
        hp, hi = extract_both(model, hsc, variant)
        lp, li = extract_both(model, leg, variant)
        hsc_phys.append(hp.float().cpu().numpy())
        leg_phys.append(lp.float().cpu().numpy())
        hsc_inst.append(hi.float().cpu().numpy())
        leg_inst.append(li.float().cpu().numpy())
    return (np.concatenate(hsc_phys), np.concatenate(leg_phys),
            np.concatenate(hsc_inst), np.concatenate(leg_inst))


def run(variant: str, n: int, batch_size: int, ckpt: str, hdf5_path: str,
        out_dir: Path, post_discord: bool):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if post_discord:
        post_text(f"🟢 [umap_pairs/{variant}] starting — n={n}, ckpt={Path(ckpt).name}")

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, variant={variant}, n={n}")

    model = load_trained(ckpt, device=device)
    dataset = make_pair_dataset(hdf5_path=hdf5_path, n=n, shuffle=False)
    print(f"dataset size requested={n}, available={len(dataset)}")

    hsc_p, leg_p, hsc_i, leg_i = encode_pairs(model, dataset, variant, batch_size, device)
    print(f"physics: hsc={hsc_p.shape}, leg={leg_p.shape}; instrument: hsc={hsc_i.shape}, leg={leg_i.shape}")

    print("UMAP physics …")
    emb_p = umap.UMAP(**UMAP_PARAMS).fit_transform(np.concatenate([hsc_p, leg_p], axis=0))
    print("UMAP instrument …")
    emb_i = umap.UMAP(**UMAP_PARAMS).fit_transform(np.concatenate([hsc_i, leg_i], axis=0))

    n_hsc = hsc_p.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.scatter(emb_p[:n_hsc, 0], emb_p[:n_hsc, 1], s=5, c="blue", alpha=0.55, label="HSC", rasterized=True)
    ax1.scatter(emb_p[n_hsc:, 0], emb_p[n_hsc:, 1], s=5, c="orange", alpha=0.55, label="Legacy", rasterized=True)
    ax1.set_title(f"Physics latent — {variant}\n(dim={hsc_p.shape[1]})")
    ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.scatter(emb_i[:n_hsc, 0], emb_i[:n_hsc, 1], s=5, c="blue", alpha=0.55, label="HSC", rasterized=True)
    ax2.scatter(emb_i[n_hsc:, 0], emb_i[n_hsc:, 1], s=5, c="orange", alpha=0.55, label="Legacy", rasterized=True)
    ax2.set_title(f"Instrument latent — instrument.flat\n(dim={hsc_i.shape[1]})")
    ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(f"UMAP HSC/Legacy pairs — variant={variant} — N={n_hsc} pairs", fontsize=13)
    plt.tight_layout()

    png_path = out_dir / f"umap_pairs_{variant}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path}")

    npz_path = out_dir / f"umap_pairs_{variant}_data.npz"
    np.savez_compressed(
        npz_path,
        hsc_phys=hsc_p, leg_phys=leg_p,
        hsc_inst=hsc_i, leg_inst=leg_i,
        emb_p=emb_p, emb_i=emb_i,
        n_hsc=np.int64(n_hsc),
    )
    print(f"saved {npz_path}")

    meta = {
        "variant": variant, "n_pairs": int(n_hsc),
        "physics_dim": int(hsc_p.shape[1]), "instrument_dim": int(hsc_i.shape[1]),
        "ckpt": str(ckpt), "hdf5": str(hdf5_path),
        "elapsed_sec": round(time.time() - t0, 1),
        "data_npz": npz_path.name,
    }
    (out_dir / f"umap_pairs_{variant}.json").write_text(json.dumps(meta, indent=2))

    if post_discord:
        post_image(png_path, message=f"📊 [umap_pairs/{variant}] N={n_hsc} pairs, "
                                     f"phys_dim={hsc_p.shape[1]}, inst_dim={hsc_i.shape[1]}, "
                                     f"{meta['elapsed_sec']:.0f}s")
        post_text(f"✅ [umap_pairs/{variant}] done in {meta['elapsed_sec']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--hdf5", default=NEIGHBORS_HDF5_DEFAULT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    try:
        run(args.variant, args.n, args.batch_size, args.ckpt, args.hdf5,
            args.out_dir, post_discord=not args.no_discord)
    except Exception as exc:
        if not args.no_discord:
            post_text(f"❌ [umap_pairs/{args.variant}] FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
