#!/usr/bin/env python3
"""UMAP regeneration for the diffusion-vs-FM ablation arms.

Same pipeline as visualization_scripts/regenerate_umap_base.py (encode → UMAP →
plot_umap_from_file), but loads either ConditionalFlowMatchingModule or
ConditionalDDPMModule. Outputs (npz / metadata / PNG) land under
diffusion_ablation/results/umap/ so paper visualization artifacts are untouched.

Examples:
  python -m galaxy_images.galaxy_model.diffusion_ablation.regenerate_umap \\
    --arm fm --checkpoint /path/to/latest-step=75000.ckpt \\
    --data-dir /orcd/pool/007/pablomer/efficient_neighs_48 \\
    --num-anchors 2000 --tag fm-control-75k --epoch 75000

  python -m galaxy_images.galaxy_model.diffusion_ablation.regenerate_umap \\
    --arm ddpm --checkpoint /path/to/latest-step=75000.ckpt \\
    --data-dir /orcd/pool/007/pablomer/efficient_neighs_48 \\
    --num-anchors 2000 --tag ddpm-eps-75k --epoch 75000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[3]
GM_DIR = Path(__file__).resolve().parents[1]
ABL_DIR = Path(__file__).resolve().parent
VIS_DIR = GM_DIR / "visualization_scripts"
for p in (str(REPO_ROOT), str(GM_DIR), str(VIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import umap

from galaxy_images.galaxy_model.contrastive_baseline.neighbors_ram48 import (
    NeighborsRAM48Dataset,
)
from galaxy_images.galaxy_model.diffusion_ablation.double_train_ddpm_neighbors import (
    ConditionalDDPMModule,
)
from galaxy_images.galaxy_model.double_train_fm_neighbors import (
    ConditionalFlowMatchingModule,
)
from galaxy_images.galaxy_model.neighbors_efficient import NeighborsEfficientDataset
import plot_umap_from_file as plotter
from discord_notify import notify

OUT_DIR = ABL_DIR / "results" / "umap"

ARM_TO_CLS = {
    "fm": ConditionalFlowMatchingModule,
    "ddpm": ConditionalDDPMModule,
}


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


def _collate(batch):
    """Return (target, samegal, anchor_surveys) — drop sameins (unused for UMAP)."""
    targets = torch.stack([b[0] for b in batch], dim=0)
    samegals = torch.stack([b[1] for b in batch], dim=0)
    anchor_surveys = [b[3]["anchor_survey"] for b in batch]
    return targets, samegals, anchor_surveys


def _encode_in_batches(images, encoder, device, batch_size: int) -> torch.Tensor:
    out = []
    for i in range(0, images.shape[0], batch_size):
        chunk = images[i : i + batch_size].to(device, non_blocking=True)
        emb = encoder(chunk)
        out.append(emb.detach().cpu())
    return torch.cat(out, dim=0)


def _collect_hsc_legacy_pairs(loader, num_anchors: int):
    hsc_list, legacy_list = [], []
    seen = 0
    for target, samegal, surveys in loader:
        for j, surv in enumerate(surveys):
            if surv == "hsc":
                hsc_list.append(target[j])
                legacy_list.append(samegal[j])
            else:
                hsc_list.append(samegal[j])
                legacy_list.append(target[j])
        seen += len(surveys)
        if seen >= num_anchors:
            break
    return (
        torch.stack(hsc_list, dim=0)[:num_anchors],
        torch.stack(legacy_list, dim=0)[:num_anchors],
    )


def _build_dataset(data_dir: Path, crop_size: int, data_backend: str):
    if data_backend == "ram48":
        return NeighborsRAM48Dataset(
            data_dir=str(data_dir),
            crop_size=crop_size,
            max_neighbors=0,
            load_into_ram=True,
        )
    if data_backend == "efficient":
        return NeighborsEfficientDataset(
            data_dir=str(data_dir),
            crop_size=crop_size,
            max_neighbors=0,
        )
    raise ValueError(f"Unknown --data-backend={data_backend!r}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=sorted(ARM_TO_CLS),
        required=True,
        help="Which LightningModule to load (fm or ddpm).",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--data-backend",
        choices=("ram48", "efficient"),
        default="ram48",
        help="Dataset class (default ram48 for Engaging ablation).",
    )
    parser.add_argument("--num-anchors", required=True, type=int)
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--umap-seed", type=int, default=42)
    parser.add_argument("--webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = parser.parse_args(argv)

    model_cls = ARM_TO_CLS[args.arm]
    job_id = os.environ.get("SLURM_JOB_ID", "<no-slurm>")
    host = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    t_start = time.perf_counter()

    notify(
        args.webhook,
        f"▶️ **UMAP ablation** start `{args.tag}` arm={args.arm}  "
        f"num_anchors={args.num_anchors} host=`{host}` jobid=`{job_id}`\n"
        f"ckpt: `{args.checkpoint}`",
    )

    try:
        device = _pick_device()
        print(
            f"[umap-ablation] device={device} arm={args.arm} tag={args.tag} "
            f"cls={model_cls.__name__}"
        )

        t = time.perf_counter()
        model = model_cls.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
        model.eval()
        torch.set_grad_enabled(False)
        model = model.to(device)
        dim = int(model.hparams.cross_attention_dim)
        print(
            f"[umap-ablation] model loaded in {time.perf_counter() - t:.1f}s  "
            f"cross_attention_dim={dim}"
        )

        dataset = _build_dataset(args.data_dir, args.crop_size, args.data_backend)
        n_avail = len(dataset)
        if args.num_anchors > n_avail:
            raise ValueError(f"num_anchors={args.num_anchors} > available anchors={n_avail}")
        loader = DataLoader(
            Subset(dataset, range(args.num_anchors)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate,
            persistent_workers=False,
        )

        t = time.perf_counter()
        hsc_images, legacy_images = _collect_hsc_legacy_pairs(loader, args.num_anchors)
        load_time = time.perf_counter() - t
        print(
            f"[umap-ablation] pairs hsc={tuple(hsc_images.shape)} "
            f"legacy={tuple(legacy_images.shape)} in {load_time:.1f}s"
        )

        t = time.perf_counter()
        hsc_e1 = _encode_in_batches(hsc_images, model.encoder_1, device, args.batch_size)
        leg_e1 = _encode_in_batches(legacy_images, model.encoder_1, device, args.batch_size)
        hsc_e2 = _encode_in_batches(hsc_images, model.encoder_2, device, args.batch_size)
        leg_e2 = _encode_in_batches(legacy_images, model.encoder_2, device, args.batch_size)
        encode_time = time.perf_counter() - t
        print(
            f"[umap-ablation] encoded e1={tuple(hsc_e1.shape)} e2={tuple(hsc_e2.shape)} "
            f"in {encode_time:.1f}s"
        )

        notify(
            args.webhook,
            f"🧠 `{args.tag}` encoded {args.num_anchors} pairs in {encode_time:.1f}s "
            f"(load {load_time:.1f}s). Running UMAP…",
        )

        num_hsc = hsc_e1.shape[0]
        all_e1 = torch.cat([hsc_e1, leg_e1], dim=0).flatten(start_dim=1).numpy()
        all_e2 = torch.cat([hsc_e2, leg_e2], dim=0).flatten(start_dim=1).numpy()

        hsc_flat_1, legacy_flat_1 = all_e1[:num_hsc], all_e1[num_hsc:]
        hsc_flat_2, legacy_flat_2 = all_e2[:num_hsc], all_e2[num_hsc:]
        rng = np.random.default_rng(42)
        shuffled = rng.permutation(num_hsc)
        d_pair_1 = float(np.mean(np.linalg.norm(hsc_flat_1 - legacy_flat_1, axis=1)))
        d_unp_1 = float(np.mean(np.linalg.norm(hsc_flat_1 - legacy_flat_1[shuffled], axis=1)))
        d_pair_2 = float(np.mean(np.linalg.norm(hsc_flat_2 - legacy_flat_2, axis=1)))
        d_unp_2 = float(np.mean(np.linalg.norm(hsc_flat_2 - legacy_flat_2[shuffled], axis=1)))
        ratio_1 = d_pair_1 / d_unp_1 if d_unp_1 > 0 else float("nan")
        ratio_2 = d_pair_2 / d_unp_2 if d_unp_2 > 0 else float("nan")
        print(
            f"[umap-ablation] pair/unpaired ratios: physics={ratio_1:.4f} "
            f"instrument={ratio_2:.4f}"
        )

        umap_params = dict(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="euclidean",
            random_state=int(args.umap_seed),
        )
        t = time.perf_counter()
        emb_1 = umap.UMAP(**umap_params).fit_transform(all_e1)
        emb_2 = umap.UMAP(**umap_params).fit_transform(all_e2)
        umap_time = time.perf_counter() - t
        print(f"[umap-ablation] UMAP done in {umap_time:.1f}s")

        rng2 = np.random.default_rng(42)
        n_pairs_to_highlight = min(20, num_hsc)
        selected_indices = rng2.choice(
            num_hsc, size=n_pairs_to_highlight, replace=False
        ).astype(np.int64)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"umap_both_encoders_zdim{dim}_zoom_flat_{args.tag}"
        data_path = OUT_DIR / f"{stem}_data.npz"
        meta_path = OUT_DIR / f"{stem}_metadata.json"
        if data_path.exists() or meta_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing files for tag '{args.tag}': "
                f"{data_path.name} / {meta_path.name}"
            )

        np.savez_compressed(
            data_path,
            hsc_umap_1=emb_1[:num_hsc],
            legacy_umap_1=emb_1[num_hsc:],
            hsc_umap_2=emb_2[:num_hsc],
            legacy_umap_2=emb_2[num_hsc:],
            selected_indices=selected_indices,
        )
        meta = {
            "epoch": int(args.epoch),
            "dim": dim,
            "mode_tag": "",
            "zoom_val": True,
            "avg_latent_space": False,
            "num_hsc": int(num_hsc),
            "tag": args.tag,
            "arm": args.arm,
            "model_class": model_cls.__name__,
            "checkpoint_path": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "data_backend": args.data_backend,
            "num_anchors": int(args.num_anchors),
            "umap_params": umap_params,
            "pair_distance_ratio_physics": ratio_1,
            "pair_distance_ratio_instrument": ratio_2,
            "encode_seconds": encode_time,
            "umap_seconds": umap_time,
            "host": host,
            "slurm_job_id": job_id,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[umap-ablation] wrote {data_path.name} + {meta_path.name}")

        dat = plotter.load_umap_data(data_path)
        png_path = plotter.plot(dat)

        total_time = time.perf_counter() - t_start
        notify(
            args.webhook,
            f"✅ `{args.tag}` done in {total_time:.1f}s — "
            f"physics ratio={ratio_1:.3f}, instrument ratio={ratio_2:.3f}\n"
            f"`{Path(png_path).name}`",
            file_path=png_path,
        )
        print(f"[umap-ablation] total {total_time:.1f}s — png={png_path}")

    except Exception as e:
        tb = traceback.format_exc()
        notify(
            args.webhook,
            f"❌ `{args.tag}` failed: `{type(e).__name__}: {e}`\n```\n{tb[-1500:]}\n```",
        )
        raise


if __name__ == "__main__":
    main()
