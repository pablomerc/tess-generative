#!/usr/bin/env python3
"""Paper-style survey-colored UMAP for the JEPA ablation arms.

Same pipeline as diffusion_ablation/regenerate_umap.py (encode -> UMAP ->
visualization_scripts/plot_umap_from_file), but loads the JEPA modules and
encodes with their EMA target encoders (`encoder_galaxy` / `encoder_instrument`).
For the single-encoder arms (hsc-resnet, hsc-vit, cross) both panels are the
same space — metadata carries `single_encoder: true`.

Outputs land under jepa_ablation/results/umap/.

Example:
  python -m galaxy_images.galaxy_model.jepa_ablation.regenerate_umap_jepa \\
    --arm cf --checkpoint /orcd/pool/007/pablomer/checkpoints_new/jepa-cf/best-*.ckpt \\
    --num-anchors 2000 --tag jepa-cf-100k
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
VIS_DIR = GM_DIR / "visualization_scripts"
for p in (str(REPO_ROOT), str(GM_DIR), str(VIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import umap  # noqa: E402

from galaxy_images.galaxy_model.contrastive_baseline.neighbors_ram48 import (  # noqa: E402
    NeighborsRAM48Dataset,
)
from galaxy_images.galaxy_model.jepa_ablation.jepa_modules import (  # noqa: E402
    JEPACounterfactualModule,
    JEPACrossSurveyModule,
    JEPAHSCModule,
)
from galaxy_images.galaxy_model.jepa_ablation.jepa_vit import IJEPAViTModule  # noqa: E402
import plot_umap_from_file as plotter  # noqa: E402
from discord_notify import notify  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results" / "umap"

from galaxy_images.galaxy_model.jepa_ablation.autoencoder_baseline import AEHSCModule  # noqa: E402
from galaxy_images.galaxy_model.jepa_ablation.jepa_round3 import (  # noqa: E402
    CFTokenJEPAModule,
    XJEPAModule,
)

# arm -> (class, hparam key unique to that arm's signature, single_encoder)
ARM_SPECS = {
    "hsc-resnet": (JEPAHSCModule, "mask_frac_min", True),
    "hsc-vit": (IJEPAViTModule, "pred_dim", True),
    "multi-vit": (IJEPAViTModule, "pred_dim", True),
    "cf": (JEPACounterfactualModule, "lambda_galaxy", False),
    "cf-vit": (JEPACounterfactualModule, "lambda_galaxy", False),
    "cf-token": (CFTokenJEPAModule, "translator_depth", False),
    "xjepa": (XJEPAModule, "phys_dim", False),
    "cross": (JEPACrossSurveyModule, None, True),
    "ae-hsc": (AEHSCModule, "decoder_channels", True),
}


def load_jepa_checkpoint(ckpt_path: str, arm: str):
    """Load with an hparams cross-check so a wrong --arm cannot silently produce
    a figure from the wrong module class (same guard as the diffusion ablation)."""
    cls, marker, single = ARM_SPECS[arm]
    hp = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("hyper_parameters", {}) or {}
    # The AE baseline (marker: decoder_channels) is the one arm in this harness
    # with no EMA machinery — everything else must carry ema_base_momentum.
    if "ema_base_momentum" not in hp and marker != "decoder_channels":
        raise ValueError(f"{ckpt_path} has no 'ema_base_momentum' hparam — not a JEPA checkpoint.")
    if marker is not None and marker not in hp:
        raise ValueError(f"--arm {arm} expects hparam '{marker}', absent in {ckpt_path}.")
    if marker is None:  # cross: must not match any of the marked arms
        others = [m for _, m, _ in ARM_SPECS.values() if m is not None]
        hit = [m for m in others if m in hp]
        if hit:
            raise ValueError(f"--arm cross but checkpoint carries {hit} — wrong arm.")
    # lambda_galaxy is shared by cf/cf-vit/cf-token/xjepa; the round-3 classes
    # must not be loadable under the round-2 arm names (and vice versa).
    if marker == "lambda_galaxy" and any(k in hp for k in ("translator_depth", "phys_dim")):
        raise ValueError(f"--arm {arm} but checkpoint is a round-3 module (cf-token/xjepa).")
    if marker == "translator_depth" and "phys_dim" in hp:
        raise ValueError(f"--arm {arm} but checkpoint carries phys_dim — this is xjepa.")
    model = cls.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model, single, dict(hp)


def _collate(batch):
    targets = torch.stack([b[0] for b in batch], dim=0)
    samegals = torch.stack([b[1] for b in batch], dim=0)
    anchor_surveys = [b[3]["anchor_survey"] for b in batch]
    return targets, samegals, anchor_surveys


def _encode_in_batches(images, encoder, device, batch_size: int) -> torch.Tensor:
    out = []
    for i in range(0, images.shape[0], batch_size):
        chunk = images[i : i + batch_size].to(device, non_blocking=True)
        out.append(encoder(chunk).flatten(start_dim=1).detach().float().cpu())
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARM_SPECS), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/orcd/pool/007/pablomer/efficient_neighs_48"))
    parser.add_argument("--num-anchors", type=int, default=2000)
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--umap-seed", type=int, default=42)
    parser.add_argument("--webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = parser.parse_args(argv)

    job_id = os.environ.get("SLURM_JOB_ID", "<no-slurm>")
    host = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    t_start = time.perf_counter()

    notify(
        args.webhook,
        f"▶️ **UMAP jepa** start `{args.tag}` arm={args.arm} "
        f"num_anchors={args.num_anchors} host=`{host}` jobid=`{job_id}`\n"
        f"ckpt: `{args.checkpoint}`",
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, single_encoder, _hp = load_jepa_checkpoint(str(args.checkpoint), args.arm)
        model = model.to(device)
        print(f"[umap-jepa] loaded {type(model).__name__} single_encoder={single_encoder}")

        dataset = NeighborsRAM48Dataset(
            data_dir=str(args.data_dir), crop_size=args.crop_size,
            max_neighbors=0,
            # This job touches ~num_anchors rows once — memmap, don't pull 17 GB into RAM.
            load_into_ram=False,
        )
        if args.num_anchors > len(dataset):
            raise ValueError(f"num_anchors={args.num_anchors} > available={len(dataset)}")
        loader = DataLoader(
            Subset(dataset, range(args.num_anchors)),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate,
            persistent_workers=False,
        )

        t = time.perf_counter()
        hsc_images, legacy_images = _collect_hsc_legacy_pairs(loader, args.num_anchors)
        load_time = time.perf_counter() - t

        t = time.perf_counter()
        hsc_e1 = _encode_in_batches(hsc_images, model.encoder_galaxy, device, args.batch_size)
        leg_e1 = _encode_in_batches(legacy_images, model.encoder_galaxy, device, args.batch_size)
        if single_encoder:
            hsc_e2, leg_e2 = hsc_e1, leg_e1
        else:
            hsc_e2 = _encode_in_batches(hsc_images, model.encoder_instrument, device, args.batch_size)
            leg_e2 = _encode_in_batches(legacy_images, model.encoder_instrument, device, args.batch_size)
        encode_time = time.perf_counter() - t
        dim = int(hsc_e1.shape[1])
        print(f"[umap-jepa] encoded dim={dim} in {encode_time:.1f}s (load {load_time:.1f}s)")

        num_hsc = hsc_e1.shape[0]
        all_e1 = torch.cat([hsc_e1, leg_e1], dim=0).numpy()
        all_e2 = torch.cat([hsc_e2, leg_e2], dim=0).numpy()

        rng = np.random.default_rng(42)
        shuffled = rng.permutation(num_hsc)
        d_pair_1 = float(np.mean(np.linalg.norm(all_e1[:num_hsc] - all_e1[num_hsc:], axis=1)))
        d_unp_1 = float(np.mean(np.linalg.norm(all_e1[:num_hsc] - all_e1[num_hsc:][shuffled], axis=1)))
        d_pair_2 = float(np.mean(np.linalg.norm(all_e2[:num_hsc] - all_e2[num_hsc:], axis=1)))
        d_unp_2 = float(np.mean(np.linalg.norm(all_e2[:num_hsc] - all_e2[num_hsc:][shuffled], axis=1)))
        ratio_1 = d_pair_1 / d_unp_1 if d_unp_1 > 0 else float("nan")
        ratio_2 = d_pair_2 / d_unp_2 if d_unp_2 > 0 else float("nan")
        print(f"[umap-jepa] pair/unpaired ratios: e1={ratio_1:.4f} e2={ratio_2:.4f}")

        umap_params = dict(n_neighbors=15, min_dist=0.1, n_components=2,
                           metric="euclidean", random_state=int(args.umap_seed))
        t = time.perf_counter()
        emb_1 = umap.UMAP(**umap_params).fit_transform(all_e1)
        emb_2 = emb_1 if single_encoder else umap.UMAP(**umap_params).fit_transform(all_e2)
        umap_time = time.perf_counter() - t

        rng2 = np.random.default_rng(42)
        selected_indices = rng2.choice(num_hsc, size=min(20, num_hsc), replace=False).astype(np.int64)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"umap_both_encoders_zdim{dim}_zoom_flat_jepa-{args.arm}_{args.tag}"
        data_path = OUT_DIR / f"{stem}_data.npz"
        meta_path = OUT_DIR / f"{stem}_metadata.json"
        if data_path.exists() or meta_path.exists():
            raise FileExistsError(f"Refusing to overwrite {data_path.name} / {meta_path.name}")

        np.savez_compressed(
            data_path,
            hsc_umap_1=emb_1[:num_hsc], legacy_umap_1=emb_1[num_hsc:],
            hsc_umap_2=emb_2[:num_hsc], legacy_umap_2=emb_2[num_hsc:],
            selected_indices=selected_indices,
        )
        meta = {
            "epoch": int(args.epoch), "dim": dim, "mode_tag": "", "zoom_val": True,
            "avg_latent_space": False, "num_hsc": int(num_hsc), "tag": args.tag,
            "arm": f"jepa-{args.arm}", "model_class": type(model).__name__,
            "single_encoder": bool(single_encoder),
            "checkpoint_path": str(args.checkpoint), "data_dir": str(args.data_dir),
            "num_anchors": int(args.num_anchors), "umap_params": umap_params,
            "pair_distance_ratio_physics": ratio_1,
            "pair_distance_ratio_instrument": ratio_2,
            "encode_seconds": encode_time, "umap_seconds": umap_time,
            "host": host, "slurm_job_id": job_id,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[umap-jepa] wrote {data_path.name} + {meta_path.name}")

        dat = plotter.load_umap_data(data_path)
        png_path = plotter.plot(dat)

        total_time = time.perf_counter() - t_start
        notify(
            args.webhook,
            f"✅ `{args.tag}` UMAP done in {total_time:.1f}s — "
            f"pair-dist ratios e1={ratio_1:.3f} e2={ratio_2:.3f}"
            + (" (single-encoder: panels identical)" if single_encoder else ""),
            file_path=png_path,
        )
        print(f"[umap-jepa] total {total_time:.1f}s — png={png_path}")

    except Exception as e:
        tb = traceback.format_exc()
        notify(
            args.webhook,
            f"❌ `{args.tag}` UMAP failed: `{type(e).__name__}: {e}`\n```\n{tb[-1500:]}\n```",
        )
        raise


if __name__ == "__main__":
    main()
