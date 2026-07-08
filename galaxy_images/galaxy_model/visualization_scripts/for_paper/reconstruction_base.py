"""Generate reconstruction plots from the registry's `base` ConditionalFlowMatchingModule snapshot.

Picks `--num-examples` random anchors from NeighborsEfficientDataset, generates `--num-samples`
flow-matching reconstructions per anchor, saves the full sampled batch to an HDF5 file, then
renders both the raw `reconstruction_plot.png` (one row per anchor, every sample + mean) and the
styled `reconstruction_all.png` via the existing `replot_reconstruction.plot_examples`. Both PNGs
are posted to Discord. Refuses to clobber an existing `<output_dir>/<tag>/` to honour the
no-overwrite policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.backends.cuda.preferred_blas_library("hipblas")

from torch.utils.data import DataLoader, Subset

# --- Path setup so plain `python -m` / direct invocation both work. ---
SCRIPT_DIR = Path(__file__).resolve().parent              # .../visualization_scripts/for_paper
VIZ_DIR = SCRIPT_DIR.parent                               # .../visualization_scripts
GM_DIR = VIZ_DIR.parent                                   # .../galaxy_model
GI_DIR = GM_DIR.parent                                    # .../galaxy_images
REPO_ROOT = GI_DIR.parent                                 # .../tess-generative
for p in (str(REPO_ROOT), str(GM_DIR), str(VIZ_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from neighbors_efficient import NeighborsEfficientDataset
from neighbors import collate_neighbors
from double_train_fm_neighbors import ConditionalFlowMatchingModule
from discord_notify import notify
from replot_reconstruction import plot_examples


COLOR_HSC = "#e8c4a0"
COLOR_LEGACY = "#8eb8e8"


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


def _row_scale_rgb(x_chw: torch.Tensor, vmin, vmax) -> torch.Tensor:
    x = x_chw[:3]
    vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = ((x - vmin_t) / (vmax_t - vmin_t + 1e-8)).clamp(0, 1)
    return y.permute(1, 2, 0)


def _collect_random_batch(dataset, indices, max_neighbors_in_dataset):
    """Build a single padded batch from a list of dataset indices."""
    items = [dataset[int(i)] for i in indices]
    targets, samegals, sameins, masks, metadata = collate_neighbors(items)
    print(
        f"[recon-base] batched {len(indices)} examples — targets={tuple(targets.shape)} "
        f"sameins={tuple(sameins.shape)} masks={tuple(masks.shape)}"
    )
    return targets, samegals, sameins, masks, metadata


def _load_anchor_directed(dataset, anchor_pos: int, target_survey: str):
    """Build a single (target, samegal, sameins, metadata) item with the target survey forced.

    Bypasses NeighborsEfficientDataset.__getitem__'s `idx % 2` direction selection so the same
    anchor row can be loaded as HSC-target *and* Legacy-target.
    """
    row_idx = int(dataset.anchor_indices[anchor_pos])
    hsc_img = dataset._preprocess(row_idx, "hsc")[:4]   # drop y band -> (4, H, W)
    legacy_img = dataset._preprocess(row_idx, "legacy")  # (4, H, W)

    if target_survey == "hsc":
        target, samegal = hsc_img, legacy_img
        neighbor_row_ids = dataset.neighbor_idx_hsc[row_idx]
        sameins_survey = "hsc"
    elif target_survey == "legacy":
        target, samegal = legacy_img, hsc_img
        neighbor_row_ids = dataset.neighbor_idx_legacy[row_idx]
        sameins_survey = "legacy"
    else:
        raise ValueError(f"target_survey must be 'hsc' or 'legacy', got {target_survey!r}")

    valid_ids = neighbor_row_ids[neighbor_row_ids >= 0][: dataset.max_neighbors]
    sameins_list = []
    for nid in valid_ids:
        nimg = dataset._preprocess(int(nid), sameins_survey)
        if sameins_survey == "hsc":
            nimg = nimg[:4]
        sameins_list.append(nimg)
    if sameins_list:
        sameins = torch.stack(sameins_list, dim=0)
    else:
        sameins = torch.empty(0, 4, dataset.crop_size, dataset.crop_size)

    metadata = {
        "anchor_survey": target_survey,
        "idx": row_idx,
        "num_same_instrument": len(sameins_list),
    }
    return target, samegal, sameins, metadata


def _collect_both_directions_batch(dataset, anchor_positions):
    """Build a 2N-item batch where consecutive rows are HSC-target / Legacy-target for the same anchor."""
    items = []
    for pos in anchor_positions:
        items.append(_load_anchor_directed(dataset, int(pos), "hsc"))
        items.append(_load_anchor_directed(dataset, int(pos), "legacy"))
    targets, samegals, sameins, masks, metadata = collate_neighbors(items)
    print(
        f"[recon-base] both-dirs: {len(anchor_positions)} anchors -> {len(items)} rows | "
        f"targets={tuple(targets.shape)} sameins={tuple(sameins.shape)}"
    )
    return targets, samegals, sameins, masks, metadata


def _generate_reconstructions(model, targets, samegals, sameins, masks, device, num_samples):
    """Generate `num_samples` reconstructions per example.

    Returns:
        samples: (B, num_samples, C, H, W) on CPU
        mean_samples: (B, C, H, W) on CPU
    """
    samegals = samegals.to(device)
    sameins = sameins.to(device)
    masks = masks.to(device)

    all_samples = []
    for i in range(targets.shape[0]):
        sg = samegals[i : i + 1].repeat(num_samples, 1, 1, 1)
        si = sameins[i : i + 1].repeat(num_samples, 1, 1, 1, 1)
        mk = masks[i : i + 1].repeat(num_samples, 1)
        s = model.sample(sg, si, masks=mk)
        all_samples.append(s.detach().cpu().unsqueeze(0))
        print(f"[recon-base]   sampled example {i + 1}/{targets.shape[0]}")
    samples = torch.cat(all_samples, dim=0)
    mean_samples = samples.mean(dim=1)
    return samples, mean_samples


def _save_h5(path, targets, samegals, sameins, masks, samples, mean_samples, metadata):
    """Write the H5 file in the schema `replot_reconstruction.load_data` expects."""
    with h5py.File(path, "w") as f:
        f.create_dataset("targets", data=targets.cpu().numpy(), compression="gzip")
        f.create_dataset("samegals", data=samegals.cpu().numpy(), compression="gzip")
        f.create_dataset("sameins", data=sameins.cpu().numpy(), compression="gzip")
        f.create_dataset("masks", data=masks.cpu().numpy(), compression="gzip")
        f.create_dataset("samples", data=samples.cpu().numpy(), compression="gzip")
        f.create_dataset("mean_samples", data=mean_samples.cpu().numpy(), compression="gzip")
        surveys = [m.get("anchor_survey", "unknown") for m in metadata]
        f.create_dataset("anchor_surveys", data=np.array(surveys, dtype="S10"))
        idxs = [int(m.get("idx", -1)) for m in metadata]
        f.create_dataset("indices", data=np.array(idxs, dtype=np.int64))
        nsi = [int(m.get("num_same_instrument", -1)) for m in metadata]
        f.create_dataset("num_same_instrument", data=np.array(nsi, dtype=np.int64))
        f.attrs["batch_size"] = targets.shape[0]
        f.attrs["num_samples"] = samples.shape[1]
        f.attrs["image_channels"] = targets.shape[1]
        f.attrs["image_height"] = targets.shape[2]
        f.attrs["image_width"] = targets.shape[3]
        f.attrs["num_neighbors"] = sameins.shape[1]


def _raw_plot(targets, samegals, sameins, samples, mean_samples, metadata, output_path):
    """Reproduce the raw layout: SameGal | SameIns(1st) | Target | Sample×N | Mean."""
    B, num_samples = samples.shape[0], samples.shape[1]
    num_cols = 3 + num_samples + 1

    fig, axes = plt.subplots(B, num_cols, figsize=(2 * num_cols, 2 * B), squeeze=False)
    titles = ["SameGal", "SameIns (1st)", "Target"] + [f"Sample {j + 1}" for j in range(num_samples)] + ["Mean"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=10)

    for i in range(B):
        target = targets[i]
        vmin = target[:3].amin(dim=(1, 2))
        vmax = target[:3].amax(dim=(1, 2))

        axes[i, 0].imshow(_row_scale_rgb(samegals[i, :3], vmin, vmax).numpy())
        axes[i, 0].axis("off")

        survey = metadata[i].get("anchor_survey", "unknown")
        c = COLOR_HSC if survey == "hsc" else COLOR_LEGACY
        axes[i, 0].text(
            -0.1, 0.5, f"{survey.upper()}\n[{i}]",
            transform=axes[i, 0].transAxes, ha="right", va="center",
            fontsize=9, weight="bold",
            bbox=dict(boxstyle="round", facecolor=c, alpha=0.5),
        )

        axes[i, 1].imshow(_row_scale_rgb(sameins[i, 0, :3], vmin, vmax).numpy())
        axes[i, 1].axis("off")

        axes[i, 2].imshow(_row_scale_rgb(target[:3], vmin, vmax).numpy())
        axes[i, 2].axis("off")

        for j in range(num_samples):
            axes[i, 3 + j].imshow(_row_scale_rgb(samples[i, j, :3], vmin, vmax).numpy())
            axes[i, 3 + j].axis("off")

        axes[i, -1].imshow(_row_scale_rgb(mean_samples[i, :3], vmin, vmax).numpy())
        axes[i, -1].axis("off")

    plt.suptitle("Reconstruction samples — base checkpoint", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-neighbors", type=int, default=5,
                        help="Cap on same-instrument neighbors loaded per anchor (matches training default).")
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--both-directions", action="store_true",
                        help="For each picked anchor, render two consecutive rows: HSC-target and Legacy-target.")
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--output-root", type=Path,
                        default=SCRIPT_DIR / "reconstruction_outputs_base")
    parser.add_argument("--webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "<no-slurm>")
    host = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    t_start = time.perf_counter()

    out_dir = args.output_root / args.tag
    if out_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output dir: {out_dir}. "
            f"Pick a fresh --tag."
        )
    out_dir.mkdir(parents=True, exist_ok=False)

    notify(
        args.webhook,
        f"▶️ **reconstruction base** start `{args.tag}`  "
        f"num_examples={args.num_examples}  num_samples={args.num_samples}  seed={args.seed}  "
        f"host=`{host}`  jobid=`{job_id}`\nckpt: `{args.checkpoint}`",
    )

    try:
        device = _pick_device()
        print(f"[recon-base] device={device}, tag={args.tag}, out={out_dir}")

        t = time.perf_counter()
        model = ConditionalFlowMatchingModule.load_from_checkpoint(
            str(args.checkpoint), map_location="cpu",
        )
        model.eval()
        torch.set_grad_enabled(False)
        model = model.to(device)
        print(f"[recon-base] model loaded in {time.perf_counter() - t:.1f}s")

        dataset = NeighborsEfficientDataset(
            data_dir=str(args.data_dir),
            crop_size=args.crop_size,
            max_neighbors=args.max_neighbors,
        )
        rng = np.random.default_rng(args.seed)
        if args.num_examples > len(dataset):
            raise ValueError(
                f"num_examples={args.num_examples} > anchors available={len(dataset)}"
            )
        chosen = rng.choice(len(dataset), size=args.num_examples, replace=False)
        chosen.sort()
        print(f"[recon-base] picked {len(chosen)} anchor positions (first 10: {chosen[:10].tolist()})")

        if args.both_directions:
            targets, samegals, sameins, masks, metadata = _collect_both_directions_batch(
                dataset, chosen
            )
        else:
            targets, samegals, sameins, masks, metadata = _collect_random_batch(
                dataset, chosen, args.max_neighbors
            )

        t = time.perf_counter()
        samples, mean_samples = _generate_reconstructions(
            model, targets, samegals, sameins, masks, device, args.num_samples
        )
        sample_time = time.perf_counter() - t
        print(f"[recon-base] generated {args.num_samples} samples × {targets.shape[0]} examples in {sample_time:.1f}s")

        h5_path = out_dir / "reconstruction_data.h5"
        _save_h5(h5_path, targets, samegals, sameins, masks, samples, mean_samples, metadata)
        print(f"[recon-base] wrote {h5_path.name} ({h5_path.stat().st_size / 1e6:.1f} MB)")

        raw_path = out_dir / "reconstruction_plot.png"
        _raw_plot(
            targets.cpu(), samegals.cpu(), sameins.cpu(),
            samples.cpu(), mean_samples.cpu(), metadata, raw_path,
        )
        print(f"[recon-base] wrote {raw_path.name}")

        styled_path = out_dir / "reconstruction_all.png"
        plot_examples(
            list(range(targets.shape[0])),
            targets.cpu(), samegals.cpu(), sameins.cpu(),
            samples.cpu(), mean_samples.cpu(), metadata,
            styled_path,
            row_numbers=list(range(1, targets.shape[0] + 1)),
        )
        print(f"[recon-base] wrote {styled_path.name}")

        manifest = {
            "tag": args.tag,
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "num_examples": int(args.num_examples),
            "num_samples": int(args.num_samples),
            "max_neighbors": int(args.max_neighbors),
            "crop_size": int(args.crop_size),
            "seed": int(args.seed),
            "both_directions": bool(args.both_directions),
            "chosen_indices": chosen.tolist(),
            "anchor_surveys": [m.get("anchor_survey", "unknown") for m in metadata],
            "row_numbers": list(range(1, targets.shape[0] + 1)),
            "host": host,
            "slurm_job_id": job_id,
            "sample_seconds": sample_time,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        total = time.perf_counter() - t_start
        notify(
            args.webhook,
            f"📷 `{args.tag}` raw plot — {targets.shape[0]} examples, "
            f"{args.num_samples} samples each",
            file_path=raw_path,
        )
        notify(
            args.webhook,
            f"🎨 `{args.tag}` styled plot",
            file_path=styled_path,
        )
        notify(
            args.webhook,
            f"✅ `{args.tag}` done in {total:.1f}s — out: `{out_dir}`",
        )
        print(f"[recon-base] total {total:.1f}s")

    except Exception as e:
        tb = traceback.format_exc()
        notify(
            args.webhook,
            f"❌ `{args.tag}` failed: `{type(e).__name__}: {e}`\n```\n{tb[-1500:]}\n```",
        )
        raise


if __name__ == "__main__":
    main()
