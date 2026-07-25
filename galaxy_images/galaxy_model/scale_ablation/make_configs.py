#!/usr/bin/env python
"""Emit one training config per ladder rung, derived from the diffusion-ablation FM control.

Deriving rather than hand-writing guarantees the arms are step-matched to
neighbors_fm_control.json (75k steps, batch 64 x 4 devices = effective 256, seed 42,
max_neighbors=5) with no transcription drift. Only these keys differ per arm:

  run.output_dir                      unique dir per arm -- the variant name is
                                      `neighbors_all_attn` for every arm, so a shared
                                      output_dir would collide (and each of the 4 DDP
                                      ranks creates its own dated subdir, so 4 arms
                                      would produce 16 indistinguishable directories)
  data.train_subset_json              which rung
  data.train_subset_tile_to           constant epoch length across arms (LR-period fix)
  data.downstream_holdout_ids_txt     eval-set holdout (same file for every arm)
  trainer.checkpoint_every_n_train_steps  5000 instead of 1000 -- cuts write traffic on
                                      the shared filesystem from ~220 GB to ~44 GB/run
  wandb.name                          per-arm tag; also names the best-ckpt subdir
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GM = HERE.parent
BASE_CONFIG = GM / "diffusion_ablation/configs/neighbors_fm_control.json"
HOLDOUT = GM / "downstream_evaluation/engaging/outputs/index/holdout_legacy_ids.txt"
CKPT_ROOT = Path("/orcd/pool/007/pablomer/checkpoints_new/scale_ablation")

LADDER = [1000, 3162, 10000, 31622]


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text())
    manifest = json.loads((HERE / "subsets/manifest.json").read_text())
    tile_to = manifest["tile_to"]
    assert manifest["holdout_applied"], "subsets were built without the holdout; regenerate"

    out_dir = HERE / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in LADDER:
        cfg = json.loads(json.dumps(base))  # deep copy
        tag = f"scale-{n}-75k"

        cfg["run"]["output_dir"] = str(CKPT_ROOT / "runs" / f"scale_{n}")
        cfg["run"]["shared_checkpoint_dir"] = str(CKPT_ROOT / "best")

        cfg["data"]["downstream_holdout_ids_txt"] = str(HOLDOUT)
        cfg["data"]["train_subset_json"] = str(HERE / "subsets" / f"scale_{n}.json")
        cfg["data"]["train_subset_tile_to"] = tile_to

        cfg["trainer"]["checkpoint_every_n_train_steps"] = 5000

        # W&B: keep the SAME project as the diffusion ablation so the free
        # fm-control-ram48-h200-75k run is directly comparable in the same workspace, and
        # use group/tags to make the four arms a family. extra_config carries the flat,
        # numeric sweep axis -- a config path string is not plottable, `scale/n_anchors` is.
        rung = next(r for r in manifest["rungs"] if r["n_anchors"] == n)
        cfg["wandb"]["name"] = tag
        cfg["wandb"]["group"] = "scale-ablation-75k"
        cfg["wandb"]["job_type"] = "train"
        cfg["wandb"]["tags"] = ["scale-ablation", "flow-matching", "75k", "holdout", f"n{n}"]
        cfg["wandb"]["extra_config"] = {
            "scale/n_anchors": n,
            "scale/frac_of_train": rung["frac_of_train"],
            "scale/distinct_neighbour_rows": rung["distinct_neighbour_rows"],
            "scale/distinct_images_total": rung["distinct_images_total"],
            "scale/repeats_per_epoch": rung["repeats_per_epoch"],
            "scale/train_pool": manifest["train_size"],
            "scale/tile_to": tile_to,
            "scale/steps_per_epoch": manifest["steps_per_epoch"],
            "scale/holdout_applied": True,
            "scale/val_size": manifest["val_size"],
        }

        path = out_dir / f"scale_{n}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")

        # sanity: the knobs that must NOT drift from the FM control
        for section, key in [
            ("trainer", "num_steps"),
            ("trainer", "devices"),
            ("trainer", "seed"),
            ("trainer", "scale_steps_by_devices"),
            ("data", "batch_size"),
            ("data", "max_neighbors"),
            ("data", "val_ratio"),
            ("model", "lr"),
        ]:
            assert cfg[section][key] == base[section][key], f"{section}.{key} drifted"

        print(f"  wrote {path.name:<20} tag={tag:<18} subset=scale_{n}.json tile_to={tile_to:,}")

    print(f"\nstep-matched to {BASE_CONFIG.name}: "
          f"num_steps={base['trainer']['num_steps']:,}, "
          f"devices={base['trainer']['devices']}, batch={base['data']['batch_size']} "
          f"(effective {base['trainer']['devices'] * base['data']['batch_size']}), "
          f"seed={base['trainer']['seed']}, max_neighbors={base['data']['max_neighbors']}")


if __name__ == "__main__":
    main()
