# tess-generative

Dual-encoder generative models that separate **physics** (source) from **instrument** signal in
galaxy survey images, trained on paired HSC + Legacy Survey cutouts. The main model is conditional
**flow matching**: two ResNet18 encoders feed a `diffusers` UNet through cross-attention, and the
counterfactual reconstruction objective forces the two latent spaces apart.

> **Detailed orientation lives in [`CLAUDE.md`](CLAUDE.md)** — cluster setup, how to submit jobs, the
> variant system, data formats, downstream evaluation, and a list of hard-won gotchas. Read that
> before running anything. This file covers only the repo's shape and where bytes live.

---

## Storage layout — which filesystem to use

`~/orcd/.quota` (refreshed ~every 30 min) lists only the three **personal** allocations. There is a
fourth, far larger **group** allocation it does not mention.

| Mount | Convenience path | Size | Use for |
|---|---|---|---|
| `/home/pablomer` | — | 200 GB | code, small outputs. Backed up — no bulk binaries |
| `/orcd/pool/007/pablomer` | `~/orcd/pool` | 1 TB (~81% used) | this repo, active run dirs, `checkpoints_new/` |
| `/orcd/scratch/orcd/009/pablomer` | `~/orcd/scratch` | 1 TB — **HARD FULL** | downstream image stores (read-only in practice) |
| `/orcd/data/mki_aryeh/001/pablomer` | `~/aryeh_data` | **200 TB (~1% used)** | bulk checkpoint archives, anything large |

Three things that will bite you:

1. **Scratch is 100% full and writes fail** (`touch` → `Disk quota exceeded`). It holds ~855 GB of
   irreplaceable downstream cutouts that the eval pipeline points at by default, plus the paper
   checkpoints. Worse: the repo's `galaxy_images/galaxy_model/checkpoints` is a **symlink into
   scratch**, and that is also the default for `RunConfig.shared_checkpoint_dir` — so any config that
   doesn't set an absolute `shared_checkpoint_dir` will try to write a 2 GB checkpoint into a full
   filesystem and die. Always set it explicitly.
2. **Use `/orcd/data/mki_aryeh/001` for anything big.** It is the `mki_aryeh` group allocation (same
   group as the `pg_mki_aryeh` GPU partition), setgid so files inherit group access, verified mounted
   and writable on both `node4900` (GPU) and `node4701` (CPU), and it does **not** count against the
   personal POOL quota. Convention: one directory per user at the top level.
3. **Cross-filesystem `mv` is a copy.** Moving a few GB from POOL to `/orcd/data` can outlast a
   command timeout and leave a **truncated** destination file while the source is still intact. Do
   copy → verify sizes → delete source.

### The symlink pattern

Bulk artifacts live on the group allocation, but the *canonical path stays inside the repo/POOL* via
symlinks, so code never has to know:

```
checkpoints_new/scale_ablation/snapshots          ->  /orcd/data/mki_aryeh/001/pablomer/…/snapshots
galaxy_images/galaxy_model/scale_ablation/snapshots ->  (same target)
```

The second symlink exists purely so the archive appears in the editor tree where you actually work —
VSCode only displays paths under the workspace root, so a symlink in `$HOME` is invisible when the
repo is the workspace. Both are `.gitignore`d.

`.vscode/settings.json` (tracked) excludes those paths from the file watcher and from search. Without
it VSCode churns the extension host against a network mount and every project-wide search crawls
2.2 GB `.ckpt` binaries.

---

## Layout

```
galaxy_images/galaxy_model/
├── train.py                      # unified Lightning entry point for every variant
├── variants.py                   # variant name -> model class registry
├── config.py                     # ExperimentConfig dataclasses + JSON loader + --set overrides
├── data_factory.py               # builds train/val dataloaders (holdout, subsetting, splits)
├── double_train_fm_neighbors.py  # ConditionalFlowMatchingModule — the paper baseline
├── configs/                      # per-variant JSON configs
│
├── diffusion_ablation/           # DDPM-vs-flow-matching (reviewer SYJm W1)
├── scale_ablation/               # how many CROSS-MATCHES do you need (reviewer JcH2 Q4)
├── shuffle_ablation/             # counterfactual donor-swap diagnostics
├── contrastive_baseline/         # dual-encoder InfoNCE baseline
├── single_encoder_ablation/      # no-physics-invariance control
│
├── downstream_evaluation/engaging/   # frozen-embedding MLP probes, n=5469 crossmatch
└── generative_evaluation/            # calibration, PSD, reconstruction sweeps
```

Each ablation directory is self-documenting: `PLAN.md` (design + locked decisions), `PROGRESS.md`
(running log), and for `scale_ablation` also `IMPLEMENTATION_REVIEW.md` (a verification walkthrough
with runnable check commands).

---

## Quick start

```bash
conda activate torchenv

# smallest useful thing: 200-step GPU smoke of one ablation arm
sbatch --export=ALL,CONFIG_PATH=$PWD/galaxy_images/galaxy_model/scale_ablation/configs/scale_1000.json \
       galaxy_images/galaxy_model/scale_ablation/smoke_train.slurm

# a real training run (4xH200, ~7.7h for 75k steps)
sbatch --job-name=scale-1000 \
       --export=ALL,CONFIG_PATH=$PWD/galaxy_images/galaxy_model/scale_ablation/configs/scale_1000.json \
       galaxy_images/galaxy_model/scale_ablation/train_scale.slurm

# downstream R2 for one checkpoint (~11 min, 1 GPU)
ONLY=<variant> bash galaxy_images/galaxy_model/downstream_evaluation/engaging/run_per_variant.sh
```

Overrides go through `--set section.key=value` (repeatable), applied after the JSON is merged onto the
dataclass defaults.

---

## Two behaviours worth knowing before you trust a number

- **The LR schedule does not anneal.** Every step-based run in this repo builds
  `CosineAnnealingLR(T_max=trainer.max_epochs)` while `train.py` passes only `max_steps` — so
  `max_epochs == -1` and the LR **alternates between `lr` and exactly 0.0 every epoch**. Roughly half
  of all optimizer steps apply no update. `model.lr_schedule` now exposes
  `paper` (default, reproduces published runs) / `constant` / `cosine` / `linear`. Full explanation and
  the affected file list are in [`CLAUDE.md`](CLAUDE.md).
- **Anchor survey is assigned by index parity**, `idx % 2 == 0` → HSC — not a random 50/50 draw. Any
  subsampling of anchors must be parity-stratified or it skews the survey mix, and per-survey
  reconstruction MSE differs ~3.4×.

## Reviews and rebuttal material

`reviews/` holds the **official** peer reviews and Pablo's official responses and is **read-only**.
All drafts and suggested wording go in `reviews/claude-suggestions/`.
