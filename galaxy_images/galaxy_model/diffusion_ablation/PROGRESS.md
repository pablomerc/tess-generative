# Diffusion Ablation — Progress Log

## 2026-07-23 — Phases 1–4 implemented (pre-smoke)

- Created `diffusion_ablation/__init__.py` and `ConditionalDDPMModule` in
  `double_train_ddpm_neighbors.py` (`clip_sample=False`, `t/1000` timestep trick,
  DDIM `sample(..., eta=)`).
- Wired `ModelConfig` diffusion fields + `data.mode=ram48` validation in `config.py`.
- Registered `neighbors_all_attn_ddpm` in `variants.py`.
- Added `ram48` branch in `data_factory.build_neighbors_dataloaders` → `NeighborsRAM48Dataset`.
- Configs: `configs/neighbors_ddpm_eps.json`, `configs/neighbors_fm_control.json`.
- SLURM: `train_ddpm.slurm` (4×H200, Discord `_ping`), `smoke_train.slurm` (1-GPU 200-step).
- Eval: `eval_recon_mse.py` (matched noise, manifest artifact, η rows, paper context row).
- UMAP: new `regenerate_umap.py` (`--arm fm|ddpm`); left `regenerate_umap_base.py` untouched.
- Downstream registry stubs: `diffusion-base` + `fm-control` in `variants.json`
  (glob → `latest-step=75000.ckpt` once training finishes).
- RAM48 store verified present at `/orcd/pool/007/pablomer/efficient_neighs_48`.
- Pool quota ~77% used (~791/1024 GB); each arm ~2–3 GB — OK to proceed after smoke.

## 2026-07-23 evening — audit fixes + submission (Claude, review session)

- 4-agent guide audit + 3-agent implementation review ran; corrections consolidated at the top of
  `IMPLEMENTATION_GUIDE.md`. Fixes applied here:
  - `train_ddpm.slurm`: `--ntasks=1` → `--ntasks-per-node=4`, `--cpus-per-task=32` → `8`
    (PL 2.6.1 + srun does NOT self-spawn ranks; old header silently trained on 1 GPU with 4-way
    data sharding). Real-job pass criterion: log shows "Starting with 4 processes".
  - `smoke_train.slurm`: added `--set trainer.monitor_metric=val/loss` (val/mse never logged
    without a logger → ModelCheckpoint crash) and ckpt check now globs `latest-step*.ckpt`
    (real filenames are `latest-step=step=<N>.ckpt`, auto_insert_metric_name).
  - `downstream_evaluation/engaging/variants.json`: both new ckpt_globs → `latest-step*.ckpt`.
  - Both slurm scripts: CRLF → LF (sbatch rejects DOS line endings).
- CPU pre-submit sanity PASSED for both configs (config→variant→model→loss/backward→2-step
  sample; 183.5M params both arms; init losses ≈1.12 ε-MSE / ≈2.07 velocity-MSE as expected).
- SUBMITTED: ddpm-eps-75k=18669496 (RUNNING on node4900), fm-ctrl-75k=18669497 (PD Resources),
  smoke-ddpm=18669498, smoke-fm=18669499 (mit_normal_gpu). WandB online (netrc verified).

## 2026-07-23 ~23:55 — eval-script review fixes applied (Claude, monitoring session)

3-agent adversarial review (wf_11a970ba-df6) of the Phase-7 scripts; plumbing/config/module
parity verified clean. Fixes applied (all py_compile-checked; training untouched):

- `eval_recon_mse.py`: (MAJOR) `--paper-checkpoint ""` crashed post-eval pre-write
  (`Path("")`→`'.'` is truthy+exists) — string-guard + try/except around the context row;
  (MAJOR) CSV now flushed incrementally after every model/η block instead of end-only;
  (minor) manifest now records+compares `noise_seed`/`batch_size` (legacy manifests upgraded
  in place); loud WARNINGs replace silent row omission.
- `regenerate_umap.py`: arm/checkpoint hparams cross-check (FM ckpt loaded silently under
  the DDPM class → wrong-arm figure); `--arm` folded into output stem (concurrent-arm tag
  collisions impossible); ram48 backend now memmaps (`load_into_ram=False`) instead of
  pulling 17 GB into RAM for ~2000 anchors.
- `double_train_ddpm_neighbors.py`: raise on `eta>0` without seeded generator; raise on
  `lambda_geometric>0` (Sinkhorn branch is intentionally absent).

Training at fix time: DDPM 44.8k/75k, FM 45.4k/75k, both ~3 steps/s, ETA ~02:45.

### Next

1. Monitor armed (both-ckpts-ready / job-death); heartbeat every 30 min.
2. Phase 7 once `latest-step=step=75000.ckpt` exists for both arms: downstream R², recon MSE
   (η=0 headline + η=1 robustness, --skip-paper NOT needed anymore), UMAPs via
   `regenerate_umap.py` (distinct tags per arm) → `RESULTS.md` + Discord headline numbers.

## 2026-07-23 late — mid-training preview evals (Claude)

- Reviewed `eval_recon_mse.py` (sound; matched noise, manifest enforcement, per-survey MSE).
- New `eval_recon_mse.slurm` (runs eval + posts the numbers table to Discord) and
  `summary_r2_ping.slurm` (CPU job, afterany on the R² jobs, posts base-vs-fm-vs-ddpm R² lines).
- Preview registry entries `diffusion-base-28k` / `fm-control-29k` → the `_copy` best ckpts
  (28k / 29k steps). Pinned 75k `diffusion-base` / `fm-control` entries untouched.
- Jobs: recon-mse=18685421 (η∈{0,1}, n=256, 250 steps → Discord table);
  R²: 18685533 (ddpm, running) + 18685535 (fm, pending on the 2-GPU mit_normal_gpu QOS cap);
  r2-ping=18685639 (dependency-held). Training arms 18669496/97 at ~3h10m, ETA ~02:45–03:00.
- Morning TODO: rerun recon-MSE + R² on the final 75k ckpts (pinned entries), UMAPs, RESULTS.md.
- NOTE: preview R² CSVs are named `predict_<v>-28k.csv`-style (default aion arch); rebuttal
  numbers come from the 75k finals, previews are directional only.
