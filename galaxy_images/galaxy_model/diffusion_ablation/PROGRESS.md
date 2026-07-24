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

### Next

1. Watch smokes (~30 min) — if either fails, scancel 18669496/18669497, fix, resubmit.
2. Verify "Starting with 4 processes" + sane loss curves in real-job logs/W&B.
3. Phase 7 once `latest-step=step=75000.ckpt` exists for both arms: downstream R², recon MSE
   (η=0 headline + η=1 robustness), UMAPs via `regenerate_umap.py` → `RESULTS.md`.
