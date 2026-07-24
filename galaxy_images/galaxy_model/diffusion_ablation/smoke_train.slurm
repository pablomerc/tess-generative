#!/bin/bash
#SBATCH --job-name=diff-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# 200-step GPU smoke (Phase 5). Parameterize with:
#   sbatch --export=ALL,CONFIG_PATH=.../neighbors_ddpm_eps.json diffusion_ablation/smoke_train.slurm
#   sbatch --export=ALL,CONFIG_PATH=.../neighbors_fm_control.json diffusion_ablation/smoke_train.slurm

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
ABL_DIR="${REPO}/galaxy_images/galaxy_model/diffusion_ablation"
mkdir -p "${ABL_DIR}/logs"

: "${CONFIG_PATH:?set CONFIG_PATH}"
CONFIG_BASENAME="$(basename "${CONFIG_PATH}")"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

_ping "diff-smoke 🚀 start · \`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?}"
trap '_ping "diff-smoke $([ $? -eq 0 ] && echo ✅ || echo ❌) · \`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
# Offline (not disabled): the module's validation calls the W&B-specific
# self.logger.experiment.log(), which crashes under the CSV fallback logger.
export WANDB_MODE=offline

export SMOKE_OUT="/orcd/pool/007/pablomer/checkpoints_new/diffusion_ablation/smoke/${CONFIG_BASENAME%.json}"
mkdir -p "${SMOKE_OUT}"

srun python -u -m galaxy_images.galaxy_model.train \
  --config "${CONFIG_PATH}" \
  --set trainer.devices=1 \
  --set trainer.num_steps=200 \
  --set trainer.val_check_interval=100 \
  --set data.batch_size=16 \
  --set "wandb.name=smoke-${CONFIG_BASENAME%.json}" \
  --set trainer.monitor_metric=val/loss \
  --set model.num_integration_steps=10 \
  --set model.num_mse_images=8 \
  --set model.num_sample_images=2 \
  --set "run.output_dir=${SMOKE_OUT}" \
  --set run.shared_checkpoint_dir=null

# Post-smoke checks
export CONFIG_PATH
python -u - <<'PY'
import os, sys
from pathlib import Path
import torch

smoke_out = Path(os.environ["SMOKE_OUT"])
config_path = Path(os.environ["CONFIG_PATH"])
# NOTE: periodic ckpt callback keeps auto_insert_metric_name=True, so the on-disk name
# is 'latest-step=step=200.ckpt' (PL rewrites {step} -> step={step}). Glob tolerantly;
# save_top_k=1 guarantees a single match.
ckpts = sorted(smoke_out.rglob("latest-step*.ckpt"))
print(f"[smoke-check] found ckpts: {ckpts}")
if not ckpts:
    print("[smoke-check] ERROR: no latest-step*.ckpt found", file=sys.stderr)
    sys.exit(1)
ckpt = ckpts[-1]
is_ddpm = "ddpm" in config_path.name
if is_ddpm:
    from galaxy_images.galaxy_model.diffusion_ablation.double_train_ddpm_neighbors import (
        ConditionalDDPMModule,
    )
    model = ConditionalDDPMModule.load_from_checkpoint(str(ckpt), map_location="cpu")
    assert model.hparams.prediction_type == "epsilon"
    assert model.train_scheduler.config.clip_sample is False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    B, C, H, W = 2, model.in_channels, model.image_size, model.image_size
    samegal = torch.randn(B, C, H, W, device=device)
    sameins = torch.randn(B, 5, C, H, W, device=device)
    masks = torch.ones(B, 5, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model.sample(samegal, sameins, masks=masks, num_steps=10)
    assert out.shape == (B, C, H, W), out.shape
    outside = bool(((out < -1) | (out > 1)).any().item())
    print(f"[smoke-check] sample shape={tuple(out.shape)} values_outside_[-1,1]={outside}")
    if not outside:
        print("[smoke-check] WARNING: no values outside [-1,1] — clip_sample may be on")
    print(f"[smoke-check] hparams.prediction_type={model.hparams.prediction_type}")
else:
    from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
    model = ConditionalFlowMatchingModule.load_from_checkpoint(str(ckpt), map_location="cpu")
    print(f"[smoke-check] FM reload ok: {type(model).__name__}")
print("[smoke-check] PASS")
PY
