#!/bin/bash
# ==============================================================================
# End-to-end posterior calibration pipeline.
#
# Usage:
#   sbatch run_calibration.slurm /path/to/checkpoint.ckpt
#   sbatch run_calibration.slurm /path/to/checkpoint.ckpt 256 20 250
#
# Positional arguments (after sbatch flags):
#   $1  CHECKPOINT   (required) path to a Lightning .ckpt file
#   $2  N_GALAXIES   (optional, default: 512)
#   $3  M_SAMPLES    (optional, default: 20)
#   $4  STEPS        (optional, default: 250)
# ==============================================================================
#SBATCH --job-name=calibration
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -e

# ------------------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------------------
CHECKPOINT="${1:?Error: checkpoint path required as first argument}"
N_GALAXIES="${2:-512}"
M_SAMPLES="${3:-20}"
STEPS="${4:-250}"

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
REPO_ROOT="/data/vision/billf/scratch/pablomer/projects/tess-generative"
EXPERIMENT_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/xi_squared_experiment"
GALAXY_MODEL_DIR="${REPO_ROOT}/galaxy_images/galaxy_model"
SHARD="/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5"

RUN_TAG="${N_GALAXIES}g_${STEPS}steps_job${SLURM_JOB_ID}"
OUTPUT_DIR="${EXPERIMENT_DIR}/outputs_${RUN_TAG}"
OUTPUT_SAMPLES="${OUTPUT_DIR}/samples.h5"
OUTPUT_NPZ="${OUTPUT_DIR}/calibration_data.npz"
OUTPUT_PLOT="${OUTPUT_DIR}/calibration_combined.png"

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

cd "${GALAXY_MODEL_DIR}"
mkdir -p "${OUTPUT_DIR}" "${EXPERIMENT_DIR}/logs"

echo "================================================================"
echo "Calibration pipeline — job ${SLURM_JOB_ID}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Shard      : ${SHARD}"
echo "  Galaxies   : ${N_GALAXIES}"
echo "  Samples    : ${M_SAMPLES}"
echo "  Steps      : ${STEPS}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "================================================================"

# ------------------------------------------------------------------------------
# Step 1: Generate posterior samples
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1/3] Generating samples (${N_GALAXIES} galaxies × ${M_SAMPLES} samples × ${STEPS} steps)"
echo "================================================================"
python "${EXPERIMENT_DIR}/generate_samples.py" \
    --checkpoint "${CHECKPOINT}" \
    --shard      "${SHARD}" \
    --n-galaxies "${N_GALAXIES}" \
    --m-samples  "${M_SAMPLES}" \
    --steps      "${STEPS}" \
    --output     "${OUTPUT_SAMPLES}"

# ------------------------------------------------------------------------------
# Step 2: Extract z-scores to lightweight npz
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/3] Extracting calibration z-scores"
echo "================================================================"
python "${EXPERIMENT_DIR}/extract_calibration_data.py" \
    --samples "${OUTPUT_SAMPLES}" \
    --output  "${OUTPUT_NPZ}"

# ------------------------------------------------------------------------------
# Step 3: Make combined calibration plot
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3/3] Making combined calibration plot"
echo "================================================================"
python "${EXPERIMENT_DIR}/makecalibrationplot.py" \
    --data   "${OUTPUT_NPZ}" \
    --output "${OUTPUT_PLOT}"

echo ""
echo "================================================================"
echo "Done. Results in: ${OUTPUT_DIR}"
echo "  samples.h5                — raw posterior samples"
echo "  calibration_data.npz      — z-scores (for fast re-plotting)"
echo "  calibration_combined.png  — combined HSC + Legacy figure"
echo "================================================================"

# ------------------------------------------------------------------------------
# Notify Discord
# ------------------------------------------------------------------------------
CKPT_NAME=$(basename "${CHECKPOINT}")

curl -s -F "content=**Calibration done** — job \`${SLURM_JOB_ID}\`
**Checkpoint**: \`${CKPT_NAME}\`
**Config**: ${N_GALAXIES} galaxies × ${M_SAMPLES} samples × ${STEPS} steps" \
    "${DISCORD_WEBHOOK}"

if [ -f "${OUTPUT_PLOT}" ]; then
    curl -s -F "file=@${OUTPUT_PLOT}" "${DISCORD_WEBHOOK}"
fi
