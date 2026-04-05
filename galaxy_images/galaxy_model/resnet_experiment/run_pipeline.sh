#!/bin/bash
# ==============================================================================
# Launch the full ResNet ellipticity pipeline as a SLURM dependency chain.
#
#   Step 1: prepare data  (CPU, ~1-2h)  →  resnet_data.h5
#   Step 2: train ResNet  (GPU, ~4-6h)  →  resnet_best.pth + training plots
#   Step 3: evaluate gen  (GPU, ~2-4h)  →  comparison plots
#
# Each step only starts after the previous one succeeds (afterok).
# Each step sends its figures to Discord when done.
#
# Usage:
#   bash run_pipeline.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

echo "=== ResNet Ellipticity Pipeline ==="
echo "Submitting 3 jobs with dependency chain..."
echo ""

# Step 1: Prepare data (no GPU)
JOB1=$(sbatch --parsable "${SCRIPT_DIR}/run_prepare_data.slurm")
echo "Step 1 — prepare data:     job ${JOB1}"

# Step 2: Train ResNet (waits for step 1)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} "${SCRIPT_DIR}/run_train.slurm")
echo "Step 2 — train ResNet:     job ${JOB2}  (starts after ${JOB1})"

# Step 3: Evaluate on generated images (waits for step 2)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} "${SCRIPT_DIR}/run_evaluate_generated.slurm")
echo "Step 3 — evaluate gen:     job ${JOB3}  (starts after ${JOB2})"

echo ""
echo "All 3 jobs queued. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${JOB1},${JOB2},${JOB3} --format=JobID,JobName,State,Elapsed,ExitCode"
