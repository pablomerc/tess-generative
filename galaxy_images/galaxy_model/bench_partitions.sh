#!/bin/bash
# Submit a short throughput benchmark of the base efficient training pipeline
# to multiple partitions in parallel. Each job runs 500 steps, no wandb, no
# checkpoints, no validation — pure speed measurement.
#
# Usage:
#   bash galaxy_images/galaxy_model/bench_partitions.sh
#
# Then watch logs in /work1/jeroenaudenaert/pablomer/logs/bench_*_<jobid>.out
# and compare the it/s reported by the Lightning progress bar.

set -eo pipefail

REPO=/work1/jeroenaudenaert/pablomer/tess-generative
LOGDIR=/work1/jeroenaudenaert/pablomer/logs
mkdir -p "$LOGDIR"

# (partition, gpus_per_node, batch_size_per_gpu)
# batch=64/GPU keeps VRAM pressure roughly constant; tune if MI210 OOMs at 64.
PARTITIONS=(
  "mi2104x 4 64"
  "mi2508x 8 64"
  "mi3008x 8 64"
  "mi3258x 8 64"
  "mi3508x 8 64"
)

for entry in "${PARTITIONS[@]}"; do
  read -r PART GPUS BS <<< "$entry"
  JOBNAME="bench-${PART}"
  OUT="${LOGDIR}/bench_${PART}_%j.out"
  ERR="${LOGDIR}/bench_${PART}_%j.err"

  sbatch \
    --job-name="$JOBNAME" \
    --partition="$PART" \
    --nodes=1 \
    --cpus-per-task=16 \
    --time=00:30:00 \
    --output="$OUT" \
    --error="$ERR" \
    --open-mode=append \
    --wrap "
      set -eo pipefail
      source ~/.bashrc
      conda activate torchenv
      export TORCH_BLAS_PREFER_HIPBLASLT=0
      cd ${REPO}
      export PYTHONPATH=${REPO}:\$PYTHONPATH
      echo '=== bench: partition=${PART} gpus=${GPUS} batch_per_gpu=${BS} ==='
      nvidia-smi 2>/dev/null || rocm-smi 2>/dev/null || true
      srun python -u galaxy_images/galaxy_model/neighbors_efficient_train.py \\
        --set trainer.num_steps=500 \\
        --set trainer.devices=${GPUS} \\
        --set trainer.scale_steps_by_devices=false \\
        --set trainer.val_check_interval=10000 \\
        --set trainer.checkpoint_every_n_train_steps=10000 \\
        --set data.batch_size=${BS} \\
        --set data.num_workers=8 \\
        --set data.save_heldout_validation=false \\
        --set wandb.enabled=false
      echo '=== done at '\$(date)' ==='
    "
done

echo ""
echo "Submitted. Watch with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOGDIR}/bench_*.out"
