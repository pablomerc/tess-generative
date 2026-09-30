#!/bin/bash
#SBATCH --job-name=hsc_lsst_anom
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:30:00
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/hsc_lsst_anom-%j.log

# Anomaly stage (encode -> NSF/IForest/kNN/Mahalanobis -> galleries) for one checkpoint.
#   sbatch galaxy_images/galaxy_model/lsst/hsc_lsst_anomaly.sbatch --checkpoint <ckpt> --tag <name>
#   smoke: sbatch -p devel -t 25:00 ... --checkpoint <ckpt> --tag smoke --n-max 3000 --nsf-epochs 5
# Do NOT add --mem (submission filter rejects it).

export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH=/tmp/miopen_${USER}_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH" /tmp/mpl_${USER} /tmp/numba_${USER}
export MPLCONFIGDIR=/tmp/mpl_${USER}
export NUMBA_CACHE_DIR=/tmp/numba_${USER}

source /work1/jeroenaudenaert/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate torchenv
cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=.
echo "[launcher] node=$(hostname) job=$SLURM_JOB_ID args: $*"
python -m galaxy_images.galaxy_model.lsst.hsc_lsst_anomaly "$@"
rc=$?
if [ $rc -ne 0 ]; then
  # the module posts its own success message; on failure send the log tail so the error is visible on Discord
  LOG=/work1/jeroenaudenaert/pablomer/logs/hsc_lsst_anom-${SLURM_JOB_ID}.log
  python - "$LOG" "$rc" "$*" <<'PY'
import sys
from galaxy_images.galaxy_model.lsst.hsc_lsst_anomaly import discord_post
log, rc, args = sys.argv[1], sys.argv[2], sys.argv[3]
tail = open(log, errors="replace").read()[-1400:]
ok = discord_post(f"**HSC × LSST anomaly job FAILED** (exit {rc}) args: `{args}`\n```\n{tail}\n```")
print(f"[notify] failure post {'sent' if ok else 'FAILED'}", flush=True)
PY
fi
exit $rc
