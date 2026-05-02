#!/bin/bash
#SBATCH --job-name=sentinel-job1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/sentinel_job1_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/sentinel_job1_%j.err
#SBATCH --open-mode=append

set -eo pipefail
TARGET=303624
NOTIFY=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal/discord_notify.py

source ~/.bashrc
conda activate torchenv

STATE=$(sacct -j $TARGET -X -P -n -o State 2>/dev/null | head -1 | tr -d ' ')
EXIT=$(sacct -j $TARGET -X -P -n -o ExitCode 2>/dev/null | head -1 | tr -d ' ')
ELAPSED=$(sacct -j $TARGET -X -P -n -o Elapsed 2>/dev/null | head -1 | tr -d ' ')

if [ "$STATE" = "COMPLETED" ]; then
    python3 "$NOTIFY" "✅ \`hsc-ivar-maps\` (job ${TARGET}) finished OK: state=${STATE}, exit=${EXIT}, elapsed=${ELAPSED}"
else
    python3 "$NOTIFY" "❌ \`hsc-ivar-maps\` (job ${TARGET}) ended: state=${STATE}, exit=${EXIT}, elapsed=${ELAPSED} — check logs/hsc_ivar_maps_${TARGET}.{out,err}"
fi
