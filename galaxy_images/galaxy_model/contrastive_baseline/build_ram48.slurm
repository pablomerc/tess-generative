#!/bin/bash
#SBATCH --job-name=contr-build48
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/%x_%j.out
#SBATCH --open-mode=append
set -eo pipefail
WEBHOOK="https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j"
_ping(){ curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "$WEBHOOK" >/dev/null 2>&1 || true; }
_on_exit(){ rc=$?; [[ $rc -ne 0 ]] && _ping "contr-build48 ❌ failed (rc=$rc) job ${SLURM_JOB_ID:-?}"; }
trap _on_exit EXIT
source ~/.bashrc; conda activate torchenv
export PYTHONPATH="/home/pablomer/orcd/pool/tess-generative:${PYTHONPATH:-}"
cd /home/pablomer/orcd/pool/tess-generative
_ping "contr-build48 🏗️ building compact 48x48 in-RAM store (16GB, 468k rows) to fix the mmap I/O bottleneck. job ${SLURM_JOB_ID:-?}"
python /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/build_ram48.py
_ping "contr-build48 ✅ 48x48 store built. Relaunching the 4 contrastive runs from RAM next."
echo BUILD_SLURM_DONE
