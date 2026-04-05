#!/bin/bash
#SBATCH --job-name=discord-test
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:05:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -e

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
IMG="/tmp/silly_plot_${SLURM_JOB_ID}.png"

python - <<EOF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 5), facecolor="#1a1a2e")
ax.set_facecolor("#16213e")

theta = np.linspace(0, 4 * np.pi, 1000)
for i, (color, phase) in enumerate(zip(["#e94560", "#0f3460", "#f5a623", "#7ec8e3"], [0, 0.5, 1.0, 1.5])):
    ax.plot(theta, np.sin(theta + phase) * (1 + 0.3 * i), color=color, lw=2, alpha=0.9)

ax.text(0.5, 0.92, "your galaxy model is cooked fr fr", transform=ax.transAxes,
        ha="center", fontsize=13, color="white", fontweight="bold")
ax.text(0.5, 0.04, "SLURM job notification test", transform=ax.transAxes,
        ha="center", fontsize=8, color="#aaaaaa", style="italic")

ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()
fig.savefig("${IMG}", dpi=120, bbox_inches="tight")
print("Figure saved: ${IMG}")
EOF

echo "Sending to Discord..."
curl -v \
    -F "content=🔭 **Job \`${SLURM_JOB_ID}\` done** — SLURM notification test from the cluster" \
    -F "file=@${IMG}" \
    "${DISCORD_WEBHOOK}"

rm -f "${IMG}"
echo "Discord notified."
