#!/bin/bash
#SBATCH --job-name=v5-server
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/pnp_v5/server_%j.log
#SBATCH --exclude=crane0,crane1,crane2

set -eo pipefail
cd ~/GAM_train_ready

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gam
export PYTHONPATH=gam_release:$PYTHONPATH

echo "=== $(date) === Starting GAM v5 server on $(hostname) ==="
echo "Port 8204 — connect via http://$(hostname):8204/act"

python scripts/host_gam_server.py \
    --checkpoint runs/pnp_v5/latest.pt \
    --config configs/pnp_v5.yaml \
    --device cuda:0 --port 8204 \
    --track-clicks
