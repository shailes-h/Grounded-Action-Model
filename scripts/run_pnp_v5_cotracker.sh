#!/bin/bash
#SBATCH --job-name=v5-cotrack
#SBATCH --nodelist=crane5,crane6,crane7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=runs/pnp_v5/cotracker_%j.log

set -eo pipefail
cd ~/GAM_train_ready

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gam
export PYTHONPATH=gam_release:$PYTHONPATH

mkdir -p runs/pnp_v5

echo "=== $(date) === CoTracker clicks (102 episodes, 3 slots) ==="
# seeds come from the hand clicks: run on a copy with clicks.source: annotations
sed 's/^\(    source:\) dataset /\1 annotations /' configs/pnp_v5.yaml > runs/pnp_v5/cotracker_cfg.yaml
python scripts/precompute_cotracker_clicks.py --config runs/pnp_v5/cotracker_cfg.yaml

echo ""
echo "=== $(date) === DONE ==="
