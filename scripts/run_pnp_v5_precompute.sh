#!/bin/bash
#SBATCH --job-name=v5-precomp
#SBATCH --nodelist=crane5,crane6,crane7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/pnp_v5/precompute_%j.log

set -eo pipefail
cd ~/GAM_train_ready

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gam
export PYTHONPATH=gam_release:$PYTHONPATH

mkdir -p runs/pnp_v5 data/pnp_v5_cache

echo "=== $(date) === Precompute perception cache (102 episodes) ==="
python precompute/precompute.py --config configs/pnp_v5.yaml

echo ""
echo "=== $(date) === DONE ==="
