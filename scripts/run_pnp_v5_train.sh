#!/bin/bash
#SBATCH --job-name=v5-train
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=runs/pnp_v5/train_%j.log
#SBATCH --exclude=crane0,crane1,crane2

set -eo pipefail
cd ~/GAM_train_ready

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gam
export PYTHONPATH=gam_release:$PYTHONPATH

mkdir -p runs/pnp_v5

echo "=== $(date) === Training on $(hostname) with $SLURM_GPUS_ON_NODE GPUs ==="
echo "Config: 102 eps, 3 slots, global batch 256 (128×2), LR 2e-4, 12000 steps"

torchrun --nproc_per_node=2 train/train.py --config configs/pnp_v5.yaml

echo ""
echo "=== $(date) === Training DONE ==="
