#!/bin/bash
#SBATCH --job-name=wd3d-play
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/wilddet3d_playground_%j.log
#SBATCH --exclude=crane0,crane1,crane2

set -eo pipefail
cd ~/GAM_train_ready
mkdir -p runs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gam
export PYTHONPATH=gam_release:$PYTHONPATH

echo "=== $(date) === Starting WildDet3D prompt playground on $(hostname) ==="
echo "Once it prints 'open http://...', tunnel it from your laptop with:"
echo "    ssh -L 8600:$(hostname):8600 storage"
echo "then browse http://localhost:8600"

python scripts/wilddet3d_prompt_playground.py \
    --dataset-dir datasets_GAM/PnP_version5 --episode 0 --frame 0 \
    --checkpoint gam_release/weights/wilddet3d_v1.1/wilddet3d_v1.1_droid_finetuned.ckpt \
    --device cuda:0 --port 8600
