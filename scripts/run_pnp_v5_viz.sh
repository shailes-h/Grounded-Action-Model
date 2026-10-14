#!/bin/bash
#SBATCH --job-name=v5-viz
#SBATCH --nodelist=crane5,crane6,crane7
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=runs/pnp_v5/viz_%j.log
#SBATCH --dependency=singleton

set -eo pipefail
cd ~/GAM_train_ready
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gam
export PYTHONPATH=gam_release:$PYTHONPATH

mkdir -p runs/pnp_v5/viz

# Wait for precompute cache to have at least ep0
while [ ! -f data/pnp_v5_cache/episode_0000000.npz ]; do sleep 10; done

echo "=== $(date) === Generating visualizations ==="

# 1. Verify video for a few sample episodes (clicks, 2D/3D boxes, patch mask overlay)
for ep in episode_0000000 episode_0000025 episode_0000052 episode_0000077 episode_0000101; do
    if [ -f "data/pnp_v5_cache/${ep}.npz" ]; then
        echo "--- verify: $ep ---"
        python precompute/verify.py --config configs/pnp_v5.yaml --episode "$ep" \
            --out "runs/pnp_v5/viz/${ep}_verify.mp4" || echo "WARN: verify $ep failed"
    fi
done

# 2. viz_cache for sample episodes
for ep in episode_0000000 episode_0000025 episode_0000052 episode_0000077; do
    if [ -f "data/pnp_v5_cache/${ep}.npz" ]; then
        echo "--- viz_cache: $ep ---"
        python precompute/viz_cache.py --config configs/pnp_v5.yaml --episode "$ep" \
            --out "runs/pnp_v5/viz" || echo "WARN: viz_cache $ep failed"
    fi
done

# 3. PCD alignment check (first frame of a few episodes)
for ep in episode_0000000 episode_0000025 episode_0000052 episode_0000077; do
    echo "--- PCD alignment: $ep ---"
    python scripts/inspect_pcd_alignment.py --config configs/pnp_v5.yaml --episode "$ep" \
        --out "runs/pnp_v5/viz/pcd_${ep}.png" || echo "WARN: PCD $ep failed"
done


echo ""
echo "=== $(date) === Viz DONE ==="
echo "Output in runs/pnp_v5/viz/"
ls -la runs/pnp_v5/viz/
