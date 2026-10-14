#!/bin/bash
#SBATCH --job-name=v5-pipeline
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=crane0,crane1,crane2
#SBATCH --time=24:00:00
#SBATCH --output=runs/pnp_v5/pipeline_%j.log

set -eo pipefail
cd ~/GAM_train_ready

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gam
export PYTHONPATH=gam_release:$PYTHONPATH

mkdir -p runs/pnp_v5 data/pnp_v5_cache runs/pnp_v5/viz

# ===== STEP 1: CoTracker =====
# CoTracker seeds from the hand clicks, so run it on a copy of the config with
# clicks.source: annotations (the tracked config itself consumes clicks/cotracker).
echo "=== $(date) === STEP 1: CoTracker clicks (102 episodes, 3 slots) ==="
sed 's/^\(    source:\) dataset /\1 annotations /' configs/pnp_v5.yaml > runs/pnp_v5/cotracker_cfg.yaml
python scripts/precompute_cotracker_clicks.py --config runs/pnp_v5/cotracker_cfg.yaml
echo "=== $(date) === CoTracker DONE ==="

# ===== STEP 2: Precompute =====
echo ""
echo "=== $(date) === STEP 2: Precompute perception cache (102 episodes) ==="
python precompute/precompute.py --config configs/pnp_v5.yaml
echo "=== $(date) === Precompute DONE ==="

# ===== STEP 3: Visualizations =====
echo ""
echo "=== $(date) === STEP 3: Generating visualizations ==="

for ep in episode_0000000 episode_0000025 episode_0000052 episode_0000077 episode_0000101; do
    if [ -f "data/pnp_v5_cache/${ep}.npz" ]; then
        echo "--- verify: $ep ---"
        python precompute/verify.py --config configs/pnp_v5.yaml --episode "$ep" \
            --out "runs/pnp_v5/viz/${ep}_verify.mp4" || echo "WARN: verify $ep failed"
    fi
done

for ep in episode_0000000 episode_0000025 episode_0000052 episode_0000077; do
    if [ -f "data/pnp_v5_cache/${ep}.npz" ]; then
        echo "--- viz_cache: $ep ---"
        python precompute/viz_cache.py --config configs/pnp_v5.yaml --episode "$ep" \
            --out "runs/pnp_v5/viz" || echo "WARN: viz_cache $ep failed"
    fi
done

for ep in episode_0000000 episode_0000025 episode_0000052 episode_0000077; do
    echo "--- PCD alignment: $ep ---"
    python scripts/inspect_pcd_alignment.py --config configs/pnp_v5.yaml --episode "$ep" \
        --out "runs/pnp_v5/viz/pcd_${ep}.png" || echo "WARN: PCD $ep failed"
done


echo "=== $(date) === Viz DONE ==="
echo "Output in runs/pnp_v5/viz/"
ls -la runs/pnp_v5/viz/

echo ""
echo "========================================="
echo "  PIPELINE COMPLETE — ready for training"
echo "  Submit: sbatch scripts/run_pnp_v5_train.sh"
echo "========================================="
