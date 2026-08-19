#!/bin/bash
# usage: bash scripts/download_weights.sh weights/
set -e
D=${1:-weights}; mkdir -p "$D"
# CoTracker3 online (optional, point tracking)
[ -f "$D/cotracker_scaled_online.pth" ] || wget -O "$D/cotracker_scaled_online.pth" https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
echo "GAM policy (gam_policy.pt) and WildDet3D (wilddet3d.pt) checkpoints: place them in $D/ (see README)."
