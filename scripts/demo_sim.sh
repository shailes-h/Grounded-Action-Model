#!/bin/bash
# End-to-end sim demo: precompute -> visual checks -> train -> offline eval.
#
#   bash scripts/demo_sim.sh configs/demo_sim.yaml [N_EPISODES] [TRAIN_STEPS]
#
# Needs: the gam package installed (pip install -e <gam_release>), a WildDet3D
# checkpoint and sim episodes wired into the config (see the Demo section of the
# README). Every stage prints a *_DONE marker; inspect the visual artifacts between
# stages before trusting the next one.
set -e
CFG=${1:?usage: demo_sim.sh <config.yaml> [n_episodes] [train_steps]}
N=${2:-3}
STEPS=${3:-800}
PY=${PYTHON:-python}
DIR=$(cd "$(dirname "$0")/.." && pwd)
OUT=$($PY - "$CFG" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["train"]["out_dir"])
EOF
)
echo "== [1/4] precompute (first $N episodes) =="
$PY "$DIR/precompute/precompute.py" --config "$CFG" --ep-start 0 --ep-end "$N"

echo "== [2/4] visual checks — LOOK AT THESE before training =="
EP=$($PY - "$CFG" <<'EOF'
import sys, yaml, glob, os
cfg = yaml.safe_load(open(sys.argv[1]))
p = sorted(glob.glob(os.path.join(cfg["train"]["cache_dir"], "*.npz")))[0]
print(os.path.splitext(os.path.basename(p))[0])
EOF
)
$PY "$DIR/precompute/viz_cache.py" --config "$CFG" --episode "$EP" --out "$OUT/viz"
$PY "$DIR/precompute/verify.py" --config "$CFG" --episode "$EP" --out "$OUT/viz/$EP.mp4"
echo ">> images: $OUT/viz/${EP}_f*.png   video: $OUT/viz/$EP.mp4"

echo "== [3/4] train ($STEPS steps) =="
$PY "$DIR/train/train.py" --config "$CFG" --max-steps "$STEPS"

echo "== [4/4] offline eval =="
$PY "$DIR/eval/eval_offline.py" --config "$CFG" --ckpt "$OUT/latest.pt"
echo ">> trajectory plots: $OUT/eval_offline/"
echo "DEMO_DONE"
