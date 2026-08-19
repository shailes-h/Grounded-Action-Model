#!/bin/bash
# usage: bash scripts/eval_offline.sh configs/my_task.yaml runs/yam_task/latest.pt
set -e
python "$(dirname "$0")/../eval/eval_offline.py" --config "$1" --ckpt "$2"
