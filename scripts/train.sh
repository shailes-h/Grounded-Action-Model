#!/bin/bash
# usage: bash scripts/train.sh configs/my_task.yaml
set -e
python "$(dirname "$0")/../train/train.py" --config "$1"
