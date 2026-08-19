#!/bin/bash
# usage: bash scripts/precompute.sh configs/my_task.yaml [ep_start ep_end]
# (wrap in your scheduler for sharding; finished episodes are skipped automatically)
set -e
python "$(dirname "$0")/../precompute/precompute.py" --config "$1" ${2:+--ep-start $2} ${3:+--ep-end $3}
