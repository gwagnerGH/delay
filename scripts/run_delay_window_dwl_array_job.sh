#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_cluster_env.sh"

for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

: "${OUTPUT_FOLDER:?Set OUTPUT_FOLDER to the paper output folder to postprocess}"
: "${DWL_SHARD_COUNT:?Set DWL_SHARD_COUNT to the array length}"
TASK_ID="${SGE_TASK_ID:-${TASK_ID:-}}"
: "${TASK_ID:?This wrapper must run as an array task}"

python -u scripts/postprocess_delay_window_dwl.py \
  --folder "${OUTPUT_FOLDER}" \
  --shard-index "${TASK_ID}" \
  --shard-count "${DWL_SHARD_COUNT}"
