#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_cluster_env.sh"

# Export KEY=VALUE params passed after the script name (from grid_run).
for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

if [[ -z "${REPAIR_TASK_IDS:-}" ]]; then
  echo "ERROR: REPAIR_TASK_IDS must be set to a comma-separated list of original task ids."
  exit 1
fi

if [[ -z "${SGE_TASK_ID:-}" ]]; then
  echo "ERROR: SGE_TASK_ID environment variable not found."
  exit 1
fi

IFS=',' read -r -a repair_task_ids <<< "${REPAIR_TASK_IDS}"
repair_index=$((SGE_TASK_ID - 1))

if (( repair_index < 0 || repair_index >= ${#repair_task_ids[@]} )); then
  echo "ERROR: Repair array task ${SGE_TASK_ID} is outside REPAIR_TASK_IDS length ${#repair_task_ids[@]}."
  exit 1
fi

original_task_id="${repair_task_ids[$repair_index]}"
export SGE_TASK_ID="${original_task_id}"

echo "===== EZClimate Tree Robustness Repair Task ====="
echo "Host: ${HOSTNAME}"
echo "Job: ${JOB_ID:-?}  Repair index: $((repair_index + 1))  Original task: ${SGE_TASK_ID}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER:-unset}"
echo "REPAIR_TASK_IDS=${REPAIR_TASK_IDS}"
echo "==============================================="

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=FALSE
export MKL_DYNAMIC=FALSE

python -u scripts/main_tree_robustness_cluster.py
