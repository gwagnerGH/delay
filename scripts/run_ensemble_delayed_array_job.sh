#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_cluster_env.sh"

# Export KEY=VALUE params passed after the script name (from grid_run)
for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

echo "===== EZClimate Ensemble Delayed Action Task ====="
echo "Host: ${HOSTNAME}"
echo "Job: ${JOB_ID:-?}  Task: ${SGE_TASK_ID:-?}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER:-unset}"
echo "BASELINE_NUM=${BASELINE_NUM:-unset}"
echo "=============================================="

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=FALSE
export MKL_DYNAMIC=FALSE

python -u scripts/main_ensemble_delayed_cluster.py
