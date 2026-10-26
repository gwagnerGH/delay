#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_cluster_env.sh"

for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

export REVISED_BACKSTOP=s_curve
export REVISED_NUMERICAL_GUARD=5
export LBFGSB_POLICY_UPPER_BOUND=4
export EZDELAY_LEGACY_CLUSTER=0
export DISABLE_OPTIMAL_CACHE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "===== EZDelay MAC-Shift Grid Task ====="
echo "Job: ${JOB_ID:-?} Task: ${SGE_TASK_ID:-?}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER:-paper-revised-mac-shift-ra10-eis22-scurve-v2}"
echo "Grid: 11 horizontal shifts x 11 vertical shifts x 3 delays = 363 tasks"
echo "Revised smooth S-curve; economic m cap=${LBFGSB_POLICY_UPPER_BOUND}; numerical guard=${REVISED_NUMERICAL_GUARD}"
python -u -m scripts.main_mac_shift_grid_cluster
