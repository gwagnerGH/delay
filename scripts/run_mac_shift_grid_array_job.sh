#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_cluster_env.sh"

for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

export LBFGSB_POLICY_UPPER_BOUND=1.0
export BACKSTOP_SMOOTHING_WIDTH=0
export DISABLE_OPTIMAL_CACHE=1
export LBFGSB_N_WORKERS="${LBFGSB_N_WORKERS:-${NSLOTS:-1}}"
export N_CANDIDATES="${N_CANDIDATES:-512}"
export N_LOCAL_STARTS="${N_LOCAL_STARTS:-8}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-512}"
export MAX_LOCAL_STARTS="${MAX_LOCAL_STARTS:-8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "===== EZDelay MAC-Shift Grid Task ====="
echo "Job: ${JOB_ID:-?} Task: ${SGE_TASK_ID:-?}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER:-paper-mac-shift-grid-v1}"
echo "Grid: 11 horizontal shifts x 11 vertical shifts x 3 delays = 363 tasks"
echo "m cap=${LBFGSB_POLICY_UPPER_BOUND}; smoothing=${BACKSTOP_SMOOTHING_WIDTH}"
python -u main_mac_shift_grid_cluster.py
