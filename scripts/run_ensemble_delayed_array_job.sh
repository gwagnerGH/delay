#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_cluster_env.sh"

# Export KEY=VALUE params passed after the script name (from grid_run)
for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

echo "===== EZClimate Ensemble Delayed Action Task ====="
echo "Host: ${HOSTNAME}"
echo "Job: ${JOB_ID:-?}  Task: ${SGE_TASK_ID:-?}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER:-unset}"
echo "BASELINE_NUM=${BASELINE_NUM:-unset}"
echo "N_SAMPLES=${N_SAMPLES:-10000}"
echo "GAUSSIAN_EIS_UPPER_BOUND=${GAUSSIAN_EIS_UPPER_BOUND:-untruncated}"
echo "GAUSSIAN_SAMPLE_SEED=${GAUSSIAN_SAMPLE_SEED:-${RANDOM_SEED_BASE:-20250706}}"
echo "=============================================="

if [[ "${OPTIMIZER:-}" == "lbfgsb_multistart" || "${OPTIMIZER:-}" == "adjoint_lbfgsb" || "${OPTIMIZER:-}" == "ga_adjoint_lbfgsb" || "${OPTIMIZER:-}" == "coarse_to_fine_adjoint_lbfgsb" ]]; then
  export LBFGSB_N_WORKERS="${LBFGSB_N_WORKERS:-${NSLOTS:-${SLURM_CPUS_PER_TASK:-1}}}"
  export N_CANDIDATES="${N_CANDIDATES:-128}"
  export N_LOCAL_STARTS="${N_LOCAL_STARTS:-4}"
  export MAX_CANDIDATES="${MAX_CANDIDATES:-128}"
  export MAX_LOCAL_STARTS="${MAX_LOCAL_STARTS:-4}"
  export ESCALATE_ON_DISPERSION="${ESCALATE_ON_DISPERSION:-0}"
  export LBFGSB_GRADIENT_PROGRESS_EVERY="${LBFGSB_GRADIENT_PROGRESS_EVERY:-5}"
  THREAD_DEFAULT=1
else
  THREAD_DEFAULT=4
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREAD_DEFAULT}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREAD_DEFAULT}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREAD_DEFAULT}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${THREAD_DEFAULT}}"
export DISABLE_OPTIMAL_CACHE="${DISABLE_OPTIMAL_CACHE:-1}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=FALSE
export MKL_DYNAMIC=FALSE

export ADJOINT_FAIL_ON_VALUE_MISMATCH="${ADJOINT_FAIL_ON_VALUE_MISMATCH:-0}"
export ADJOINT_VALUE_PARITY_MODE="${ADJOINT_VALUE_PARITY_MODE:-off}"

python -u main_ensemble_delayed_cluster.py
