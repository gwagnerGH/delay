#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_cluster_env.sh"

for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

# Direct grid submissions do not set SGE_TASK_ID. main_delayed_partial.py
# accepts TASK_ID, so preserve it as the explicit task selector in that case.
if [[ -z "${SGE_TASK_ID:-}" && -n "${TASK_ID:-}" ]]; then
  export TASK_ID
fi

echo "===== Partial Mitigation Analysis Task ====="
echo "Host: ${HOSTNAME}"
echo "Job: ${JOB_ID:-?}  Task: ${SGE_TASK_ID:-${TASK_ID:-?}}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER:-unset}"
echo "============================================="

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

python -u main_delayed_partial.py
