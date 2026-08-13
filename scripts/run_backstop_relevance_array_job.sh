#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_cluster_env.sh"

for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

echo "===== EZClimate Backstop Relevance Task ====="
echo "Host: ${HOSTNAME}"
echo "Job: ${JOB_ID:-?}  Task: ${SGE_TASK_ID:-?}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER:-paper-backstop-relevance-v1}"
echo "BACKSTOP_RELEVANCE_CASES=${BACKSTOP_RELEVANCE_CASES:-default}"
echo "BACKSTOP_RELEVANCE_BACKSTOP_CAP=${BACKSTOP_RELEVANCE_BACKSTOP_CAP:-1.5}"
echo "==============================================="

export LBFGSB_N_WORKERS="${LBFGSB_N_WORKERS:-${NSLOTS:-1}}"
export N_CANDIDATES="${N_CANDIDATES:-128}"
export N_LOCAL_STARTS="${N_LOCAL_STARTS:-4}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-128}"
export MAX_LOCAL_STARTS="${MAX_LOCAL_STARTS:-4}"
export ESCALATE_ON_DISPERSION="${ESCALATE_ON_DISPERSION:-1}"
export LBFGSB_GRADIENT_PROGRESS_EVERY="${LBFGSB_GRADIENT_PROGRESS_EVERY:-5}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=FALSE
export MKL_DYNAMIC=FALSE
export DISABLE_OPTIMAL_CACHE=1
export ADJOINT_FAIL_ON_VALUE_MISMATCH="${ADJOINT_FAIL_ON_VALUE_MISMATCH:-0}"
export ADJOINT_VALUE_PARITY_MODE="${ADJOINT_VALUE_PARITY_MODE:-off}"

python -u main_backstop_relevance_cluster.py
