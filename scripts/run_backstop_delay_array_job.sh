#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_cluster_env.sh"

for kv in "$@"; do
  case "$kv" in *=*) export "$kv" ;; esac
done

echo "EZDelay revised backstop sensitivity: task ${SGE_TASK_ID:-${TASK_ID:-?}}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER:-revised-backstop-delay-RA10-EIS22-curvature-v3}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -u -m scripts.main_backstop_delay_cluster
