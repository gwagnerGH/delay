#!/bin/bash
# Default: 13 post-2250 growth cases x three delays = SGE array 1-39.
# DELAY_YEARS=10 selects 13 tasks. --list-cases performs preflight only.
set -euo pipefail
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${runner_dir}/_cluster_env.sh"

python_args=()
for arg in "$@"; do
  case "$arg" in
    [A-Za-z_]*=*)
      if [[ "$arg" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        export "$arg"
      else
        python_args+=("$arg")
      fi ;;
    *) python_args+=("$arg") ;;
  esac
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Bash 3.2 treats an empty array as unset under nounset.
if [[ -n "${python_args[0]+present}" ]]; then
  exec python -u -m scripts.main_growth_sensitivity_cluster "${python_args[@]}"
else
  exec python -u -m scripts.main_growth_sensitivity_cluster
fi
