#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export EZClimate_TM_ROOT="${EZClimate_TM_ROOT:-${PROJECT_ROOT}}"

# Production-safe defaults for every cluster wrapper. Script arguments are
# parsed after this file is sourced, so an explicit submission value still
# takes precedence. In particular, ftol=0 prevents L-BFGS-B from declaring
# convergence on objective reduction before the strict stationarity audit.
export LBFGSB_MAXITER="${LBFGSB_MAXITER:-1000}"
export LBFGSB_FTOL="${LBFGSB_FTOL:-0}"
export LBFGSB_GTOL="${LBFGSB_GTOL:-1e-8}"
export LBFGSB_POLICY_UPPER_BOUND="${LBFGSB_POLICY_UPPER_BOUND:-1.5}"
export LBFGSB_KKT_CHECK="${LBFGSB_KKT_CHECK:-1}"
export LBFGSB_PROJECTED_GRADIENT_TOL="${LBFGSB_PROJECTED_GRADIENT_TOL:-2e-5}"
export LBFGSB_NONSMOOTH_KKT_CHECK="${LBFGSB_NONSMOOTH_KKT_CHECK:-1}"
export LBFGSB_NONSMOOTH_KKT_STEP="${LBFGSB_NONSMOOTH_KKT_STEP:-1e-4}"
export LBFGSB_NONSMOOTH_KKT_UTILITY_GAIN_TOL="${LBFGSB_NONSMOOTH_KKT_UTILITY_GAIN_TOL:-1e-8}"
export LBFGSB_REMOVAL_GAIN_TOL="${LBFGSB_REMOVAL_GAIN_TOL:-1e-8}"
export LBFGSB_NONSMOOTH_KKT_MAX_COORDINATES="${LBFGSB_NONSMOOTH_KKT_MAX_COORDINATES:-255}"
export LBFGSB_PERTURBATION_CHECK="${LBFGSB_PERTURBATION_CHECK:-1}"
export ADJOINT_VALIDATE_GRADIENT="${ADJOINT_VALIDATE_GRADIENT:-1}"
export ADJOINT_VALIDATION_DIRECTIONS="${ADJOINT_VALIDATION_DIRECTIONS:-4}"
export ADJOINT_VALIDATION_ABS_TOL="${ADJOINT_VALIDATION_ABS_TOL:-1e-5}"
export ADJOINT_VALIDATION_REL_TOL="${ADJOINT_VALIDATION_REL_TOL:-1e-3}"
export ESCALATE_ON_DISPERSION="${ESCALATE_ON_DISPERSION:-1}"
export LBFGSB_ABORT_ON_DIAGNOSTIC_FAILURE="${LBFGSB_ABORT_ON_DIAGNOSTIC_FAILURE:-1}"
export LBFGSB_REMOVAL_ACTIVE_SET="${LBFGSB_REMOVAL_ACTIVE_SET:-1}"
export LBFGSB_REMOVAL_MAX_ROUNDS="${LBFGSB_REMOVAL_MAX_ROUNDS:-32}"
export LBFGSB_REMOVAL_BATCH_SIZE="${LBFGSB_REMOVAL_BATCH_SIZE:-8}"

if [[ "${EZDELAY_SKIP_CONDA:-0}" != "1" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  elif [[ -f "/apps/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/apps/anaconda3/etc/profile.d/conda.sh"
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate "${EZDELAY_CONDA_ENV:-cap6}"
  fi
fi
