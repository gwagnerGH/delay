#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export EZClimate_TM_ROOT="${EZClimate_TM_ROOT:-${PROJECT_ROOT}}"

if [[ "${EZDELAY_SKIP_CONDA:-0}" != "1" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  elif [[ -f "/apps/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/apps/anaconda3/etc/profile.d/conda.sh"
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate "${EZDELAY_CONDA_ENV:-EZClimate}"
  fi
fi
