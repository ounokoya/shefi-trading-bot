#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY="${PROJECT_ROOT}/venv_optuna/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi

CFG="configs/optuna_macd_momentum_two_tf_weighted_example.yaml"
if [[ $# -ge 1 ]]; then
  CFG="$1"
  shift
fi
if [[ "${CFG}" != /* ]]; then
  CFG="${PROJECT_ROOT}/${CFG}"
fi

exec "${PY}" "${PROJECT_ROOT}/scripts/29_optuna_macd_momentum_two_tf_weighted.py" --config "${CFG}" "$@"
