#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv_optuna"

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  IS_SOURCED=1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Erreur: python3 est introuvable dans PATH." >&2
  echo "Installe python3, puis relance ce script." >&2
  if [[ $IS_SOURCED -eq 1 ]]; then
    return 1
  fi
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Création du virtualenv: $VENV_DIR"
  if ! python3 -m venv "$VENV_DIR"; then
    echo "Erreur: échec de création du venv. Sur Debian/Ubuntu: sudo apt-get install python3-venv" >&2
    if [[ $IS_SOURCED -eq 1 ]]; then
      return 1
    fi
    exit 1
  fi
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip

if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
  "$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

# Activation du venv dans le shell courant
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ $IS_SOURCED -eq 1 ]]; then
  echo "venv_optuna activé."
  if [[ $# -gt 0 ]]; then
    "$@"
  fi
else
  if [[ $# -gt 0 ]]; then
    exec "$@"
  else
    echo "venv_optuna prêt. Ouverture d'un shell avec l'environnement activé."
    exec "${SHELL:-bash}"
  fi
fi
