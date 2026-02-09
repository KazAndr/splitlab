#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11.9}"
VENV_NAME="${VENV_NAME:-splitlab-${PYTHON_VERSION}}"

pyenv install -s "$PYTHON_VERSION"
pyenv virtualenv -f "$PYTHON_VERSION" "$VENV_NAME"
pyenv local "$VENV_NAME"

python -m pip install -U pip setuptools wheel
pip install -e ".[dev]"

echo "OK: pyenv local -> $(pyenv version)"
echo "Run: python -m splitlab  (or: splitlab)"
