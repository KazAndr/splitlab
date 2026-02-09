#!/usr/bin/env bash
set -euo pipefail

mkdir -p legacy
if [ -f gui.py ]; then
  git mv gui.py legacy/gui_matplotlib_old.py
  echo "Moved gui.py -> legacy/gui_matplotlib_old.py"
fi

if [ -d icons ]; then
  mkdir -p src/splitlab/resources
  git mv icons src/splitlab/resources/icons || true
  echo "Moved icons -> src/splitlab/resources/icons"
fi

echo "Migration step done. Add scaffold files and commit."
