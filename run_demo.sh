#!/usr/bin/env bash
set -euo pipefail
# Small helper to start the application using the project's virtualenv
ROOT_DIR=$(dirname "$0")
VENV="$ROOT_DIR/.venv"
if [ ! -d "$VENV" ]; then
  echo "Virtualenv not found at $VENV — create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
echo "Activating virtualenv and starting app (reads config.yaml in project root)..."
source "$VENV/bin/activate"
python3 -m src.main
