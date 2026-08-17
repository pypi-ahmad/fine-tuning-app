#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "Fine-Tuning Studio requires uv: https://docs.astral.sh/uv/getting-started/installation/"
  read -r -p "Press Enter to close..." _
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "First-time setup: creating the virtual environment and installing dependencies."
  echo "This can take several minutes, mostly for the PyTorch/CUDA download."
  echo
  if ! uv sync --locked; then
    echo
    echo "Setup failed. See the output above for details."
    read -r -p "Press Enter to close..." _
    exit 1
  fi
fi

uv run --locked fine-tuning-studio run
status=$?
if [ "$status" -ne 0 ]; then
  echo
  echo "Fine-Tuning Studio stopped with an error."
  read -r -p "Press Enter to close..." _
fi
exit "$status"
