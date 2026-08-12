#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "Fine-Tuning Studio requires uv: https://docs.astral.sh/uv/getting-started/installation/"
  read -r -p "Press Enter to close..." _
  exit 1
fi
uv run --locked fine-tuning-studio run
status=$?
if [ "$status" -ne 0 ]; then
  echo
  echo "Fine-Tuning Studio stopped with an error."
  read -r -p "Press Enter to close..." _
fi
exit "$status"
