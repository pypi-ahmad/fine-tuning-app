@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Fine-Tuning Studio requires uv: https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)

if not exist ".venv" (
  echo First-time setup: creating the virtual environment and installing dependencies.
  echo This can take several minutes, mostly for the PyTorch/CUDA download.
  echo.
  uv sync --locked
  if errorlevel 1 (
    echo.
    echo Setup failed. See the output above for details.
    pause
    exit /b 1
  )
)

uv run --locked fine-tuning-studio run
if errorlevel 1 (
  echo.
  echo Fine-Tuning Studio stopped with an error.
  pause
)
