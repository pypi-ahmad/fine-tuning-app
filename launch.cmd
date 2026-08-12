@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Fine-Tuning Studio requires uv: https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)
uv run --locked streamlit run src/fine_tuning_studio/app.py
if errorlevel 1 (
  echo.
  echo Fine-Tuning Studio stopped with an error.
  pause
)

