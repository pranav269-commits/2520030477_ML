@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m ml.training
) else (
  echo Local environment not found. Run setup_and_run.bat first.
  pause
)
