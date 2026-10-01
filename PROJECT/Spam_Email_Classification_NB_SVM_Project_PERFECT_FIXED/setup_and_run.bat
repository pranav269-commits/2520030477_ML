@echo off
setlocal
cd /d "%~dp0"

echo.
echo ===========================================================
echo   Spam Email Classification - Isolated Setup and Run
echo ===========================================================
echo.

python --version
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or newer and enable Add Python to PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating an isolated Python environment for this project...
  python -m venv .venv
  if errorlevel 1 (
    echo Could not create the local virtual environment.
    pause
    exit /b 1
  )
)

echo.
echo Installing or updating project packages inside .venv...
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check your internet connection and try again.
  pause
  exit /b 1
)

echo.
echo Running project preflight checks...
".venv\Scripts\python.exe" preflight.py
if errorlevel 1 (
  echo.
  echo Preflight failed. Copy the error shown above if you need help.
  pause
  exit /b 1
)

echo.
echo Starting the project with its isolated environment...
echo The correct local URL will be printed below and opened automatically.
echo Keep this window open while using the website.
echo.
".venv\Scripts\python.exe" app.py
pause
