@echo off
REM One-click launcher for Aspect-Based Hotel Review Analysis.
REM Double-click this file: opens a native window (pywebview), or the browser if
REM pywebview is missing. Uses .venv\Scripts\python.exe when it exists, so a
REM double-click works without activating the environment first.
REM
REM   run.bat --port 9000
REM   run.bat --browser
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if "%PY%"=="python" (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ first.
    pause
    exit /b 1
  )
)
if "%~1"=="--browser" (
  "%PY%" app\serve.py %2 %3 %4 %5
) else (
  "%PY%" app\serve.py --native %*
)
if errorlevel 1 pause
