@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 ( echo Python 3.12 is required. Install from https://www.python.org/downloads/ & pause & exit /b 1 )
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 ( echo Python 3.12 installation required & pause & exit /b 1 )
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 ( echo Dependency installation failed & pause & exit /b 1 )
)
.venv\Scripts\python.exe gui.py
if errorlevel 1 pause
