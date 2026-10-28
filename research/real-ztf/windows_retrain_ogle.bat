@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run windows_run.bat first to install dependencies.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Retraining exclusively with published OGLE catalog labels and real photometry.
echo Requires internet for any OGLE records missing from the archive.
python run.py --n-per-class 500 --workers 8
if errorlevel 1 goto :failed
python external_smc.py
if errorlevel 1 goto :failed
python audit.py
if errorlevel 1 goto :failed
echo Training and integrity checks finished. Send the metrics JSON to Jarvis.
pause
exit /b 0
:failed
echo Retraining failed. Send the terminal output to Jarvis; don't report a score.
pause
exit /b 1
