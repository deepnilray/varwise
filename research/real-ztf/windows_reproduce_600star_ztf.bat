@echo off
setlocal
cd /d "%~dp0"
echo Reproducing the completed real-label ZTF experiment (600 stars).
echo No synthetic photometry or guessed labels are used.
if not exist results\ztf_supervised\runs\n200\accepted_training_sources.json (
  echo ERROR: The archived 600-star accepted-source manifest is missing.
  pause
  exit /b 1
)
py -3.12 verify_package.py
if errorlevel 1 goto :failed
if not exist .venv\Scripts\activate.bat (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :failed
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
python ztf_supervised.py --per-class 200 --train-only
if errorlevel 1 goto :failed
python ztf_supervised_audit.py --per-class 200
if errorlevel 1 goto :failed
python ztf_learning_curve.py --per-class 200
if errorlevel 1 goto :failed
echo Reproduction successful. Check results\ztf_supervised\runs\n200\
pause
exit /b 0
:failed
echo ERROR: Reproduction failed. Send the entire output to Jarvis.
pause
exit /b 1
