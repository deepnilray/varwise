@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Deepnil Real ZTF DR24 Supervised Trainer
echo Exact published AAVSO VSX classes: RRAB / DCEP / EW
echo No simulated, invented or pseudo-labeled training data.
echo ============================================================
echo.
py -3.12 verify_package.py
if errorlevel 1 goto :failed
set "TARGET=400"
set /p TARGET=Accepted real stars PER CLASS [default 400; current benchmark 200]: 
if not defined TARGET set "TARGET=400"
set "CATALOG_ROWS=1000"
set /p CATALOG_ROWS=Published VSX catalog rows per 30-degree RA sector [default 1000]: 
if not defined CATALOG_ROWS set "CATALOG_ROWS=1000"
if not exist .venv\Scripts\activate.bat (
  echo Preparing Python 3.12 environment. This requires internet.
  py -3.12 -m venv .venv
  if errorlevel 1 goto :failed
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
echo.
echo Downloading authentic published labels and telescope observations.
echo Depending on target count/network, this can take a long time.
echo Safe to rerun: previous real data is cached and never replaced with fabricated rows.
python ztf_supervised.py --per-class "%TARGET%" --catalog-per-sector "%CATALOG_ROWS%" --workers 3 --max-attempts 3000
if errorlevel 1 goto :failed
python ztf_supervised_audit.py --per-class "%TARGET%"
if errorlevel 1 goto :failed
python ztf_learning_curve.py --per-class "%TARGET%"
if errorlevel 1 goto :failed
echo.
echo Training and source integrity audit PASSED.
echo Results folder: results\ztf_supervised\runs\n%TARGET%\
echo Send training_metrics.json and training_integrity_audit.json to Jarvis.
echo Do not call model a discovery system until unknown-object vetting is complete.
pause
exit /b 0
:failed
echo.
echo ERROR: Training or audit failed. No success claim. Send the full console output.
pause
exit /b 1
