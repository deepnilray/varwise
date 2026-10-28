@echo off
setlocal
cd /d "%~dp0"
echo This runs REAL data only. Model was trained on OGLE. Candidate labels are unvetted.
py -3.12 -m venv .venv
if errorlevel 1 goto :failed
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
python audit.py
if errorlevel 1 goto :failed
python ztf_discovery.py --input data\ztf_dr24_field_298_29_45arcsec.json
if errorlevel 1 goto :failed
python ztf_catalog_check.py
if errorlevel 1 goto :failed
python ztf_integrity.py
if errorlevel 1 goto :failed
python ztf_plot.py
if errorlevel 1 goto :failed
echo.
echo Done. Real candidates: results\ztf_catalog_check.json
echo Actual model metrics: results\metrics.json and results\external_smc_metrics.json
pause
exit /b 0
:failed
echo The process stopped on an error. Send the exact output to Jarvis.
pause
exit /b 1
