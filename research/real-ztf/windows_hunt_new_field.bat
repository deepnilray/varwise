@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run windows_reproduce_600star_ztf.bat once to install Python dependencies.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Enter a NEW short tag per field, e.g. cygnus01. Only letters, digits, underscore or hyphen.
set /p TAG=Run tag: 
set /p RA=Right ascension in degrees [0 to 360): 
set /p DEC=Declination in degrees [-90 to 90]: 
echo Downloading only real public ZTF DR24 photometry. This may take a minute.
python ztf_discovery.py --ra "%RA%" --dec "%DEC%" --radius 45 --tag "%TAG%"
if errorlevel 1 goto :failed
echo Crosschecking published VSX, Gaia and older ZTF variable catalogs.
python ztf_catalog_check.py --fetch --ra "%RA%" --dec "%DEC%" --radius 50 --tag "%TAG%"
if errorlevel 1 goto :failed
echo Checking repeat signals, two filters, aliases and close neighbors.
python ztf_candidate_vet.py --tag "%TAG%"
if errorlevel 1 goto :failed
echo.
echo Results: results\fields\%TAG%\CANDIDATE_REVIEW.md
echo Details: results\fields\%TAG%\ztf_candidate_vetting.json
echo IMPORTANT: These are unconfirmed candidates, NOT discoveries.
pause
exit /b 0
:failed
echo The hunt stopped on an error; send the terminal output to Jarvis.
pause
exit /b 1
