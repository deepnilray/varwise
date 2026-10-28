@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Python environment not installed.
  echo First double-click windows_reproduce_600star_ztf.bat
  pause
  exit /b 1
)
.venv\Scripts\python.exe verify_package.py
if errorlevel 1 goto :failed
echo Checking the real 750-ZTF-source field from the included archive.
.venv\Scripts\python.exe ztf_candidate_vet.py
if errorlevel 1 goto :failed
echo.
echo Open results\CANDIDATE_REVIEW.md to read the results.
echo Individual evidence: results\ztf_candidate_vetting.json
echo NO object is a verified discovery without independent confirmation.
pause
exit /b 0
:failed
echo Verification stopped on an error. Send the terminal output to Jarvis.
pause
exit /b 1
