@echo off
REM Runs one pipeline step. Works from any folder, even after the project is moved.
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
if not exist logs mkdir logs
if not exist .venv\Scripts\python.exe (
  echo First run in this folder: setting up Python 3.13, 2-3 minutes...
  py -3.13 -m venv .venv
  if errorlevel 1 ( echo Python 3.13 not found. Run: py install 3.13 & pause & exit /b 1 )
)
echo Checking libraries...
.venv\Scripts\python.exe -m pip install -q -r requirements.txt --disable-pip-version-check
echo Running ai ... (output also saved to logs\ai.log)
.venv\Scripts\python.exe -m dashboard.ai %* > logs\ai.log 2>&1
type logs\ai.log
echo.
echo Finished. Type "done" in the Claude chat - Claude reads logs\ai.log itself.
pause
