@echo off
REM Prints market cap, EV and multiples from the database (after the daily run has loaded prices).
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
if not exist logs mkdir logs
.venv\Scripts\python.exe -m dashboard.valuation > logs\valuation.log 2>&1
type logs\valuation.log
echo.
echo Finished. Type "done" in the Claude chat - Claude reads logs\valuation.log itself.
pause
