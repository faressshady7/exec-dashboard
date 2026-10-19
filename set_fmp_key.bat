@echo off
REM Saves your FMP API key into .env (hidden while you type), then tests it with one call.
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
.venv\Scripts\python.exe -m dashboard.setkey
echo.
echo Finished. Type "done" in the Claude chat (do NOT paste the key there).
pause
