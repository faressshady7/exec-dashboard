@echo off
REM Runs the whole pipeline in order: bronze, silver, gold, compute, ai. Stops at the first failure.
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
if not exist logs mkdir logs
echo Checking libraries...
.venv\Scripts\python.exe -m pip install -q -r requirements.txt --disable-pip-version-check
for %%S in (bronze silver gold compute ai) do (
  echo.
  echo ===== %%S =====
  .venv\Scripts\python.exe -m dashboard.%%S > logs\%%S.log 2>&1
  if errorlevel 1 ( type logs\%%S.log & echo. & echo STOPPED: %%S failed. Type "done" in the Claude chat. & pause & exit /b 1 )
  type logs\%%S.log
)
echo.
echo All 5 steps finished. Type "done" in the Claude chat - Claude reads the logs itself.
pause
