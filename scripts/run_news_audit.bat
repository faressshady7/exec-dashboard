@echo off
REM News filter audit. No argument: make docs\news_audit.csv. "score": score your y/n answers.
cd /d "%~dp0.."
set PYTHONPATH=%~dp0..\src
if not exist logs mkdir logs
set ARG=
if /i "%1"=="score" set ARG=--score
.venv\Scripts\python.exe -m dashboard.news_audit %ARG% > logs\news_audit.log 2>&1
type logs\news_audit.log
echo.
echo Finished. Type "done" in the Claude chat - Claude reads logs\news_audit.log itself.
pause
