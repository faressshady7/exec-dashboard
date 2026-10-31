@echo off
REM After resetting the database password in Supabase: saves it into DATABASE_URL in .env and tests it.
cd /d "%~dp0.."
set PYTHONPATH=%~dp0..\src
.venv\Scripts\python.exe -m dashboard.setdbpw
pause
