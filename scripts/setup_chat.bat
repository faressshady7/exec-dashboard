@echo off
REM One-time chatbot setup: creates its read-only database account and saves the connection in .env.
cd /d "%~dp0.."
set PYTHONPATH=%~dp0..\src
.venv\Scripts\python.exe -m pip install -q -r requirements-chat.txt --disable-pip-version-check
.venv\Scripts\python.exe -m dashboard.chat_setup
pause
