@echo off
REM Test the chatbot once from the command line; output goes to logs\chat_check.log (Claude reads it).
cd /d "%~dp0.."
set PYTHONPATH=%~dp0..\src
if not exist logs mkdir logs
echo Asking the chatbot one test question... (two questions, about 1-2 minutes)
.venv\Scripts\python.exe -m dashboard.chat UAL "How is our operating margin compared with the other airlines?" > logs\chat_check.log 2>&1
.venv\Scripts\python.exe -m dashboard.chat UAL "What if fuel rises 20%% and rates rise 1 point?" 4 >> logs\chat_check.log 2>&1
type logs\chat_check.log
echo.
echo Done. Type "done" in the Claude chat.
pause
