@echo off
REM Starts the chat page at http://localhost:8501 (the Power BI button opens it). Close this window to stop it.
cd /d "%~dp0.."
set PYTHONPATH=%~dp0..\src
.venv\Scripts\python.exe -m pip install -q -r requirements-chat.txt --disable-pip-version-check
echo.
echo Chat page: http://localhost:8501   (keep this window open while you use the dashboard)
.venv\Scripts\python.exe -m streamlit run src\dashboard\chat_app.py
pause
