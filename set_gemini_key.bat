@echo off
REM Saves your Gemini API key into .env (private file, never shared or uploaded).
cd /d "%~dp0"
echo Paste your Gemini API key (right-click to paste) and press Enter:
set /p KEY=
.venv\Scripts\python.exe -c "import pathlib,sys; p=pathlib.Path('.env'); k=sys.argv[1].strip(); lines=[l for l in p.read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('GEMINI_API_KEY=')] if p.exists() else []; lines.append('GEMINI_API_KEY='+k); p.write_text(chr(10).join(lines)+chr(10), encoding='utf-8'); print('Saved. Key ends with ...'+k[-4:])" "%KEY%"
pause
