@echo off
cd /d "%~dp0"
echo Hand Portal website: http://localhost:8000/web/
echo Keep this window open while you use it. Close it to stop.
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8000/web/"
".venv\Scripts\python.exe" -m http.server 8000 --bind 127.0.0.1
pause
