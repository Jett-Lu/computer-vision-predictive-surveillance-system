@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" src\main.py --process-media
echo.
echo Annotated files are available in the output folder.
pause
