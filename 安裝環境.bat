@echo off
cd /d "%~dp0"
echo Installing the WSL side environment (venv + ffmpeg + python packages).
echo This may take several minutes on first run.
echo.
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '%CD:\=/%' 2>/dev/null; bash /mnt/d/claude/translate/scripts/setup_wsl.sh"
echo.
pause
