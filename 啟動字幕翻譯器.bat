@echo off
cd /d "%~dp0"
where pythonw.exe >nul 2>&1
if %errorlevel%==0 (
    start "" pythonw.exe "gui\app.pyw"
) else (
    start "" py.exe -3 "gui\app.pyw"
)
