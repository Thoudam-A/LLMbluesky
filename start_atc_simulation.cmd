@echo off
setlocal

set "REPO=%~dp0"
set "PROJECT=%REPO%bluesky_project"
set "PYTHON=%ATC_PYTHON%"

if not defined PYTHON if exist "%REPO%.venv\Scripts\python.exe" set "PYTHON=%REPO%.venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"

if not exist "%PROJECT%\BlueSky.py" (
    echo ERROR: BlueSky.py was not found in %PROJECT%
    pause
    exit /b 1
)

set "ATC_AUTO_START=1"
set "ATC_EVAL_URL=http://127.0.0.1:8765"
start "ATC Evaluation Service" /min /D "%REPO%" "%PYTHON%" -m evaluation_platform.server --open
cd /d "%PROJECT%"
start "ATC Simulation" "%PYTHON%" BlueSky.py

endlocal
