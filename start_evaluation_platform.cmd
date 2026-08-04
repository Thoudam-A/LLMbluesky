@echo off
setlocal

set "REPO=%~dp0"
set "PYTHON=%ATC_PYTHON%"

if not defined PYTHON if exist "%REPO%.venv\Scripts\python.exe" set "PYTHON=%REPO%.venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"

cd /d "%REPO%"
"%PYTHON%" -m evaluation_platform.server --open

endlocal
