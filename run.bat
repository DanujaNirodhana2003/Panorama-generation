@echo off
set "MARKER=.installed"

REM Check if marker file exists, if not install
if not exist "%MARKER%" goto install

REM Check if requirements.txt is newer than the marker file
for /f "tokens=*" %%a in ('xcopy /d /l /y requirements.txt "%MARKER%" ^| findstr /V "File(s)"') do set "UPDATED=%%a"
if defined UPDATED goto install

goto run

:install
echo Checking and installing requirements...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements.
    exit /b %ERRORLEVEL%
)
type nul > "%MARKER%"

:run
echo Running baseline.py...
python src\baseline.py
