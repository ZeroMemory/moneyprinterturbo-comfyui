@echo off
setlocal DisableDelayedExpansion
rem Probe installed runtimes without triggering a Python manager download.
set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"
set "PYLAUNCHER_ALLOW_INSTALL="
set "MPT_TARGET=%MPT_ADDON_BOOTSTRAP_TARGET%"
set "REQUESTED_PYTHON="
set "TAKE_NEXT="
set "PYTHON_CMD="
set "INSTALLER=%~dp0install.py"
set "NO_ARGS="
if "%~1"=="" set "NO_ARGS=1"
:parse
if "%~1"=="" goto select_python
if "%TAKE_NEXT%"=="python" (
    set "REQUESTED_PYTHON=%~1"
    set "TAKE_NEXT="
    goto advance
)
if "%TAKE_NEXT%"=="skip" (
    set "TAKE_NEXT="
    goto advance
)
set "ARG=%~1"
if "%ARG%"=="--mpt-python" (set "TAKE_NEXT=python"& goto advance)
if "%ARG:~0,13%"=="--mpt-python=" (set "REQUESTED_PYTHON=%ARG:~13%"& goto advance)
if "%ARG%"=="install" goto advance
if "%ARG%"=="configure" goto advance
if "%ARG%"=="uninstall" goto advance
if "%ARG%"=="--help" goto advance
if "%ARG%"=="-h" goto advance
if "%ARG%"=="--external-server" goto advance
if "%ARG%"=="--allow-insecure-comfyui-http" goto advance
for /f "tokens=1 delims==" %%A in ("%ARG%") do set "ARG_HEAD=%%A"
if "%ARG:~0,1%"=="-" if not "%ARG%"=="%ARG_HEAD%" goto advance
if "%ARG:~0,1%"=="-" (set "TAKE_NEXT=skip"& goto advance)
if not defined MPT_TARGET set "MPT_TARGET=%~1"
:advance
shift
goto parse

:select_python
if defined REQUESTED_PYTHON (
    set "CHECK_PYTHON=%REQUESTED_PYTHON%"
    call :check_file
    if not defined PYTHON_CMD (
        echo The --mpt-python executable must run Python 3.11 or later. 1>&2
        exit /b 1
    )
) else (
    call :project_python
    if not defined PYTHON_CMD (
        for /f "delims=" %%P in ('where py.exe 2^>nul') do (
            set "CHECK_PYTHON=%%P"
            if not defined PYTHON_CMD call :check_launcher
        )
    )
    if not defined PYTHON_CMD (
        for /f "delims=" %%P in ('where python.exe 2^>nul') do (
            set "CHECK_PYTHON=%%P"
            if not defined PYTHON_CMD call :check_file
        )
    )
)
if defined PYTHON_CMD goto launch
if not defined NO_ARGS goto missing_python
set /p "MPT_TARGET=MoneyPrinterTurbo folder: "
set "MPT_TARGET=%MPT_TARGET:"=%"
call :project_python
if not defined PYTHON_CMD goto missing_python
set "MPT_ADDON_BOOTSTRAP_TARGET=%MPT_TARGET%"
:launch
%PYTHON_CMD% "%INSTALLER%" %*
:done
set "result=%errorlevel%"
if defined NO_ARGS pause
exit /b %result%

:missing_python
echo Python 3.11 or later was not found. Pass the MPT folder or use --mpt-python with its Python executable. 1>&2
exit /b 1

:project_python
if not defined MPT_TARGET exit /b
set "CHECK_PYTHON=%MPT_TARGET%\.venv\Scripts\python.exe"
call :check_file
if defined PYTHON_CMD exit /b
set "CHECK_PYTHON=%MPT_TARGET%\lib\python\python.exe"
call :check_file
if defined PYTHON_CMD exit /b
set "CHECK_PYTHON=%MPT_TARGET%\Scripts\python.exe"
call :check_file
exit /b

:check_file
if not exist "%CHECK_PYTHON%" exit /b
if /I "%CHECK_PYTHON%"=="%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" exit /b
"%CHECK_PYTHON%" -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 set PYTHON_CMD="%CHECK_PYTHON%"
exit /b

:check_launcher
if not exist "%CHECK_PYTHON%" exit /b
"%CHECK_PYTHON%" -3 -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 set PYTHON_CMD="%CHECK_PYTHON%" -3
exit /b
