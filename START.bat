@echo off
setlocal EnableDelayedExpansion
title Ghost in the GPU

:: ─── Enable ANSI colors (clean, no 'rem' artifact) ────
for /f "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "ESC=%%b"
if defined ESC (
    set "RST=!ESC![0m"
    set "BOLD=!ESC![1m"
    set "DIM=!ESC![2m"
    set "RED=!ESC![91m"
    set "GRN=!ESC![92m"
    set "YEL=!ESC![93m"
    set "MAG=!ESC![95m"
    set "CYN=!ESC![96m"
    set "WHT=!ESC![97m"
) else (
    set "RST="
    set "BOLD="
    set "DIM="
    set "RED="
    set "GRN="
    set "YEL="
    set "MAG="
    set "CYN="
    set "WHT="
)

cd /d "%~dp0"

:: ─── Banner ───────────────────────────────────────────
echo.
echo !CYN!+==========================================================+!RST!
echo !CYN!^|!RST!  !BOLD!!WHT!GHOST IN THE GPU!RST!                                        !CYN!^|!RST!
echo !CYN!^|!RST!  !DIM!Clone your visual sensory mind into a neural core!RST!            !CYN!^|!RST!
echo !CYN!+==========================================================+!RST!
echo.

:: ─── 1/4 Python ───────────────────────────────────────
echo !DIM![1/4]!RST! !CYN!Checking Python!RST!...
where python >nul 2>nul
if !errorlevel! neq 0 (
    echo   !RED![X]!RST! Python not found in PATH.
    echo     !YEL!Install Python 3.10+ from https://python.org!RST!
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo   !GRN![OK]!RST! Python !WHT!!PYVER!!RST!

:: ─── 2/4 venv ─────────────────────────────────────────
echo !DIM![2/4]!RST! !CYN!Checking virtual environment!RST!...
if not exist ".venv\Scripts\python.exe" (
    echo   !YEL![*]!RST! Creating .venv...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo   !RED![X]!RST! Failed to create venv.
        pause
        exit /b 1
    )
    echo   !YEL![*]!RST! Installing dependencies ^(first run, be patient^)...
    .venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    .venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
    if !errorlevel! neq 0 (
        echo   !RED![X]!RST! Dependency install failed.
        pause
        exit /b 1
    )
)
echo   !GRN![OK]!RST! Venv ready.

if not exist ".venv\Lib\site-packages\uvicorn" (
    echo   !YEL![*]!RST! Missing deps - reinstalling...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo   !RED![X]!RST! Dependency install failed.
        pause
        exit /b 1
    )
)
echo   !GRN![OK]!RST! Dependencies verified.

:: ─── 3/4 Port ─────────────────────────────────────────
echo !DIM![3/4]!RST! !CYN!Checking port 8770!RST!...
set "PORT_BUSY=0"
for /f "tokens=*" %%L in ('netstat -ano ^| findstr ":8770 " ^| findstr "LISTENING"') do set "PORT_BUSY=1"

if "!PORT_BUSY!"=="1" (
    echo   !YEL![!]!RST! Port 8770 is already in use.
    choice /c YN /n /m "   Kill existing server and start fresh? [Y/N] "
    if !errorlevel! neq 1 (
        echo   !YEL![*]!RST! Opening existing server...
        call :open_browser
        exit /b 0
    )
    echo   !YEL![*]!RST! Killing old server...
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8770 " ^| findstr "LISTENING"') do (
        taskkill /f /pid %%P >nul 2>nul
    )
    timeout /t 1 /nobreak >nul
)
echo   !GRN![OK]!RST! Port 8770 is free.

:: ─── 4/4 Launch (Single Window with Live Logs) ────────
echo !DIM![4/4]!RST! !CYN!Starting Ghost Neural Engine!RST!...
echo.
echo !GRN!+==========================================================+!RST!
echo !GRN!^|!RST!  !GRN!!BOLD![OK] Ghost is starting at http://127.0.0.1:8770!RST!       !GRN!^|!RST!
echo !GRN!^|!RST!  !DIM!Live server logs below. Press Ctrl+C to stop.!RST!         !GRN!^|!RST!
echo !GRN!+==========================================================+!RST!
echo.

:: Background helper waits for TCP 8770 to open, then immediately launches Brave
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "$d=(Get-Date).AddSeconds(20); while((Get-Date) -lt $d){ try{ $c=New-Object System.Net.Sockets.TcpClient('127.0.0.1',8770); $c.Close(); break } catch{ Start-Sleep -Milliseconds 100 } }; $b1='C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe'; $b2=\"$env:LOCALAPPDATA\BraveSoftware\Brave-Browser\Application\brave.exe\"; if(Test-Path $b1){ Start-Process $b1 'http://127.0.0.1:8770/' } elseif(Test-Path $b2){ Start-Process $b2 'http://127.0.0.1:8770/' } else { Start-Process 'http://127.0.0.1:8770/' }"

:: Run uvicorn directly in THIS terminal so live logs stream here
.venv\Scripts\python.exe -m uvicorn ghost.rater.server:app --host 127.0.0.1 --port 8770
exit /b 0

:: ─── Browser opener (when reusing existing server) ────
:open_browser
if exist "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" (
    start "" "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" "http://127.0.0.1:8770/"
) else if exist "%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe" (
    start "" "%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe" "http://127.0.0.1:8770/"
) else (
    start "" "http://127.0.0.1:8770/"
)
exit /b 0
