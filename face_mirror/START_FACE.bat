@echo off
setlocal EnableDelayedExpansion
title Ghost Face Mirror // GPU ArcFace Engine

:: ─── Enable ANSI colors ───────────────────────────────
for /f "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "ESC=%%b"
if defined ESC (
    set "RST=!ESC![0m"
    set "BOLD=!ESC![1m"
    set "DIM=!ESC![2m"
    set "RED=!ESC![91m"
    set "GRN=!ESC![92m"
    set "YEL=!ESC![93m"
    set "CYN=!ESC![96m"
    set "WHT=!ESC![97m"
) else (
    set "RST="
    set "BOLD="
    set "DIM="
    set "RED="
    set "GRN="
    set "YEL="
    set "CYN="
    set "WHT="
)

cd /d "%~dp0.."

echo.
echo !CYN!+==============================================================+!RST!
echo !CYN!^|!RST!  !BOLD!!WHT!GHOST FACE MIRROR — 100%% GPU ARCFACE LOOKALIKE ENGINE!RST!       !CYN!^|!RST!
echo !CYN!^|!RST!  !DIM!Standalone subproject · Shared corpus · Strict Pose Gate!RST!    !CYN!^|!RST!
echo !CYN!+==============================================================+!RST!
echo.

:: ─── 1/5 Check Python & Virtual Environment ──────────
echo !DIM![1/5]!RST! !CYN!Checking Python & GPU Environment!RST!...
if not exist ".venv\Scripts\python.exe" (
    echo   !YEL![*]!RST! Creating .venv for first-time user...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo   !RED![X]!RST! Failed to create .venv. Please install Python 3.10+.
        pause
        exit /b 1
    )
    echo   !YEL![*]!RST! Installing CUDA-enabled PyTorch ^& dependencies...
    .venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
    .venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
)

:: Ensure face_mirror specific lightweight packages exist
.venv\Scripts\python.exe -c "import cv2, onnx, onnx2torch" 2>nul
if !errorlevel! neq 0 (
    echo   !YEL![*]!RST! Installing Face Mirror vision packages ^(opencv, onnx, onnx2torch^)...
    .venv\Scripts\python.exe -m pip install opencv-python onnx onnx2torch --quiet
)
echo   !GRN![OK]!RST! GPU Environment verified.

:: ─── 2/5 Verify/Bootstrap GPU Models ─────────────────
echo !DIM![2/5]!RST! !CYN!Verifying SCRFD 10G ^& ArcFace R50 GPU Models!RST!...
set "PYTHONPATH=%CD%"
.venv\Scripts\python.exe -c "from face_mirror.config import ensure_gpu_models; ensure_gpu_models()"
if !errorlevel! neq 0 (
    echo   !RED![X]!RST! Model initialization failed.
    pause
    exit /b 1
)
echo   !GRN![OK]!RST! GPU models ready in face_mirror\models\.

:: ─── 3/5 Dynamic Incremental Corpus Face Sync ────────
echo !DIM![3/5]!RST! !CYN!Syncing new corpus photos into face_cache.db on GPU!RST!...
.venv\Scripts\python.exe -m face_mirror.cache
if !errorlevel! neq 0 (
    echo   !RED![X]!RST! Corpus sync encountered an error.
    pause
    exit /b 1
)

:: ─── 4/5 Check Port 8775 ─────────────────────────────
echo !DIM![4/5]!RST! !CYN!Checking port 8775!RST!...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8775 " ^| findstr "LISTENING"') do (
    echo   !YEL![*]!RST! Restarting existing Face Mirror server on 8775...
    taskkill /f /pid %%P >nul 2>nul
)
echo   !GRN![OK]!RST! Port 8775 ready.

:: ─── 5/5 Launch Server & Open Browser ────────────────
echo !DIM![5/5]!RST! !CYN!Launching Ghost Face Mirror UI at http://127.0.0.1:8775!RST!...
echo.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "$d=(Get-Date).AddSeconds(20); while((Get-Date) -lt $d){ try{ $c=New-Object System.Net.Sockets.TcpClient('127.0.0.1',8775); $c.Close(); break } catch{ Start-Sleep -Milliseconds 100 } }; Start-Process 'http://127.0.0.1:8775/'"

.venv\Scripts\python.exe -m uvicorn face_mirror.server:app --host 127.0.0.1 --port 8775 --http httptools --backlog 2048
exit /b 0
