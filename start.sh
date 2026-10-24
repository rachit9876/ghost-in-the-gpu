#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "========================================================"
echo "              GHOST IN THE GPU"
echo "  Clone your visual sensory mind into a neural core"
echo "========================================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3.10+ was not found. Please install Python."
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    python3 -m venv .venv
    echo "[*] Installing dependencies..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

if lsof -i :8770 >/dev/null 2>&1; then
    echo "[!] Ghost is ALREADY RUNNING on port 8770!"
    echo "[*] Opening your existing session in browser..."
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://127.0.0.1:8770/"
    elif command -v open &> /dev/null; then
        open "http://127.0.0.1:8770/"
    fi
    exit 0
fi

echo "[*] Initializing Ghost Neural Engine on http://127.0.0.1:8770 ..."
if command -v xdg-open &> /dev/null; then
    (sleep 1 && xdg-open "http://127.0.0.1:8770/") &
elif command -v open &> /dev/null; then
    (sleep 1 && open "http://127.0.0.1:8770/") &
fi
.venv/bin/python -m uvicorn ghost.rater.server:app --host 127.0.0.1 --port 8770
