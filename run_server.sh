#!/bin/bash

# ===================================================
# Run Server Script for Fractalic Application
# ===================================================
# 
# Description:
#   This script starts the Uvicorn server from the project root directory.
#   This approach is more robust and avoids path-related issues.
#
# Usage:
#   ./run_server.sh
#
# Requirements:
#   - Virtual environment at ./.venv
#   - Server module at ./core/ui_server/server.py
# ===================================================

# Get the directory of the script (project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

# Activate virtual environment
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Stay in project root directory
cd "$SCRIPT_DIR" || { echo "❌ Error: Failed to enter $SCRIPT_DIR"; exit 1; }

echo "🚀 Starting Fractalic server from project root..."
echo "📂 Working directory: $(pwd)"
echo "🌐 Server will be available at: http://localhost:8000"
echo ""

# Run Uvicorn server using module notation (more robust)
uvicorn core.ui_server.server:app --host 0.0.0.0 --port 8000 --reload
