#!/bin/bash

# ===================================================
# Run Server Script for Fractalic Application
# ===================================================
# 
# Description:
#   This script starts the Uvicorn server for the Fractalic application.
#   It activates the virtual environment, navigates to the server directory,
#   and launches the server with hot-reload enabled.
#
# Usage:
#   ./run_server.sh
#
# Requirements:
#   - Virtual environment at ./venv
#   - Server module at ./core/ui_server/server.py
# ===================================================

# Get the directory of the script (relative path resolution)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"
SERVER_DIR="$SCRIPT_DIR/core/ui_server"

# Activate virtual environment
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Navigate to server directory
cd "$SERVER_DIR" || { echo "Error: Failed to enter $SERVER_DIR"; exit 1; }

# Run Uvicorn server (fixing module import issue)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
