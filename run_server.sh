#!/bin/bash

# ===================================================
# Run Server Script for Fractalic Application (Dev Auto-Reload)
# ===================================================
# Adds automatic reload on file changes using uvicorn --reload
# Set DISABLE_RELOAD=1 to disable.
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

if [ "${DISABLE_RELOAD}" = "1" ]; then
  RELOAD_FLAG=""
  echo "🚀 Starting Fractalic server (no auto-reload)"
else
  RELOAD_FLAG="--reload"
  echo "🚀 Starting Fractalic server with auto-reload enabled"
fi

echo "📂 Working directory: $(pwd)"
echo "🌐 Server will be available at: http://localhost:8000"
echo ""

# If watchfiles not installed, suggest it (non-fatal)
if ! python -c "import watchfiles" 2>/dev/null; then
  echo "ℹ️  (Optional) Install 'watchfiles' for faster reload: pip install watchfiles"
fi

# Run Uvicorn server using module notation (more robust)
uvicorn core.ui_server.server:app --host 0.0.0.0 --port 8000 $RELOAD_FLAG
