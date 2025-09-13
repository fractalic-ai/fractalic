#!/bin/bash

# ===================================================
# Fractalic Local Development Setup Script
# ===================================================
# 
# Description:
#   Sets up complete local development environment with:
#   - Python virtual environment
#   - Fractalic backend dependencies
#   - Fractalic-UI frontend 
#   - Starts both backend and frontend servers
#
# Usage:
#   ./local-dev-setup.sh
#
# Requirements:
#   - Python 3.8+
#   - Node.js 16+
#   - Git
# ===================================================

set -e  # Exit on any error

echo "🚀 Setting up Fractalic Local Development Environment"
echo ""

# Get the directory of the script (should be fractalic root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if we're in the fractalic directory
if [ ! -f "$SCRIPT_DIR/fractalic.py" ]; then
    echo "❌ Error: This script must be run from the fractalic repository root"
    echo "Current directory: $SCRIPT_DIR"
    exit 1
fi

echo "📂 Working in: $SCRIPT_DIR"

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Set up Python virtual environment
echo ""
echo "🐍 Setting up Python virtual environment..."
VENV_PATH="$SCRIPT_DIR/.venv"

if [ -d "$VENV_PATH" ]; then
    echo "⚠️  Virtual environment already exists at $VENV_PATH"
    echo "   Remove it and re-run this script if you want a fresh setup"
else
    python3 -m venv "$VENV_PATH"
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"
echo "✅ Virtual environment activated"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

echo "✅ Python dependencies installed"

# Clone fractalic-ui if it doesn't exist
echo ""
echo "🎨 Setting up Fractalic UI..."
UI_PATH="$PARENT_DIR/fractalic-ui"

if [ -d "$UI_PATH" ]; then
    echo "✅ Fractalic UI already exists at $UI_PATH"
else
    echo "📥 Cloning Fractalic UI..."
    cd "$PARENT_DIR"
    git clone https://github.com/fractalic-ai/fractalic-ui.git
    echo "✅ Fractalic UI cloned"
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd "$UI_PATH"
npm install
echo "✅ Node.js dependencies installed"

# Return to fractalic directory
cd "$SCRIPT_DIR"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 Starting services..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo "👋 Goodbye!"
}

trap cleanup EXIT INT TERM

# Start backend server in background
echo "🔧 Starting backend server..."
source "$VENV_PATH/bin/activate"
./run_server.sh &
BACKEND_PID=$!
sleep 3
echo "✅ Backend server started (PID: $BACKEND_PID)"

# Start frontend server in background
echo "🎨 Starting frontend server..."
cd "$UI_PATH"
npm run dev &
FRONTEND_PID=$!
sleep 5
echo "✅ Frontend server started (PID: $FRONTEND_PID)"

# Return to fractalic directory
cd "$SCRIPT_DIR"

echo ""
echo "🌐 Services are running:"
echo "   📊 Frontend (UI): http://localhost:3000"
echo "   ⚙️  Backend API: http://localhost:8000"
echo "   🤖 AI Server: http://localhost:8001"
echo "   🔧 MCP Manager: http://localhost:5859"
echo ""
echo "👀 Opening frontend in browser..."

# Try to open browser (cross-platform)
if command -v open &> /dev/null; then
    # macOS
    open http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:3000
elif command -v start &> /dev/null; then
    # Windows
    start http://localhost:3000
else
    echo "   Please open http://localhost:3000 in your browser"
fi

echo ""
echo "✨ Development environment is ready!"
echo ""
echo "💡 Tips:"
echo "   • Frontend will auto-reload on file changes"
echo "   • Backend logs are visible in this terminal"
echo "   • Press Ctrl+C to stop all services"
echo ""
echo "🔧 Waiting for services (press Ctrl+C to stop)..."

# Wait for user to stop services
wait
