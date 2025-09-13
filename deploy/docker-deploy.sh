#!/bin/bash
# One-click Docker deployment script for Fractalic
# Usage: curl -s https://raw.githubusercontent.com/fractalic-ai/fractalic/main/deploy/docker-deploy.sh | bash

echo "🚀 Starting Fractalic One-Click Docker Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first:"
    echo "   https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Parse URL parameters (if any)
CONTAINER_NAME="${1:-fractalic-app}"
PORT_OFFSET="${2:-0}"

# Calculate ports with offset
FRONTEND_PORT=$((3000 + PORT_OFFSET))
BACKEND_PORT=$((8000 + PORT_OFFSET))
AI_SERVER_PORT=$((8001 + PORT_OFFSET))
MCP_MANAGER_PORT=$((5859 + PORT_OFFSET))

echo "📦 Container: $CONTAINER_NAME"
echo "🔌 Frontend: http://localhost:$FRONTEND_PORT"
echo "⚙️  Backend: http://localhost:$BACKEND_PORT"
echo "🤖 AI Server: http://localhost:$AI_SERVER_PORT"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "📥 Cloning repositories..."

# Clone fractalic
git clone https://github.com/fractalic-ai/fractalic.git
if [ $? -ne 0 ]; then
    echo "❌ Failed to clone fractalic repository"
    exit 1
fi

# Clone fractalic-ui
git clone https://github.com/fractalic-ai/fractalic-ui.git
if [ $? -ne 0 ]; then
    echo "❌ Failed to clone fractalic-ui repository"
    exit 1
fi

echo "🐳 Building and starting Docker container..."

# Use the existing Docker build script
cd fractalic
chmod +x docker_build_run.sh

# Set environment variables for the script
export CONTAINER_NAME="$CONTAINER_NAME"
export FRONTEND_PORT="$FRONTEND_PORT"
export BACKEND_PORT="$BACKEND_PORT"
export AI_SERVER_PORT="$AI_SERVER_PORT"
export MCP_MANAGER_PORT="$MCP_MANAGER_PORT"
export FRACTALIC_SKIP_EMPTY_CHECK="1"  # Skip empty directory check since we just cloned

# Run the Docker build script
./docker_build_run.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Fractalic deployed successfully!"
    echo ""
    echo "📋 Access URLs:"
    echo "   🖥️  Frontend: http://localhost:$FRONTEND_PORT"
    echo "   ⚙️  Backend API: http://localhost:$BACKEND_PORT"
    echo "   🤖 AI Server: http://localhost:$AI_SERVER_PORT"
    echo ""
    echo "🛑 To stop: docker stop $CONTAINER_NAME"
    echo "🗑️  To remove: docker rm $CONTAINER_NAME"
else
    echo "❌ Deployment failed. Check the logs above for details."
    exit 1
fi

# Cleanup
cd /
rm -rf "$TEMP_DIR"
