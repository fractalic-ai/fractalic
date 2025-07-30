#!/bin/bash

# Fractalic Docker Build & Run Script
# This script is designed for fresh GitHub installations via:
# curl -s https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docker_build_run.sh | bash
#
# It will clone both fractalic and fractalic-ui repositories and build/run them in Docker.
# For deploying existing local installations with custom content, use publish_docker.py instead.

# Parse command line arguments
BUILD_MODE="${1:-production}"  # Default to production mode
CONTAINER_NAME="${2:-fractalic-app}"

echo "Setting up Fractalic from GitHub..."
echo "🔧 Build mode: $BUILD_MODE"
echo "📦 Container name: $CONTAINER_NAME"

if [[ "$BUILD_MODE" == "production" ]]; then
    echo "🚀 Production mode: AI server only, optimized for production deployment"
elif [[ "$BUILD_MODE" == "full" ]]; then
    echo "🛠️ Full mode: Complete UI + AI server, for development"
else
    echo "❌ Invalid build mode: $BUILD_MODE"
    echo "Usage: $0 [production|full] [container-name]"
    echo "  production: AI server only (default)"
    echo "  full: Complete UI + AI server"
    exit 1
fi

# Safety check: Ensure we're in an empty or nearly empty directory
# This prevents accidentally polluting existing projects
CURRENT_FILES=$(ls -la | wc -l)
if [ $CURRENT_FILES -gt 3 ]; then  # . .. and maybe one other file
    echo "❌ ERROR: This directory is not empty!"
    echo "This script is designed for fresh installations in empty directories."
    echo ""
    echo "Current directory contents:"
    ls -la
    echo ""
    echo "If you want to deploy an existing Fractalic installation with custom content,"
    echo "use the publish_docker.py script instead:"
    echo "  python publish_docker.py --help"
    echo ""
    echo "If you want to continue anyway, create a new empty directory and run this script there."
    exit 1
fi

echo "✅ Directory is suitable for fresh installation"

# 1. Clone fractalic repository
echo "Cloning fractalic repository..."
git clone https://github.com/fractalic-ai/fractalic.git
cd fractalic

# 2. Clone fractalic-ui repository (only for full mode)
if [[ "$BUILD_MODE" == "full" ]]; then
    echo "Cloning fractalic-ui repository..."
    cd ..
    git clone https://github.com/fractalic-ai/fractalic-ui.git
else
    echo "⏭️ Skipping UI repository clone (production mode)"
    cd ..
fi

# 3. Create temporary build directory
BUILD_DIR=$(mktemp -d -t fractalic-build-XXXXXX)
echo "Created temporary build directory: $BUILD_DIR"

# 4. Copy source repos to build directory
echo "Copying source repositories to build directory..."
cp -r fractalic "$BUILD_DIR/"

if [[ "$BUILD_MODE" == "full" ]]; then
    cp -r fractalic-ui "$BUILD_DIR/"
fi

# 5. Copy Docker files to build directory
echo "Setting up Docker build context..."
cp -a fractalic/docker/. "$BUILD_DIR/"

# 6. Build Docker image from temporary directory
echo "Building Docker image..."
cd "$BUILD_DIR"

if [[ "$BUILD_MODE" == "production" ]]; then
    echo "🏭 Building production image (AI server only)..."
    docker build -f Dockerfile.production-ai-only -t $CONTAINER_NAME .
else
    echo "🛠️ Building full image (UI + AI server)..."
    docker build -f Dockerfile.production -t $CONTAINER_NAME .
fi

# 7. Return to original directory and cleanup
ORIGINAL_DIR="$PWD"
cd - > /dev/null
echo "Cleaning up temporary build directory..."
rm -rf "$BUILD_DIR"
cd fractalic  # Move into fractalic directory for container management

# 7. Check if container already exists and remove it
if [ "$(docker ps -qa -f name=$CONTAINER_NAME)" ]; then
  echo "Removing existing container..."
  docker stop $CONTAINER_NAME >/dev/null 2>&1
  docker rm $CONTAINER_NAME >/dev/null 2>&1
fi

# 8. Run the container with appropriate port mappings
echo "Starting container..."

if [[ "$BUILD_MODE" == "production" ]]; then
    echo "🚀 Starting production container (AI server on auto-detected port)..."
    # Find available port for AI server
    AI_PORT=8001
    while nc -z localhost $AI_PORT 2>/dev/null; do
        AI_PORT=$((AI_PORT + 1))
    done
    
    echo "📍 AI Server will be available on port $AI_PORT"
    
    docker run -d \
      -p $AI_PORT:8001 \
      -e TZ=$(cat /etc/timezone 2>/dev/null || date +%Z || echo "UTC") \
      -v /etc/localtime:/etc/localtime:ro \
      --name $CONTAINER_NAME \
      $CONTAINER_NAME
      
    # Store the port for later reference
    AI_SERVER_PORT=$AI_PORT
    
else
    echo "🛠️ Starting full container (all services)..."
    docker run -d \
      -p 8000:8000 \
      -p 3000:3000 \
      -p 8001:8001 \
      -p 8002:8002 \
      -p 8003:8003 \
      -p 8004:8004 \
      -p 5859:5859 \
      -e TZ=$(cat /etc/timezone 2>/dev/null || date +%Z || echo "UTC") \
      -v /etc/localtime:/etc/localtime:ro \
      --name $CONTAINER_NAME \
      $CONTAINER_NAME
      
    AI_SERVER_PORT=8001
fi

# 9. Wait for services to be ready
echo "Waiting for services to start (this may take a moment)..."
sleep 10

# 10. Check if services are running
if docker ps | grep -q $CONTAINER_NAME; then
  echo "✅ Container is running"
  
  if [[ "$BUILD_MODE" == "production" ]]; then
    # Production mode - only check AI server
    echo ""
    echo "🔍 Checking AI server..."
    sleep 5  # Give more time for AI server to start
    
    if curl -s http://localhost:$AI_SERVER_PORT/health >/dev/null 2>&1; then
      echo "✅ AI Server is available at: http://localhost:$AI_SERVER_PORT"
    else
      echo "⚠️ AI Server may still be starting at: http://localhost:$AI_SERVER_PORT"
    fi
    
    echo ""
    echo "🎉 Fractalic AI Server deployed successfully!"
    echo ""
    echo "🌐 AI Server: http://localhost:$AI_SERVER_PORT"
    echo "📚 API docs: http://localhost:$AI_SERVER_PORT/docs"
    echo "📝 Execute script: curl -X POST http://localhost:$AI_SERVER_PORT/execute -H \"Content-Type: application/json\" -d '{\"filename\": \"/payload/script.md\"}'"
    echo ""
    echo "🐳 Container: $CONTAINER_NAME"
    echo "🛑 Stop: docker stop $CONTAINER_NAME"
    echo "🗑️ Remove: docker rm $CONTAINER_NAME"
    echo "📋 Logs: docker logs $CONTAINER_NAME"
    
  else
    # Full mode - check all services
    # Check if service is responding
    if curl -s http://localhost:8000 >/dev/null 2>&1; then
      echo "✅ Backend is available at: http://localhost:8000"
    else
      echo "⚠️ Backend may still be starting at: http://localhost:8000"
    fi
    
    # Check AI server
    if curl -s http://localhost:8001 >/dev/null 2>&1; then
      echo "✅ AI Server is available at: http://localhost:8001"
    else
      echo "⚠️ AI Server may still be starting at: http://localhost:8001"
    fi
    
    # Check MCP manager
    if curl -s http://localhost:5859 >/dev/null 2>&1; then
      echo "✅ MCP Manager is available at: http://localhost:5859"
    else
      echo "⚠️ MCP Manager may still be starting at: http://localhost:5859"
    fi
    
    echo "✅ UI should be available at: http://localhost:3000"
    echo ""
    echo "📋 All Services Summary:"
    echo "   • Frontend UI:    http://localhost:3000"
    echo "   • Backend API:    http://localhost:8000"
    echo "   • AI Server:      http://localhost:8001"
    echo "   • MCP Manager:    http://localhost:5859"
    echo ""
    echo "Setup complete! Container is ready."
  fi
  
else
  echo "❌ Container failed to start. Check logs with: docker logs $CONTAINER_NAME"
fi