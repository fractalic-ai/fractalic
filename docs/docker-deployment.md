# Fractalic Docker Deployment & Publishing Guide

This comprehensive guide covers Fractalic Docker deployment scenarios and the publishing system architecture.

## 🚀 Deployment Scenarios

Fractalic supports THREE different deployment approaches with two build modes:

### Build Modes

- **Production Mode (Default)**: AI server only, no UI, lightweight, production-ready
- **Full Mode**: Complete setup with UI, backend, and AI server for development

### 🆕 Fresh Installation from GitHub (New Users)

Perfect for users who want to try Fractalic without any existing setup:

```bash
# Production deployment (AI server only, recommended)
mkdir my-fractalic && cd my-fractalic
curl -s https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docker_build_run.sh | bash

# Full deployment with UI (development/testing)
mkdir my-fractalic && cd my-fractalic
curl -s https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docker_build_run.sh | bash -s full
```

**What this does:**
- Checks for empty directory (safety)
- Clones fractalic and fractalic-ui from GitHub (if needed)
- Creates temporary build context
- Builds and runs Docker container with auto-port detection
- Cleans up temporary files
- Returns AI server URL and sample curl command

### 🔧 Deploy Existing Installation (Developers/Custom Content)

For developers with existing Fractalic installations containing custom tutorials, scripts, or modifications:

```bash
# Production deployment (AI server only, recommended)
python publish_docker.py --mode production

# Full deployment with UI (development/testing)
python publish_docker.py --mode full

# Custom container name and port handling
python publish_docker.py --name my-fractalic --mode production
```

**What this does:**
- Uses your current installation with all customizations
- Includes your tutorials, scripts, and modifications
- Deploys to Docker with automatic port detection
- Supports both production and full deployment modes

### ☁️ Cloud-Ready Registry Deployment (Production/Cloud)

For production deployments using pre-built Docker images from the registry:

```bash
# Deploy user scripts to a pre-built production container
python publisher_cli.py deploy-docker-registry \
  --script-name hello-world-test \
  --script-folder tutorials/01_Basics/hello-world \
  --container-name fractalic-production
```

**What this does:**
- Pulls pre-built image from container registry (ghcr.io/fractalic-ai/fractalic:latest-production)
- Deploys user scripts into running container (no volume mounts)
- Copies config files (settings.toml, mcp_servers.json) into container
- Ensures proper API endpoints and networking
- Fully cloud-ready with proper file isolation

---

## 🔧 Production Deployment Features

### Automatic Port Detection

Fractalic automatically detects available ports to avoid conflicts:

- **Default AI Server Port**: 8001
- **Auto-increment**: If 8001 is occupied, tries 8002, 8003, etc.
- **Docker Container Detection**: Checks for existing Docker containers using the same ports
- **System Port Availability**: Verifies ports are not in use by other processes

Example output:
```bash
⚠️ Port 8001 occupied by container: fractalic-old
🏗️ Using alternative port: 8003
✅ Fractalic AI Server deployed successfully!

🌐 AI Server: http://localhost:8003
📝 Execute script: curl -X POST http://localhost:8003/execute -H "Content-Type: application/json" -d '{"filename": "/payload/script.md", "parameter_text": "optional context"}'
```

### Production vs Full Mode

| Feature | Production Mode | Full Mode |
|---------|----------------|-----------|
| **Size** | ~800MB | ~1.2GB |
| **Startup Time** | ~15 seconds | ~30 seconds |
| **External Ports** | AI Server only (8001+) | All services (3000, 8000, 8001, 5859) |
| **UI Access** | None (API only) | Full web interface |
| **Use Case** | Production/Cloud/API | Development/Testing |
| **Security** | Minimal attack surface | Full development environment |

### Container Images

```bash
# Production images (recommended for deployment)
ghcr.io/fractalic-ai/fractalic:latest-production  # Latest production build
ghcr.io/fractalic-ai/fractalic:production         # Production tag

# Full images (for development)
ghcr.io/fractalic-ai/fractalic:latest             # Latest full build  
ghcr.io/fractalic-ai/fractalic:full               # Full tag
```

### Timezone Configuration

By default, Fractalic Docker containers use UTC timezone. For local development, the deployment script automatically passes through your system timezone. For production deployments, you can configure timezone using environment variables:

```bash
# Run with custom timezone
docker run -e TZ=America/New_York -p 8001:8001 ghcr.io/fractalic-ai/fractalic:latest-production

# For docker-compose deployments
version: '3.8'
services:
  fractalic:
    image: ghcr.io/fractalic-ai/fractalic:latest-production
    environment:
      - TZ=Europe/London
    ports:
      - "8001:8001"
```

**Note**: Git branch timestamps in session names will reflect the configured timezone. Local development automatically uses your system timezone, while production deployments default to UTC unless explicitly configured.

---

## 📁 Expected Directory Structure

### For Existing Installation Deployment

```
your-workspace/
├── fractalic/              (your main fractalic repo with custom content)
├── fractalic-ui/           (UI repo - automatically detected)
└── my-custom-tutorials/    (optional additional content)
```

### For Cloud-Ready Registry Deployment

```
your-workspace/
├── fractalic/              (deployment tools and config)
│   ├── settings.toml       (LLM provider settings - automatically copied)
│   ├── mcp_servers.json    (MCP server configurations - automatically copied)
│   └── publisher_cli.py    (deployment command)
└── your-script-directory/  (script files to deploy)
    ├── your_script.md
    └── any_other_files.py
```

### Prerequisites

#### Standard Deployment
- Existing Fractalic installation with custom content
- fractalic-ui repository at `../fractalic-ui` (one level up)
- Docker Desktop installed and running
- Python 3.11+ with required dependencies

#### Cloud-Ready Registry Deployment
- Docker Desktop installed and running
- Internet connection (to pull registry image)
- Python 3.11+ with publisher dependencies
- settings.toml and mcp_servers.json in project root
- Script files to deploy

---

## ☁️ Cloud-Ready Registry Deployment Guide

### Overview

The registry deployment system provides a production-ready way to deploy user scripts to pre-built Fractalic containers without building images locally. This approach is ideal for:

- **Cloud deployments** (Railway, DigitalOcean, AWS, etc.)
- **CI/CD pipelines** where you don't want to build images
- **Rapid deployments** of user scripts
- **Multi-tenant environments** where users deploy scripts to shared infrastructure

### Step-by-Step Cloud Deployment

#### 1. Prepare Your Configuration

Ensure you have proper configuration files in your fractalic directory:

```bash
# Check required config files exist
ls -la fractalic/settings.toml fractalic/mcp_servers.json

# Example settings.toml structure
cat fractalic/settings.toml
```

#### 2. Deploy Using Registry Plugin

```bash
# Basic deployment
python publisher_cli.py deploy-docker-registry \
  --script-name my-script \
  --script-folder path/to/your/script \
  --container-name fractalic-production

# With custom registry and ports
python publisher_cli.py deploy-docker-registry \
  --script-name my-script \
  --script-folder path/to/your/script \
  --container-name fractalic-production \
  --registry-image ghcr.io/fractalic-ai/fractalic:latest \
  --ports frontend=3000 backend=8000 ai_server=8001 mcp_manager=5859
```

#### 3. Verify Deployment

```bash
# Check container is running
docker ps | grep fractalic-production

# Check services are healthy
curl http://localhost:8000/health
curl http://localhost:8001/health

# Check MCP servers are available (through frontend proxy)
curl http://localhost:3000/mcp/status

# Check settings are loaded
curl http://localhost:8000/load_settings/
```

#### 4. Test Script Execution

```bash
# Execute your deployed script
curl -X POST http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -d '{"filename": "/payload/my-script/your_script.md"}' | jq
```

### Registry Deployment Architecture

The cloud-ready deployment system works as follows:

1. **Image Pull**: Pulls pre-built image from container registry
2. **Container Start**: Starts container with proper port mappings
3. **File Deployment**: Copies user scripts to `/payload/{script-name}/` in container
4. **Config Deployment**: Copies settings.toml and mcp_servers.json to `/fractalic/` and `/` (for backend compatibility)
5. **Frontend Fix**: Updates frontend config.json with correct API endpoints
6. **Next.js Fix**: Updates Next.js config with proper API rewrites
7. **Service Restart**: Restarts container to apply configuration changes

### File Layout in Container

After successful deployment, files are organized as:

```
/fractalic/                 (main application directory)
├── settings.toml           (LLM provider settings)
├── mcp_servers.json        (MCP server configurations)
└── payload/                (symlink to /payload for UI visibility)

/payload/                   (user script deployment directory)
└── {script-name}/          (your deployed script directory)
    ├── your_script.md
    └── other_files.py

/settings.toml              (settings copy for backend compatibility)

/fractalic-ui/              (Next.js frontend)
├── next.config.mjs         (updated with API rewrites)
└── public/
    └── config.json         (updated with correct API endpoints)
```

---

## 🛠 Publishing System Architecture

The publishing system provides a clean way to publish your working Fractalic repository to Docker without polluting your development environment.

### Core Components

1. **`publish_docker.py`** - Core Python script handling the entire publishing process
2. **`publish_docker.sh`** - Simple bash wrapper for command-line usage
3. **`publish_api.py`** - FastAPI server providing HTTP endpoints for UI integration
4. **`ui_integration_example.py`** - Example code for UI integration

### Key Features

- ✅ **Clean Publishing**: Uses temporary directories, doesn't modify source repo
- ✅ **Auto-Detection**: Automatically finds and includes adjacent `fractalic-ui` repository
- ✅ **Port Management**: Supports custom port offsets to avoid conflicts
- ✅ **Service Management**: Includes all Fractalic services (UI, Backend, AI Server, MCP Manager)
- ✅ **Container Management**: Start, stop, and manage published containers
- ✅ **UI Integration**: API endpoints ready for frontend integration

---

## 🚀 Quick Usage

### Command Line Publishing

```bash
# Publish with default settings
./publish_docker.sh

# Publish with custom container name
./publish_docker.sh my-fractalic-container

# Publish with port offset (useful for multiple deployments)
./publish_docker.sh my-container 100
```

### Python Script Publishing

```bash
# Basic publish
python publish_docker.py

# Custom container name and port offset
python publish_docker.py --name my-container --port-offset 100

# Keep temporary files for debugging
python publish_docker.py --keep-temp
```

### API Publishing

```bash
# Start the publishing API server
python publish_api.py

# Then use HTTP endpoints:
# POST /publish - Start publishing
# GET /status/{container_name} - Check status
# POST /stop/{container_name} - Stop container
```

---

## 📋 Step-by-Step Deployment Process

### Step 1: Project Structure Verification

First, verify that your project structure is correct:

```bash
# Check tutorials directory exists
ls -la /path/to/fractalic/tutorials/

# Expected output:
# 01_Basics/
# 02_tutorial_yahoofinance_tavily_stocks_news_analytics/

# Check hello world tutorial
ls -la /path/to/fractalic/tutorials/01_Basics/hello-world/

# Expected output:
# .git/
# hello_world.md
```

### Step 2: Build and Deploy

#### Using Shell Script (Fresh Install)
```bash
mkdir my-fractalic && cd my-fractalic
curl -s https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docker_build_run.sh | bash
```

#### Using Python Script (Existing Installation)
```bash
cd /path/to/fractalic
python publish_docker.py
```

**Expected Build Output:**
```
✅ Container is running
✅ Backend is available at: http://localhost:8000
✅ AI Server is available at: http://localhost:8001
⚠️ MCP Manager may still be starting at: http://localhost:5859
✅ UI should be available at: http://localhost:3000

📋 All Services Summary:
   • Frontend UI:    http://localhost:3000
   • Backend API:    http://localhost:8000
   • AI Server:      http://localhost:8001
   • MCP Manager:    http://localhost:5859

Setup complete! Container is ready.
```

### Step 3: Verify Container and Content

Check that the container is running:

```bash
docker ps
```

Verify tutorials are present in the container:

```bash
docker exec -it fractalic-app ls -la /app/tutorials/
```

### Step 4: Test AI Server Functionality

Execute the hello world tutorial using the AI server:

```bash
curl -X POST http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -d '{"filename": "tutorials/01_Basics/hello-world/hello_world.md"}' \
  | jq
```

**Expected Success Response:**
```json
{
  "success": true,
  "explicit_return": true,
  "return_content": "# File summary\nHere's a summary of the files and directories...",
  "branch_name": "20250623154202_0338efda_Testing-git-operations",
  "ctx_file": null,
  "output": "Execution completed. Branch: ..., Context: ..."
}
```

---

## 🌐 Service Endpoints & API Access

Once deployed, Fractalic provides multiple service endpoints for different purposes:

### Core Services

| Service | Default Port | Purpose | API Documentation |
|---------|-------------|---------|-------------------|
| **Frontend UI** | 3000 | Web interface for Fractalic | Interactive UI |
| **Backend API** | 8000 | File management, settings, git operations | `/docs` endpoint |
| **AI Server** | 8001 | Script execution, LLM integration | `/docs` endpoint |
| **MCP Manager** | 5859 | MCP server management and tools | JSON API |

### API Endpoint Details

#### Frontend UI (Port 3000)
```bash
# Main application interface
http://localhost:3000

# Configuration (served by Next.js)
http://localhost:3000/config.json

# MCP API (proxied through Next.js rewrites)
http://localhost:3000/mcp/status
http://localhost:3000/mcp/tools
http://localhost:3000/mcp/call_tool

# AI Server API (proxied through Next.js rewrites)  
http://localhost:3000/ai/health
http://localhost:3000/ai/execute

# Backend API (proxied through Next.js rewrites)
http://localhost:3000/list_directory
http://localhost:3000/load_settings
http://localhost:3000/save_settings
```

#### Backend API (Port 8000)
```bash
# API Documentation
http://localhost:8000/docs

# Health Check
http://localhost:8000/health

# File Management
http://localhost:8000/list_directory?path=/payload
http://localhost:8000/get_file_content?path=/payload/script.md
http://localhost:8000/create_file
http://localhost:8000/save_file
http://localhost:8000/delete_item

# Settings Management
http://localhost:8000/load_settings/
http://localhost:8000/save_settings

# Git Operations
http://localhost:8000/branches_and_commits?repo_path=/payload/my-script
```

#### AI Server (Port 8001)
```bash
# API Documentation
http://localhost:8001/docs

# Health Check
http://localhost:8001/health

# Script Execution
POST http://localhost:8001/execute
Content-Type: application/json
{
  "filename": "/payload/my-script/script.md",
  "parameter_text": "optional context"
}

# Provider Information
http://localhost:8001/providers
```

#### MCP Manager (Port 5859)
```bash
# Server Status
http://localhost:5859/status

# Available Tools
http://localhost:5859/tools

# Call Tool
POST http://localhost:5859/call_tool
Content-Type: application/json
{
  "name": "tool_name",
  "arguments": {"arg1": "value1"}
}

# Server Management
POST http://localhost:5859/start/{server_name}
POST http://localhost:5859/stop/{server_name}
POST http://localhost:5859/restart/{server_name}
```

### API Routing in Containers

For cloud-ready deployments, API routing works as follows:

#### Direct Access (from host machine)
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000` 
- AI Server: `http://localhost:8001`
- MCP Manager: `http://localhost:5859`

#### Through Frontend Proxies (recommended for web apps)
- MCP APIs: `http://localhost:3000/mcp/*` → `http://localhost:5859/*`
- AI APIs: `http://localhost:3000/ai/*` → `http://localhost:8001/*`
- Backend APIs: `http://localhost:3000/*` → `http://localhost:8000/*`

#### Internal Container Routing
- All services communicate via `localhost` on their internal ports
- Next.js rewrites handle API proxying for browser requests
- Backend services can call each other directly

### Example API Usage

#### Check Deployment Health
```bash
# Check all services are responding
curl http://localhost:3000/config.json
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:3000/mcp/status
```

#### Execute a Script
```bash
# Execute deployed script via AI server
curl -X POST http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "/payload/my-script/hello_world.md"
  }' | jq
```

#### List Available MCP Servers
```bash
# Get all MCP servers and their status
curl http://localhost:3000/mcp/status | jq 'keys'

# Get tools for all servers
curl http://localhost:3000/mcp/tools | jq
```

#### Load Settings
```bash
# Get current LLM provider settings
curl http://localhost:8000/load_settings/ | jq '.settings.defaultProvider'
```

#### Browse Deployed Files
```bash
# List files in payload directory
curl "http://localhost:8000/list_directory?path=/payload" | jq

# List files in specific script directory
curl "http://localhost:8000/list_directory?path=/payload/my-script" | jq
```

With port offset (e.g., `--port-offset 100`):
- **Frontend UI**: http://localhost:3100
- **Backend API**: http://localhost:8100
- **AI Server**: http://localhost:8101
- **MCP Manager**: http://localhost:5959

---

## 🔧 Advanced Configuration

### Port Management

```bash
# Deploy with custom ports (adds offset to all services)
python publish_docker.py --port-offset 100 --name fractalic-dev

# This creates:
# Frontend: 3100, Backend: 8100, AI Server: 8101, MCP Manager: 5959
```

### Container Management

```bash
# List running containers
docker ps

# View container logs
docker logs fractalic-app

# Stop container
docker stop fractalic-app

# Remove container
docker rm fractalic-app

# Remove image
docker rmi fractalic-app
```

### Multiple Deployments

```bash
# Deploy multiple instances with different ports
python publish_docker.py --name fractalic-dev --port-offset 0
python publish_docker.py --name fractalic-staging --port-offset 100
python publish_docker.py --name fractalic-prod --port-offset 200
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Container fails to start
```bash
# Check Docker Desktop is running
docker --version

# Ensure ports are not in use
lsof -i :3000 -i :8000 -i :8001 -i :5859

# Check container logs
docker logs fractalic-app
```

#### 2. Services not responding
```bash
# Wait 30-60 seconds for services to fully start
sleep 30

# Check individual service logs in container
docker exec fractalic-app cat /tmp/backend.out.log
docker exec fractalic-app cat /tmp/frontend.out.log
docker exec fractalic-app cat /tmp/ai_server.out.log
docker exec fractalic-app cat /tmp/mcp_manager.err.log

# Verify firewall/network settings
curl -v http://localhost:8000/health
```

#### 3. MCP Servers not visible in UI
```bash
# Check MCP manager is running
curl http://localhost:5859/status

# Check MCP API through frontend proxy
curl http://localhost:3000/mcp/status

# Check frontend config has correct endpoints
curl http://localhost:3000/config.json | jq '.api'

# Expected output:
# {
#   "backend": "",
#   "ai_server": "/ai", 
#   "mcp_manager": "/mcp"
# }
```

#### 4. Settings not loading
```bash
# Check settings file exists in container
docker exec fractalic-app ls -la /settings.toml
docker exec fractalic-app ls -la /fractalic/settings.toml

# Test settings endpoint
curl http://localhost:8000/load_settings/ | jq '.settings'

# Should return settings object, not null
```

#### 5. Script execution fails
```bash
# Check script was deployed correctly
docker exec fractalic-app ls -la /payload/
docker exec fractalic-app ls -la /payload/your-script-name/

# Check payload symlink exists
docker exec fractalic-app ls -la /fractalic/payload

# Test execution endpoint
curl -X POST http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -d '{"filename": "/payload/your-script-name/script.md"}' | jq
```

#### 6. fractalic-ui not found (Legacy deployments)
```bash
# Ensure fractalic-ui is at ../fractalic-ui relative to fractalic directory
ls -la ../fractalic-ui

# Clone it manually if missing
git clone https://github.com/fractalic-ai/fractalic-ui.git ../fractalic-ui
```

#### 7. Registry image pull failures
```bash
# Check internet connection
ping github.com

# Try pulling image manually
docker pull ghcr.io/fractalic-ai/fractalic:latest

# Check Docker registry access
docker login ghcr.io
```

### Debug Mode

#### For Registry Deployments
```bash
# Check container status
docker ps -a | grep fractalic

# Inspect container configuration
docker inspect fractalic-container-name

# Check container logs
docker logs fractalic-container-name

# Enter container for inspection
docker exec -it fractalic-container-name /bin/bash

# Inside container, check file structure
ls -la /payload/
ls -la /fractalic/
cat /fractalic-ui/next.config.mjs
cat /fractalic-ui/public/config.json
```

#### For Legacy Deployments
```bash
# Keep temporary files for inspection
python publish_docker.py --keep-temp

# Check container logs
docker logs fractalic-published

# Check running containers
docker ps

# Check all containers (including stopped)
docker ps -a

# Inspect container filesystem
docker exec -it fractalic-app bash
```

### Performance Issues

#### 1. Slow container startup
```bash
# Check available system resources
docker system df
docker system info

# Monitor resource usage
docker stats fractalic-container-name

# Check for port conflicts
netstat -tulpn | grep :3000
netstat -tulpn | grep :8000
```

#### 2. High memory usage
```bash
# Check MCP servers status
curl http://localhost:3000/mcp/status | jq 'to_entries[] | select(.value.healthy == false)'

# Restart container if needed
docker restart fractalic-container-name
```

### Configuration Issues

#### 1. Wrong API endpoints in frontend
```bash
# Check current frontend config
curl http://localhost:3000/config.json | jq

# Should show relative paths for cloud deployment:
# {
#   "api": {
#     "backend": "",
#     "ai_server": "/ai",
#     "mcp_manager": "/mcp"
#   }
# }

# If wrong, redeploy to fix
python publisher_cli.py deploy-docker-registry --script-name your-script --script-folder your-folder --container-name your-container
```

#### 2. Missing Next.js rewrites
```bash
# Check Next.js config in container
docker exec fractalic-container cat /fractalic-ui/next.config.mjs | grep -A 10 "rewrites"

# Should contain:
# source: '/mcp/:path*',
# destination: 'http://localhost:5859/:path*'
```

### Tutorials Missing

If tutorials are missing:
1. Verify the tutorials directory exists in source
2. Check Dockerfile includes `COPY fractalic/tutorials/ /app/tutorials/`
3. Rebuild the image
4. Verify with: `docker exec -it fractalic-app ls -la /app/tutorials/`

---

## 🔌 UI Integration

To integrate the publishing system with the main Fractalic UI:

### 1. Start Publishing API

```bash
# Start the API server alongside main UI
python publish_api.py
```

### 2. Add UI Components

```javascript
// Example: Add publish button to UI
const publishToDocker = async () => {
  const response = await fetch('/api/publish', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      container_name: 'my-fractalic',
      port_offset: 0
    })
  });
  
  const result = await response.json();
  console.log('Published:', result);
};
```

### 3. API Endpoints

- `POST /publish` - Start publishing process
- `GET /status/{container_name}` - Check container status
- `POST /stop/{container_name}` - Stop container
- `GET /containers` - List all published containers

The system is designed to be non-intrusive and can be easily integrated into the existing Fractalic UI workflow.

---

## 📊 What Deployment Demonstrates

A successful deployment confirms:

1. **Successful Build**: Docker image built with all components
2. **Content Inclusion**: Tutorials and custom content properly copied
3. **Service Integration**: All services (UI, Backend, AI Server, MCP Manager) running
4. **AI Server Functionality**: Can execute tutorial scripts via API
5. **Network Configuration**: Proper port mapping and service communication
6. **Tutorial Execution**: Demonstrates:
   - Shell command execution (`ls -la`)
   - LLM integration for analysis
   - Formatted output generation
   - Git branch tracking for operations

---

## 🔒 Key Files and Safety

### Modified Files
1. **docker/Dockerfile**: Updated with proper build context paths
2. **docker/supervisord.conf**: Fixed working directories and command syntax
3. **docker_build_run.sh**: Enhanced with safety checks for fresh installs
4. **publish_docker.py**: Enhanced with safety checks for existing installations

### Safety Features
- **Empty Directory Check**: Shell script only runs in empty directories
- **Path Validation**: Python script validates all paths to prevent repo pollution
- **Temporary Directories**: All builds use temporary directories, never modify source
- **Auto Cleanup**: Automatic cleanup of temporary files after build
- **.gitignore Protection**: Prevents accidental commit of build artifacts

---

## 🎯 Summary

The Fractalic Docker deployment system provides:

### Deployment Options
- **Fresh installs**: GitHub-based automatic setup for new users
- **Custom deployments**: Existing installation deployment with customizations  
- **Cloud-ready registry**: Production deployments using pre-built images

### Key Features
- **Safety-first approach**: Never pollutes source repositories
- **Flexible configuration**: Custom ports, names, and content inclusion
- **Complete service stack**: UI, Backend, AI Server, and MCP Manager
- **Cloud-ready architecture**: No volume mounts, proper file isolation
- **Automatic API routing**: Next.js rewrites and frontend configuration
- **Multi-service coordination**: All services properly networked and configured

### Service Architecture
- **Frontend (3000)**: Next.js UI with API rewrites for seamless integration
- **Backend (8000)**: FastAPI server for file management and settings
- **AI Server (8001)**: Script execution and LLM integration
- **MCP Manager (5859)**: Model Context Protocol server management

### API Access Patterns
- **Direct access**: Each service on its own port
- **Proxied access**: All APIs available through frontend rewrites
- **Container networking**: Internal service-to-service communication
- **Browser compatibility**: CORS-free API access through rewrites

### Configuration Management
- **Automatic config fixing**: Frontend config.json updated with correct endpoints
- **Next.js rewrite injection**: API routing automatically configured
- **Settings deployment**: LLM provider settings properly copied and accessible
- **MCP server integration**: All MCP servers visible and functional in UI

### Developer Experience
- **Developer-friendly**: Easy debugging and troubleshooting
- **UI integration ready**: API endpoints for frontend integration
- **Container inspection**: Full access to container filesystem for debugging
- **Comprehensive logging**: All services log to accessible locations

### Production Ready
- **Registry-based deployment**: Fast, consistent deployments from pre-built images
- **No build requirements**: Deploy without Docker build context
- **Isolated file system**: User scripts deployed without affecting base image
- **Service health monitoring**: Comprehensive health checks for all services
- **Automatic restarts**: Container restarts to apply configuration changes

Whether you're a new user trying Fractalic, a developer deploying custom content, or running production deployments in the cloud, the system provides a clean, safe, and powerful way to run Fractalic in Docker containers with full service integration and API accessibility.
