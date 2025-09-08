---
title: UI Server & API
description: Internal UI server for development and debugging (not for production use)
outline: deep
---

# UI Server & API

> Disclaimer: This UI Server is an internal development component. Regular users running Fractalic workflows (via the AI Server or production deployments) do not need it. This document is intended for contributors, advanced operators, or anyone studying Fractalic's internal architecture. Do not expose this server publicly.

## Internal TOC
- [What is the UI Server?](#what-is-the-ui-server)
- [When Do You Use the UI Server?](#when-do-you-use-the-ui-server)
- [What Does the UI Server Do?](#what-does-the-ui-server-do)
- [Core Concepts](#core-concepts)
- [How It Works](#how-it-works)
- [Starting the UI Server](#starting-the-ui-server)
- [Key Endpoints by Function](#key-endpoints-by-function)
- [Common Usage Examples](#common-usage-examples)
- [Quick Reference](#quick-reference)
- [Cross References](#cross-references)

## What is the UI Server?

The UI Server is a **local development backend** designed specifically to power Fractalic's interactive web IDE. If you're new to Fractalic, think of it as the "behind-the-scenes engine" that makes the web interface work smoothly when you're building and testing AI workflows on your computer.

**Important:** This server is **not** the main Fractalic execution API. The core Fractalic functionality (running AI workflows remotely) is provided by the separate **AI Server** which runs on port 8001+. The UI Server handles local development tasks like file browsing, git history, and managing development tools.

## When Do You Use the UI Server?

The UI Server is automatically started in these scenarios:

1. **Local Development**: When you run `./run_server.sh` to start the full Fractalic development environment
2. **Full Docker Container**: When using the main `fractalic:latest` Docker image (includes both UI and backend)
3. **Not Used**: When deploying the production-only container (`fractalic:latest-production`) - this lightweight version only includes the AI Server

If you're just getting started with Fractalic, you'll typically use option 1 or 2, which automatically starts the UI Server for you.

## What Does the UI Server Do?

The UI Server is a FastAPI service that handles six main areas:

1. **File & Project Management** - Browse folders, edit files, manage your Fractalic project
2. **Git History Visualization** - View execution traces and workflow evolution over time  
3. **MCP Tool Integration** - Start/stop and monitor Model Context Protocol services
4. **Live Execution Streaming** - Watch Fractalic workflows run in real-time
5. **Development Tools** - Get tool schemas, manage settings, debug issues
6. **Deployment Helpers** - Publish your workflows to production environments

This guide covers concepts, flows, endpoints, usage patterns, optimization, and safety constraints.

## Core Concepts

| Concept | What | Why It Matters | For Newcomers |
| ------- | ---- | -------------- | ------------- |
| UI Server vs AI Server | UI Server (port 8000) handles development tools; AI Server (port 8001+) executes workflows | Separation of concerns: local dev vs remote execution | Use UI Server for building; AI Server for running |
| Local Development Mode | Full environment with file browser, git visualization, live streaming | Rapid iteration and debugging | Start with `./run_server.sh` |
| MCP Manager | Background process that connects external tools (weather, databases, etc.) | Extends Fractalic with real-world integrations | Think of it as a "plugin system" |
| Streaming Execution | Real-time output as your Fractalic workflows run | See progress, catch errors immediately | Like watching a script run in terminal |
| Git-Based History | Every workflow execution creates git commits for traceability | Rollback, compare versions, understand evolution | Your workflow changes are saved automatically |
| Tool Schema Discovery | Automatically finds available tools and their parameters | Know what tools you can use without reading docs | Like auto-complete for tools |
| One-Click Deployment | Push your local workflow to cloud hosting | Share your AI agents with others easily | From development to production in one step |

## How It Works

The UI Server operates as a local development companion that bridges your file system, git repository, and Fractalic execution environment:

1. **Bootstrap Phase**: FastAPI app starts with CORS enabled for local web UI access
2. **Process Management**: Can spawn and monitor the MCP Manager subprocess for tool integrations
3. **File System Bridge**: Provides safe access to browse, read, create, and modify project files
4. **Git Integration**: Uses GitPython to read commit history and construct execution lineage trees
5. **Streaming Interface**: Captures subprocess output (shell commands, Fractalic runs) in UTF-8 safe chunks for real-time display
6. **Tool Discovery**: Scans your tools directory and generates schema for dynamic UI construction
7. **Deployment Pipeline**: Integrates with publisher plugins to push workflows to production

**Technical Note**: The server includes readiness probes, trace logging (when enabled), and handles both managed and externally-started MCP processes gracefully.

## Starting the UI Server

### Option 1: Local Development (Recommended for Beginners)
```bash
# Clone the repository and enter the directory
cd fractalic

# Run the setup script (installs dependencies, sets up environment)
./run_server.sh
```
This starts the UI Server on port 8000 and opens the web interface.

### Option 2: Docker Development Environment
```bash
# Start full development container (includes UI)
docker run -p 3000:3000 -p 8000:8000 -p 8001:8001 ghcr.io/fractalic-ai/fractalic:latest
```
Access the UI at `http://localhost:3000`, backend at `http://localhost:8000`.

### Option 3: Manual Start (Advanced Users)
```bash
# Activate virtual environment
source .venv/bin/activate

# Start UI Server directly
uvicorn core.ui_server.server:app --host 0.0.0.0 --port 8000
```

### What's NOT Started
The production-only container (`fractalic:latest-production`) excludes the UI Server entirely. It only includes the AI Server for executing workflows in cloud environments.

## Key Endpoints by Function

### Basic Health & Information
| Endpoint | Method | Purpose | For Newcomers |
| -------- | ------ | ------- | ------------- |
| `/health` | GET | Check if server is running | Use to verify server started correctly |
| `/info` | GET | Get server version and features | See what capabilities are available |

### File & Project Management 
| Endpoint | Method | Purpose | For Newcomers |
| -------- | ------ | ------- | ------------- |
| `/list_directory/` | GET | Browse folders and files | Like a file explorer in the web UI |
| `/get_file_content_disk/` | GET | Read file contents | View/edit your Fractalic scripts |
| `/save_file` | POST | Save file changes | Persist your workflow edits |
| `/create_file/` | POST | Create new files | Start new workflow documents |
| `/create_folder/` | POST | Create directories | Organize your project structure |
| `/delete_item/` | DELETE | Remove files/folders | Clean up unused files |
| `/rename_item/` | POST | Rename files/folders | Keep your project organized |

### Workflow Execution & Monitoring
| Endpoint | Method | Purpose | For Newcomers |
| -------- | ------ | ------- | ------------- |
| `/ws/run_fractalic` | POST | Execute Fractalic document with live output | Run your AI workflows and see results |
| `/ws/run_command` | POST | Run shell commands with streaming | Execute system commands safely |
| `/branches_and_commits/` | GET | View execution history | See how your workflows evolved |
| `/get_enriched_call_tree/` | GET | Get detailed execution trace | Debug workflow execution paths |

### MCP Tool Integration
| Endpoint | Method | Purpose | For Newcomers |
| -------- | ------ | ------- | ------------- |
| `/mcp/start` | POST | Start MCP tool manager | Enable external tool integrations |
| `/mcp/stop` | POST | Stop MCP tool manager | Disable tools when not needed |
| `/mcp/status` | GET | Check MCP tool status | See which tools are available |
| `/tools_schema/` | GET | Get available tool definitions | Know what tools you can use |

### Configuration & Deployment
| Endpoint | Method | Purpose | For Newcomers |
| -------- | ------ | ------- | ------------- |
| `/save_settings/` | POST | Save UI preferences | Persist your development settings |
| `/load_settings/` | GET | Load UI preferences | Restore your saved settings |
| `/api/deploy/docker-registry` | POST | Deploy workflow to cloud | Share your AI agent with others |
| `/api/deploy/docker-registry/stream` | POST | Deploy with progress updates | Watch deployment progress live |

## Common Usage Examples

### Getting Started: Check if Everything is Working
```bash
# 1. Verify the UI Server is running
curl http://localhost:8000/health

# Expected response: {"status": "healthy", "ui_server": "running", ...}

# 2. See what features are available
curl http://localhost:8000/info
```

### Working with Files (Beginner-Friendly)
```bash
# Browse your project directory
curl 'http://localhost:8000/list_directory/?path=/your/project/path'

# Read a Fractalic document
curl 'http://localhost:8000/get_file_content_disk/?path=/path/to/workflow.md'

# Create a new workflow file
curl -X POST 'http://localhost:8000/create_file/?path=/your/project&name=my-workflow.md'
```

### Running Your First Fractalic Workflow
```bash
# Execute a Fractalic document and watch the output
curl -N -X POST http://localhost:8000/ws/run_fractalic \
  -H 'Content-Type: application/json' \
  -d '{"file_path": "examples/hello-world.md"}'
```
The `-N` flag prevents curl from buffering output, so you see results in real-time.

### Setting Up Tool Integrations
```bash
# Start the MCP manager to enable external tools
curl -X POST http://localhost:8000/mcp/start

# Check if tools are available
curl http://localhost:8000/mcp/status

# See what tools you can use in your workflows
curl 'http://localhost:8000/tools_schema/?tools_dir=tools'
```

### Exploring Workflow History (Advanced)
```bash
# See your workflow execution history
curl 'http://localhost:8000/branches_and_commits/?repo_path=/your/project'

# Get detailed execution trace for debugging
curl 'http://localhost:8000/get_enriched_call_tree/?repo_path=/your/project&branch=my-experiment'
```

### Deploying Your Workflow to Production
```bash
# Deploy with progress monitoring
curl -N -X POST http://localhost:8000/api/deploy/docker-registry/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "script_name": "my-agent", 
    "script_folder": "./workflows/",
    "container_name": "my-ai-agent"
  }'
```

## Quick Reference

### Server Startup Methods
```bash
# Method 1: Development script (recommended for beginners)
./run_server.sh

# Method 2: Docker development environment  
docker run -p 3000:3000 -p 8000:8000 -p 8001:8001 ghcr.io/fractalic-ai/fractalic:latest

# Method 3: Manual start (advanced)
uvicorn core.ui_server.server:app --host 0.0.0.0 --port 8000
```

### Essential Endpoints Checklist
- ✅ Health Check: `GET /health`
- ✅ Start Tools: `POST /mcp/start` 
- ✅ Run Workflow: `POST /ws/run_fractalic` (with `{"file_path": "path/to/workflow.md"}`)
- ✅ Browse Files: `GET /list_directory/?path=/your/path`
- ✅ Tool Schema: `GET /tools_schema/`
- ✅ Deploy: `POST /api/deploy/docker-registry/stream`

### Port Reference
- **UI Server**: 8000 (local development backend)
- **Web Interface**: 3000 (React frontend, if using Docker)
- **AI Server**: 8001+ (remote workflow execution API)
- **MCP Manager**: 5859 (tool integration service)

### Container Types
- `fractalic:latest` - Full development environment (includes UI Server)
- `fractalic:latest-production` - AI Server only (no UI Server)

## Cross References
See also:
- [Syntax Reference](syntax-reference.md)
- [Operations Reference](operations-reference.md)
- [Advanced LLM Features](advanced-llm-features.md)
- [Context Management](context-management.md)
- [MCP Integration](mcp-integration.md)
- [Configuration](configuration.md)
