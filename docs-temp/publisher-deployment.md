---
title: Publisher & Deployment
description: Deploy Fractalic workflows as remote AI services for sharing and production use
outline: deep
---

# Publisher & Deployment

> Audience: Contributors and users who want to deploy their Fractalic workflows as remote AI services. Regular users running local workflows may not need this until they want to share their AI agents with others.

## What is Deployment?

Deployment transforms your local Fractalic workflow (a Markdown file with AI operations) into a remote web service that others can use. Think of it as packaging your AI agent and making it available on the internet.

**Example:** You create a workflow called `summarizer.md` that takes long text and returns a summary. After deployment, anyone can send text to your service via HTTP and get back a summary - without needing to install Fractalic themselves.

## The Production Container

When you deploy, Fractalic uses a special lightweight Docker container built specifically for running AI workflows:

### Container Contents
- **Base Image**: `ghcr.io/fractalic-ai/fractalic:latest-production`
- **Size**: Minimal - only what's needed to run workflows
- **What's Inside**:
  - Python 3.11 runtime
  - Fractalic execution engine
  - AI Server (port 8001) - the main service
  - Backend Server (port 8000) - internal management
  - MCP Manager (port 5859) - tool integration
  - Supervisor process manager

### What's NOT Included
- Web UI components (keeps the container small)
- Development tools
- Node.js/frontend dependencies

### Container Structure
```
/fractalic/          # Core Fractalic system
  ai_server/         # Main AI service code
  core/             # Backend management
  settings.toml     # Your LLM provider settings
  mcp_servers.json  # Your tool configurations

/payload/           # Your deployed scripts go here
  your-workflow/    # Each deployment gets its own folder
    your-script.md  # Your actual workflow file
    data/          # Any supporting files you include
```

## How Deployment Works

### Step 1: Preparation
1. You specify which local folder contains your workflow
2. Deployment system finds your main script file (`.md`, `.py`, etc.)
3. Copies your files to a temporary staging area
4. Locates and copies your configuration files (`settings.toml`, `mcp_servers.json`)

### Step 2: Container Launch
1. Pulls the production container image
2. Creates a new container with a unique name
3. Mounts your files at `/payload/your-workflow/`
4. Copies configuration to both `/fractalic/` and `/` (for compatibility)
5. Starts three services using Supervisor:
   - **AI Server** (port 8001+) - your main API endpoint
   - **Backend Server** (port 8000) - internal management
   - **MCP Manager** (port 5859) - tool integrations

### Step 3: Service Startup
1. AI Server scans for available port (starts at 8001, increments if busy)
2. Loads your `settings.toml` for LLM provider access
3. Starts MCP Manager if you have tool configurations
4. Reports ready via health check endpoint

### Step 4: Ready to Use
- Your workflow is now accessible via HTTP REST API
- Health check: `http://localhost:8001/health`
- Execute endpoint: `http://localhost:8001/execute`
- API documentation: `http://localhost:8001/docs`

## Your Scripts & Files

### What Gets Deployed
The deployment system copies your specified folder with intelligent filtering:

**Included by Default:**
- `.md` files (your Fractalic workflows)
- `.py` files (custom Python scripts)
- `.txt`, `.json`, `.yaml` files (data/config)
- Supporting folders and subdirectories

**Automatically Excluded:**
- `.git/` (version control)
- `__pycache__/` (Python cache)
- `node_modules/` (development dependencies)
- `.DS_Store` (macOS system files)
- `*.log` files

### File Organization in Container
```
/payload/your-workflow-name/
├── your-main-script.md      # Your primary workflow
├── data/                    # Supporting data files
│   ├── examples.json
│   └── templates/
├── helpers/                 # Additional scripts
│   └── utilities.py
└── README.md               # Documentation
```

### Configuration Files
These are copied to `/fractalic/` for the system to use:

- **`settings.toml`** - LLM provider settings (OpenAI, Anthropic keys, etc.)
- **`mcp_servers.json`** - Tool integration configurations
- **`.env`** - Environment variables (if present)
- **`requirements.txt`** - Additional Python dependencies (if needed)

## The AI Server

The AI Server is the main service that runs your workflows. It provides a REST API that accepts HTTP requests and returns results.

### Main Endpoint: `/execute`
This is how external users interact with your deployed workflow:

**Request Format:**
```json
{
  "filename": "payload/your-workflow/your-script.md",
  "parameter_text": "Optional input parameters"
}
```

**What Happens When Called:**
1. AI Server receives the HTTP request
2. Validates the file exists in `/payload/`
3. Sets working directory to `/fractalic/` (where configs are)
4. If `parameter_text` provided, creates temporary parameter file
5. Calls `run_fractalic()` with your script and parameters
6. Executes your workflow (all `@llm`, `@shell`, `@return` operations)
7. Returns structured result

**Response Format:**
```json
{
  "success": true,
  "explicit_return": true,
  "return_content": "Your workflow's @return output",
  "branch_name": "git-branch-created",
  "output": "Full execution log",
  "ctx_file": null
}
```

### Processing @return Statements
When your workflow includes an `@return` operation:
1. The content of that block becomes `return_content`
2. `explicit_return` is set to `true`
3. External callers receive this as the main result
4. This is how you provide structured output to API users

### Health Check: `/health`
Simple endpoint that returns `{"status": "healthy"}` when the service is running.

### API Documentation: `/docs`
FastAPI automatically generates interactive Swagger documentation available at:
`http://localhost:8001/docs`

**In your browser, you can:**
- See all available endpoints
- Test API calls interactively
- View request/response schemas
- Try example requests

## Quick Start Examples

### Deploy via UI Server
1. Start Fractalic UI: `./run_server.sh`
2. Open browser to: `http://localhost:3000`
3. Navigate to deployment section
4. Fill in:
   - **Script Name**: `my-agent`
   - **Script Folder**: `./workflows`
   - **Container Name**: `my-agent` (optional)
5. Click "Deploy" and watch progress

### Deploy via Command Line
```bash
python publisher_cli.py deploy docker-registry \
  --name my-agent \
  --script-name my-agent \
  --script-folder workflows
```

## Using the Web API

### Test Your Deployed Service
```bash
# Check if service is running
curl http://localhost:8001/health

# Execute your workflow
curl -X POST http://localhost:8001/execute \
  -H 'Content-Type: application/json' \
  -d '{
    "filename": "payload/my-agent/my-agent.md",
    "parameter_text": "Summarize this: Your input text here"
  }'
```

### Example Workflow File
```markdown
# Text Summarizer {id=main}

@llm
prompt: |
  Please summarize the following text in 2-3 sentences:
  
  {{input-parameters}}
to: summary

# Summary Result {id=summary}

@return
blocks: summary
```

### API Response
```json
{
  "success": true,
  "explicit_return": true,
  "return_content": "Here is a 2-3 sentence summary of the input text...",
  "branch_name": "workflow-execution-20241208-143022",
  "output": "Full execution log with all operations..."
}
```

### Interactive API Explorer
Visit `http://localhost:8001/docs` in your browser to:
- See all endpoints visually
- Try API calls with a web interface
- View example requests and responses
- Test different parameter combinations

## Configuration Files

### settings.toml
Contains your LLM provider configurations:
```toml
[anthropic]
api_key = "your-key-here"
model = "claude-3-5-sonnet-20241022"

[openai]
api_key = "your-openai-key"
model = "gpt-4"

defaultProvider = "anthropic"
```

### mcp_servers.json
Defines which tools your workflows can use:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"]
    }
  }
}
```

**Important**: These configuration files are automatically copied from your project root during deployment. OAuth files like `oauth_redirect_state.json` and `oauth_tokens.json` are NOT automatically copied and would need manual setup if required for specific MCP tools.

## Troubleshooting Common Issues

### "No model specified" Error
**Problem**: The AI server can't find your LLM provider settings.
**Solution**: 
1. Ensure `settings.toml` exists in your project root
2. Check that it contains valid provider configuration
3. Restart deployment

### Port Conflicts
**Problem**: "Address already in use" when starting.
**Solution**: The system automatically tries ports 8001, 8002, 8003, etc. Wait for auto-resolution or stop conflicting services.

### Missing Tools
**Problem**: Your workflow can't access external tools.
**Solution**:
1. Check that `mcp_servers.json` exists in your project root
2. Verify tool configurations are correct
3. Ensure MCP Manager started successfully

### Script Not Found
**Problem**: "File not found" when calling `/execute`.
**Solution**:
1. Check the exact filename in your request
2. Ensure path starts with `payload/your-container-name/`
3. Verify file was included in deployment (not excluded by filters)

### Container Won't Start
**Problem**: Deployment says success but service isn't responding.
**Solution**:
```bash
# Check container status
docker ps

# View container logs
docker logs your-container-name

# Check internal services
docker exec -it your-container-name curl http://localhost:8001/health
```

## What Runs in Production

When your container starts, these services run automatically:

1. **AI Server** (port 8001+) - Executes your workflows
2. **UI Server** (port 8000) - Provides web interface and API
3. **MCP Manager** - Automatically starts and manages external tools

All services are managed by Supervisor and restart automatically if they crash.

## Cross References
- [AI Server & API](ui-server-api.md) - Understanding the UI server that triggers deployments
- [Configuration](configuration.md) - Setting up `settings.toml` and provider keys
- [MCP Integration](mcp-integration.md) - Configuring external tools
- [Advanced LLM Features](advanced-llm-features.md) - Workflow capabilities
- [Syntax Reference](syntax-reference.md) - Writing effective workflows
