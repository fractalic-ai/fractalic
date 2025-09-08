# Use the official Python slim image for a smaller footprint
FROM python:3.11-slim

# Install system dependencies, including git and Docker CLI
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gnupg \
    supervisor \
    git \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    tzdata && \
    # Install Docker CLI using architecture-specific static binary
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        DOCKER_ARCH="x86_64"; \
    elif [ "$ARCH" = "arm64" ]; then \
        DOCKER_ARCH="aarch64"; \
    else \
        DOCKER_ARCH="x86_64"; \
    fi && \
    curl -fsSL https://download.docker.com/linux/static/stable/$DOCKER_ARCH/docker-24.0.7.tgz -o docker.tgz && \
    tar xzf docker.tgz && \
    mv docker/docker /usr/local/bin/ && \
    rm -rf docker docker.tgz && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js (version 20) and npm
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy backend code to /fractalic directory
COPY fractalic/ /fractalic/

# Create settings.toml from sample for development (includes pre-filled MCP servers)
RUN cp /fractalic/settings.toml.sample /fractalic/settings.toml

# Create mcp_servers.json from sample for development (includes example MCP servers)
RUN cp /fractalic/mcp_servers.json.sample /fractalic/mcp_servers.json

# Fix MCP manager binding for Docker deployment (bind to all interfaces)
RUN sed -i 's/127.0.0.1/0.0.0.0/g' /fractalic/fractalic_mcp_manager.py

# Update settings.toml MCP server URL for Docker container networking
RUN sed -i 's/127.0.0.1:5859/0.0.0.0:5859/g' /fractalic/settings.toml

# Copy and install backend dependencies
RUN pip install --no-cache-dir -r /fractalic/requirements.txt

# Copy and install frontend dependencies (fractalic-ui at same level as fractalic)
COPY fractalic-ui/package*.json /fractalic-ui/
RUN mkdir -p /fractalic-ui && \
    cd /fractalic-ui && \
    npm config set registry https://registry.npmjs.org/ && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm config set fetch-retries 5 && \
    (npm install --network-timeout=300000 || \
     echo "npm install failed - will try to run with existing node_modules or minimal setup")

# Copy the frontend code to same level as fractalic
COPY fractalic-ui/ /fractalic-ui/

# Replace next.config.mjs with Docker-specific version for container networking
# This MUST come after copying fractalic-ui/ to ensure it doesn't get overwritten
COPY fractalic/docker/next.config.docker.mjs /fractalic-ui/next.config.mjs

# Set working directory to fractalic for backend operations
WORKDIR /fractalic

# Skip npm run build - we use npm run dev in supervisord anyway
# This makes Docker builds much faster since production build is unused

# Create necessary directories with full permissions (development environment)
RUN mkdir -p /fractalic/logs /fractalic/oauth-cache /tmp && \
    chmod -R 777 /fractalic /fractalic-ui /tmp

# Set timezone environment variable support (can be overridden at runtime)
ENV TZ=UTC

# Create initial mcp_servers.json if it doesn't exist (from sample)
RUN if [ ! -f /fractalic/mcp_servers.json ] && [ -f /fractalic/mcp_servers.json.sample ]; then \
        cp /fractalic/mcp_servers.json.sample /fractalic/mcp_servers.json; \
        echo "📋 Created mcp_servers.json from sample"; \
    elif [ ! -f /fractalic/mcp_servers.json ]; then \
        echo '{"mcpServers": {}}' > /fractalic/mcp_servers.json; \
        echo "📋 Created empty mcp_servers.json"; \
    fi

# Copy Supervisor configuration
COPY fractalic/docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set PYTHONPATH environment variable to fractalic directory
ENV PYTHONPATH=/fractalic

# Stay as root user for development - full permissions to everything

# Create an enhanced entrypoint script that shows port information
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Fractalic Development Services..."\n\
echo "📍 Working directory: $(pwd)"\n\
echo "👤 User: $(whoami) (UID: $(id -u))"\n\
echo "🔓 Running with full root privileges (development mode)"\n\
echo ""\n\
echo "🐳 Docker CLI Status:"\n\
if command -v docker >/dev/null 2>&1; then\n\
    echo "  ✅ Docker CLI available: $(docker --version)"\n\
else\n\
    echo "  ❌ Docker CLI not found in PATH"\n\
    echo "  PATH: $PATH"\n\
fi\n\
echo ""\n\
echo "🌐 Service Ports:"\n\
echo "  - Frontend (UI):     http://localhost:3000"\n\
echo "  - Backend API:       http://localhost:8000"\n\
echo "  - AI Server:         http://localhost:8001"\n\
echo "  - MCP Manager:       http://localhost:5859 (Start via UI)"\n\
echo "  - Additional:        8002, 8003, 8004"\n\
echo ""\n\
echo "📝 Logs will be in /tmp/"\n\
echo "🎛️ MCP Manager: Start via UI when needed"\n\
echo "📁 mcp_servers.json will be created when servers are added"\n\
echo ""\n\
echo "Starting supervisord..."\n\
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf\n\
' > /fractalic/entrypoint.sh && chmod +x /fractalic/entrypoint.sh

# Expose the necessary ports (explicitly list 5859 for MCP manager)
EXPOSE 8000 3000 8001 5859 8002 8003 8004

# Start with enhanced entrypoint script
CMD ["/fractalic/entrypoint.sh"]
