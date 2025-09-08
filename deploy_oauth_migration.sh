#!/bin/bash
# deploy_oauth_migration.sh - OAuth Token Migration for Fractalic Deployment
# 
# This script handles OAuth token migration from development to production deployment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE_DIR="$SCRIPT_DIR/oauth-cache"
DEFAULT_TARGET_DIR="/data/oauth-cache"
FRACTALIC_MANAGER="$SCRIPT_DIR/fractalic_mcp_manager.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[OAuth Deploy]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[OAuth Deploy]${NC} ⚠️  $1"
}

error() {
    echo -e "${RED}[OAuth Deploy]${NC} ❌ $1"
}

success() {
    echo -e "${GREEN}[OAuth Deploy]${NC} ✅ $1"
}

# Parse command line arguments
SOURCE_DIR="${1:-$DEFAULT_SOURCE_DIR}"
TARGET_DIR="${2:-$DEFAULT_TARGET_DIR}"

log "OAuth Token Migration for Fractalic Deployment"
log "================================================"
log "Source Directory: $SOURCE_DIR"
log "Target Directory: $TARGET_DIR"
log ""

# Step 1: Check if source directory exists and has tokens
if [[ ! -d "$SOURCE_DIR" ]]; then
    warn "Source directory does not exist: $SOURCE_DIR"
    log "Setting up OAuth tokens first..."
    
    # Check if Fractalic is running
    if pgrep -f "fractalic_mcp_manager.py serve" > /dev/null; then
        error "Fractalic server is already running. Please stop it first:"
        error "  pkill -f 'fractalic_mcp_manager.py serve'"
        exit 1
    fi
    
    # Run OAuth setup
    log "Running OAuth setup for HTTP servers..."
    if python3 "$FRACTALIC_MANAGER" oauth-setup; then
        success "OAuth setup completed"
    else
        error "OAuth setup failed"
        exit 1
    fi
fi

# Step 2: List current tokens if directory exists
if [[ -d "$SOURCE_DIR" ]] && [[ -n "$(ls -A "$SOURCE_DIR" 2>/dev/null)" ]]; then
    log "Checking available OAuth tokens..."
    log "Files in source directory:"
    ls -la "$SOURCE_DIR"
    log ""
else
    warn "No OAuth tokens found in $SOURCE_DIR"
    log "You may need to run OAuth setup first."
    log ""
fi

# Step 3: Copy tokens to deployment directory
log "Copying OAuth tokens to deployment directory..."
if [[ ! -d "$TARGET_DIR" ]]; then
    mkdir -p "$TARGET_DIR"
    log "Created target directory: $TARGET_DIR"
fi

# Copy all files from source to target
if cp -r "$SOURCE_DIR"/* "$TARGET_DIR"/ 2>/dev/null; then
    success "OAuth tokens copied successfully"
else
    warn "No OAuth tokens found to copy, or copy failed"
fi

# Step 4: Verify migration
log "Verifying copied tokens..."
if [[ -d "$TARGET_DIR" ]] && [[ -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]]; then
    success "Token copy verified - files present in $TARGET_DIR"
    log "Files in target directory:"
    ls -la "$TARGET_DIR"
else
    warn "Target directory is empty or does not exist"
fi

log ""
log "🎉 OAuth Token Copy Complete!"
log ""
log "Next steps for deployment:"
log "1. Copy the $TARGET_DIR to your deployment container"
log "2. Mount $TARGET_DIR as /fractalic/oauth-cache in your container"
log "3. Deploy your containerized Fractalic instance"
log ""
log "Example Docker command:"
log "  docker run -v $TARGET_DIR:/fractalic/oauth-cache \\"
log "             your-fractalic-image"
log ""
log "Example docker-compose.yml volume:"
log "  volumes:"
log "    - $TARGET_DIR:/fractalic/oauth-cache"
