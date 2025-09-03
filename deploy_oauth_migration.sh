#!/bin/bash
# deploy_oauth_migration.sh - OAuth Token Migration for Fractalic Deployment
# 
# This script handles OAuth token migration from development to production deployment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE_DIR="/tmp/fractalic_oauth"
DEFAULT_TARGET_DIR="/data/oauth"
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

# Step 2: List current tokens
log "Checking available OAuth tokens..."
if python3 "$FRACTALIC_MANAGER" oauth-list --storage-dir "$SOURCE_DIR"; then
    log ""
else
    error "Failed to list OAuth tokens"
    exit 1
fi

# Step 3: Migrate tokens
log "Migrating OAuth tokens to deployment directory..."
if python3 "$FRACTALIC_MANAGER" oauth-migrate "$SOURCE_DIR" "$TARGET_DIR"; then
    success "OAuth tokens migrated successfully"
else
    error "OAuth token migration failed"
    exit 1
fi

# Step 4: Verify migration
log "Verifying migrated tokens..."
if python3 "$FRACTALIC_MANAGER" oauth-list --storage-dir "$TARGET_DIR"; then
    success "Token migration verified"
else
    warn "Could not verify migrated tokens"
fi

log ""
log "🎉 OAuth Migration Complete!"
log ""
log "Next steps for deployment:"
log "1. Set environment variable: OAUTH_STORAGE_PATH=$TARGET_DIR"
log "2. Mount $TARGET_DIR as a volume in your container"
log "3. Deploy your containerized Fractalic instance"
log ""
log "Example Docker command:"
log "  docker run -v $TARGET_DIR:/data/oauth \\"
log "             -e OAUTH_STORAGE_PATH=/data/oauth \\"
log "             your-fractalic-image"
log ""
log "Example docker-compose.yml volume:"
log "  volumes:"
log "    - $TARGET_DIR:/data/oauth"
log "  environment:"
log "    - OAUTH_STORAGE_PATH=/data/oauth"
