#!/usr/bin/env python3
"""
OAuth Helper for Fractalic MCP Manager
Simple helper to use project-relative cache directory for OAuth tokens
"""

from pathlib import Path
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import FileTokenStorage

# Project root directory (where this file is located)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# OAuth cache directory relative to project root
OAUTH_CACHE_DIR = PROJECT_ROOT / "oauth-cache"

def get_oauth_cache_dir() -> Path:
    """Get the OAuth cache directory path"""
    return OAUTH_CACHE_DIR

def ensure_oauth_cache_dir() -> Path:
    """Ensure OAuth cache directory exists and return its path"""
    OAUTH_CACHE_DIR.mkdir(exist_ok=True)
    return OAUTH_CACHE_DIR

def create_custom_oauth_client(mcp_url: str, **kwargs) -> OAuth:
    """Create OAuth client with custom cache directory"""
    oauth_cache_dir = ensure_oauth_cache_dir()
    kwargs['token_storage_cache_dir'] = oauth_cache_dir
    return OAuth(mcp_url, **kwargs)

def create_custom_token_storage(server_url: str) -> FileTokenStorage:
    """Create FileTokenStorage with custom cache directory"""
    oauth_cache_dir = ensure_oauth_cache_dir()
    storage = FileTokenStorage(server_url=server_url)
    storage.cache_dir = oauth_cache_dir
    return storage
