#!/usr/bin/env python3
"""
OAuth Helper for Fractalic MCP Manager
Simple helper to use project-relative cache directory for OAuth tokens
"""

from pathlib import Path
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import FileTokenStorage

# Import centralized path management
from core.paths import get_oauth_cache_directory

def get_oauth_cache_dir() -> Path:
    """Get the OAuth cache directory path using centralized path management"""
    return get_oauth_cache_directory()

def ensure_oauth_cache_dir() -> Path:
    """Ensure OAuth cache directory exists and return its path"""
    oauth_cache_dir = get_oauth_cache_directory()
    oauth_cache_dir.mkdir(exist_ok=True)
    return oauth_cache_dir

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
