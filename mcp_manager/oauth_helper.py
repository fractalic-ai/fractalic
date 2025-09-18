#!/usr/bin/env python3
"""
OAuth Helper for Fractalic MCP Manager
Simple helper to use project-relative cache directory for OAuth tokens
"""

from pathlib import Path
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import FileTokenStorage
import logging

# Import centralized path management
from core.paths import get_oauth_cache_directory

logger = logging.getLogger(__name__)

def get_oauth_cache_dir() -> Path:
    """Get the OAuth cache directory path using centralized path management"""
    return get_oauth_cache_directory()

def ensure_oauth_cache_dir() -> Path:
    """Ensure OAuth cache directory exists and return its path"""
    oauth_cache_dir = get_oauth_cache_directory()
    oauth_cache_dir.mkdir(exist_ok=True)
    return oauth_cache_dir

def create_custom_oauth_client(mcp_url: str, **kwargs) -> OAuth:
    """Create OAuth client with custom cache directory.
    Let FastMCP choose callback port automatically to avoid conflicts.
    """
    from pathlib import Path
    oauth_cache_dir = Path(ensure_oauth_cache_dir())  # Convert to Path object

    # Prefer modern param names first
    params = dict(kwargs)
    # Don't set callback_port - let FastMCP choose a free port

    # Try modern name 'token_storage_cache_dir'
    try:
        logger.debug("Creating OAuth with token_storage_cache_dir (auto port)")
        return OAuth(mcp_url, token_storage_cache_dir=oauth_cache_dir, **params)
    except TypeError as e1:
        logger.debug(f"OAuth init failed with token_storage_cache_dir: {e1}; trying cache_dir")
        # Try legacy name 'cache_dir'
        try:
            return OAuth(mcp_url, cache_dir=oauth_cache_dir, **params)
        except TypeError as e2:
            logger.debug(f"OAuth init failed with cache_dir: {e2}; trying without callback_port")
            # As a last resort, try without callback_port
            try:
                return OAuth(mcp_url, token_storage_cache_dir=oauth_cache_dir)
            except TypeError as e3:
                logger.debug(f"OAuth init failed without callback_port (modern name): {e3}; trying legacy without callback_port")
                return OAuth(mcp_url, cache_dir=oauth_cache_dir)


def create_custom_token_storage(server_url: str) -> FileTokenStorage:
    """Create FileTokenStorage with custom cache directory"""
    oauth_cache_dir = ensure_oauth_cache_dir()
    return FileTokenStorage(server_url=server_url, cache_dir=oauth_cache_dir)
