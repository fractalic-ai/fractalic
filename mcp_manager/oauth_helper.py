#!/usr/bin/env python3
"""
OAuth Helper for Fractalic MCP Manager
======================================
Provides OAuth clients with working auto-refresh functionality.

FIXES FastMCP Bugs:
1. DiskStore deletes refresh_token when access_token expires (TTL issue)
2. FastMCP recalculates token expiry incorrectly on load

Solution: NoTTLDiskStore + FixedExpiryOAuth
- NoTTLDiskStore: Ignores TTL, returns expired tokens (refresh_token accessible)
- FixedExpiryOAuth: Uses DB expire_time instead of recalculating
- Result: Working OAuth auto-refresh without browser prompts
"""

import time
import sqlite3
import json
from pathlib import Path
from typing import Any, Optional
import logging

from fastmcp.client.auth import OAuth
from key_value.aio.stores.disk import DiskStore

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


class NoTTLDiskStore:
    """
    DiskStore wrapper that IGNORES TTL for .get() operations.

    Problem: DiskStore.get() returns None for expired entries, deleting refresh_token
    Solution: Query SQLite directly, bypassing TTL check

    This allows refresh_token to be retrieved even when access_token has expired.
    """

    def __init__(self, directory: str | Path):
        self._directory = Path(directory)
        self._store = DiskStore(directory=str(directory))

    async def get(self, key: str, collection: Optional[str] = None) -> Optional[Any]:
        """
        Get value from database IGNORING expire_time.

        Returns expired entries that DiskStore would normally filter out.
        Critical for OAuth refresh_token retrieval.
        """
        # Handle collection prefix
        full_key = f"{collection}::{key}" if collection else key
        logger.debug(f"NoTTLDiskStore.get() called for key: {full_key}")

        db_path = self._directory / "cache.db"
        if not db_path.exists():
            logger.debug(f"NoTTLDiskStore: DB not found at {db_path}")
            return None

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Query WITHOUT checking expire_time
            cursor.execute(
                "SELECT value, expire_time FROM Cache WHERE key = ?",
                (full_key,)
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                logger.debug(f"NoTTLDiskStore: No entry found for key {full_key}")
                return None

            # Parse nested JSON structure
            value_json, expire_time = row[0], row[1]
            data = json.loads(value_json)
            value = data.get('value')

            if expire_time:
                import time
                is_expired = expire_time < time.time()
                logger.info(f"NoTTLDiskStore: Token loaded for {full_key} (expired: {is_expired})")
            else:
                logger.info(f"NoTTLDiskStore: Token loaded for {full_key} (no expiry)")

            return value

        except Exception as e:
            logger.error(f"Error in NoTTLDiskStore.get for {full_key}: {e}", exc_info=True)
            return None

    async def put(self, key: str, value: Any, ttl: Optional[int] = None, collection: Optional[str] = None) -> None:
        """Delegate writes to underlying DiskStore."""
        await self._store.put(key, value, ttl=ttl, collection=collection)

    async def delete(self, key: str, collection: Optional[str] = None) -> None:
        """Delegate delete to underlying store."""
        await self._store.delete(key, collection=collection)

    async def exists(self, key: str) -> bool:
        """Check if key exists (ignoring TTL)."""
        value = await self.get(key)
        return value is not None

    def __getattr__(self, name):
        """Delegate unknown methods to underlying store."""
        return getattr(self._store, name)


class FixedExpiryOAuth(OAuth):
    """
    OAuth client with FIXED token expiry calculation.

    Problem: FastMCP's OAuth._initialize() recalculates expiry as:
        self.context.token_expiry_time = time.time() + token.expires_in

    This makes expired tokens appear fresh! expires_in is a DURATION (3600s),
    not an absolute timestamp.

    Solution: After loading tokens, query the DiskStore's database to get
    the actual expire_time timestamp (correctly calculated when saved).
    Use that as token_expiry_time instead of recalculating.
    """

    async def _initialize(self) -> None:
        """Load stored tokens and CORRECTLY set token expiry from database."""
        logger.debug(f"FixedExpiryOAuth._initialize() called for {self.server_base_url}")

        # Call parent to load tokens and client info
        await super()._initialize()

        # If we have tokens, get the ACTUAL expiry from database
        if self.context.current_tokens:
            logger.debug(f"FixedExpiryOAuth: Tokens loaded from storage, reading expire_time from DB")
            try:
                # Get token cache key (same as TokenStorageAdapter)
                token_key = f"{self.server_base_url}/tokens"

                # Access underlying DiskStore to read expire_time
                disk_store = self.token_storage_adapter._key_value_store

                # Get database path
                if hasattr(disk_store, '_directory'):
                    db_path = disk_store._directory / "cache.db"
                else:
                    # Fallback for regular DiskStore
                    db_dir = getattr(disk_store, 'directory', None) or getattr(disk_store, '_directory', None)
                    db_path = Path(str(db_dir)) / "cache.db"

                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()

                    # Get expire_time from database
                    full_key = f"mcp-oauth-token::{token_key}"
                    cursor.execute(
                        "SELECT expire_time FROM Cache WHERE key = ?",
                        (full_key,)
                    )
                    row = cursor.fetchone()
                    conn.close()

                    if row and row[0]:
                        db_expire_time = row[0]

                        # Use database expire_time as the token expiry
                        self.context.token_expiry_time = db_expire_time

                        time_remaining = db_expire_time - time.time()

                        if time_remaining > 0:
                            logger.info(f"OAuth token loaded: {time_remaining:.0f}s remaining")
                        else:
                            logger.info(f"OAuth token EXPIRED: {-time_remaining:.0f}s ago (will auto-refresh)")

                        return
                    else:
                        logger.warning(f"FixedExpiryOAuth: No expire_time found in DB for {full_key}")

            except Exception as e:
                logger.error(f"Could not read database expire_time: {e}", exc_info=True)

            # Fallback to parent behavior if database read fails
            if self.context.current_tokens and self.context.current_tokens.expires_in:
                self.context.update_token_expiry(self.context.current_tokens)
                logger.warning(
                    f"Using fallback expiry calculation (may be incorrect after reload)"
                )
        else:
            logger.debug("FixedExpiryOAuth: No tokens loaded from storage")


def create_custom_oauth_client(mcp_url: str, service_name: str, **kwargs) -> OAuth:
    """
    Create OAuth client with working auto-refresh.

    Uses NoTTLDiskStore + FixedExpiryOAuth to fix FastMCP OAuth bugs:
    - Loads expired tokens (refresh_token accessible)
    - Correctly calculates token expiry from database
    - Enables automatic token refresh without browser

    Each service gets its own storage directory to prevent token conflicts.

    Args:
        mcp_url: MCP server URL
        service_name: Name of the MCP service (for per-service storage)
        **kwargs: Additional OAuth parameters (e.g., callback_port)

    Returns:
        FixedExpiryOAuth instance with working auto-refresh
    """
    oauth_cache_dir = Path(ensure_oauth_cache_dir())

    # Create per-service storage directory (prevents token conflicts)
    service_cache_dir = oauth_cache_dir / service_name
    service_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating NoTTLDiskStore for {service_name} at {service_cache_dir}")
    token_storage = NoTTLDiskStore(directory=service_cache_dir)

    params = dict(kwargs)
    logger.debug(f"Creating FixedExpiryOAuth client for {service_name} with auto-refresh support")

    return FixedExpiryOAuth(mcp_url, token_storage=token_storage, **params)
