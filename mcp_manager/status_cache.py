#!/usr/bin/env python3
"""
Status Cache - Simple in-memory cache for MCP runtime data
DOES NOT use DiskStore - OAuth tokens are managed by FastMCP OAuth helper
"""
import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class StatusCache:
    """Simple in-memory cache for MCP status/tools/prompts/resources

    NOTE: Does NOT persist to disk - all data cleared on restart.
    OAuth tokens are managed separately by FastMCP OAuth helper.
    """

    def __init__(self, default_ttl: float = 60.0):
        self.default_ttl = default_ttl
        # In-memory cache: key -> (data, expire_timestamp)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        logger.info(f"StatusCache initialized with {default_ttl}s TTL (in-memory)")

    def _is_expired(self, expire_time: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() > expire_time

    def _get(self, key: str) -> Optional[Any]:
        """Get from cache if not expired"""
        if key in self._cache:
            data, expire_time = self._cache[key]
            if not self._is_expired(expire_time):
                return data
            # Expired - remove
            del self._cache[key]
        return None

    def _set(self, key: str, data: Any, ttl: float):
        """Set cache entry with TTL"""
        expire_time = time.time() + ttl
        self._cache[key] = (data, expire_time)

    # Service status caching
    async def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get cached service status"""
        return self._get(f"status_{service_name}")

    async def set_service_status(self, service_name: str, status: Dict[str, Any]):
        """Cache service status with TTL"""
        self._set(f"status_{service_name}", status, self.default_ttl)

    # Service tools caching
    async def get_service_tools(self, service_name: str) -> Optional[list]:
        """Get cached service tools"""
        return self._get(f"tools_{service_name}")

    async def set_service_tools(self, service_name: str, tools: list):
        """Cache service tools with TTL"""
        self._set(f"tools_{service_name}", tools, self.default_ttl)

    # OAuth status caching
    async def get_oauth_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get cached OAuth status for a service"""
        return self._get(f"oauth_{service_name}")

    async def set_oauth_status(self, service_name: str, oauth_data: Dict[str, Any]):
        """Cache OAuth status for a service with TTL"""
        self._set(f"oauth_{service_name}", oauth_data, self.default_ttl)

    # Generic data caching
    async def get_cached_data(self, key: str, ttl: float | None = None) -> Optional[Any]:
        """Get generic cached data"""
        return self._get(key)

    async def set_cached_data(self, key: str, data: Any, ttl: float | None = None):
        """Cache generic data with optional TTL"""
        actual_ttl = ttl if ttl is not None else self.default_ttl
        self._set(key, data, actual_ttl)

    # Invalidation
    async def invalidate_service(self, service_name: str):
        """Invalidate all cached data for a service"""
        keys_to_delete = [
            f"status_{service_name}",
            f"tools_{service_name}",
            f"prompts_{service_name}",
            f"resources_{service_name}",
            f"oauth_{service_name}"
        ]
        for key in keys_to_delete:
            self._cache.pop(key, None)

    async def invalidate_cached_data(self, key: str):
        """Invalidate specific cached data"""
        self._cache.pop(key, None)

    # Aggregate helpers (compat with previous cache API)
    async def get_all_services_status(self) -> Optional[Dict[str, Any]]:
        """Return aggregated status if we have a recent index; otherwise None to trigger a poll."""
        try:
            idx = await self.get_cached_data("status_index")
            if not idx or not isinstance(idx, list):
                return None

            services: Dict[str, Any] = {}
            total_enabled = 0
            total_disabled = 0
            for name in idx:
                status = await self.get_service_status(name)
                if not status:
                    continue
                services[name] = status
                if status.get("enabled", False):
                    total_enabled += 1
                else:
                    total_disabled += 1

            if not services:
                return None

            return {
                "services": services,
                "total_enabled": total_enabled,
                "total_disabled": total_disabled,
            }
        except Exception:
            return None
