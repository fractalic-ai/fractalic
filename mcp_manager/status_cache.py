#!/usr/bin/env python3
"""
Service Status Cache Module
Handles caching of MCP service status, tools, and capabilities
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Single cache entry with TTL tracking"""
    data: Dict[str, Any]
    timestamp: float
    ttl: float = 60.0  # Default 60 second TTL
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - self.timestamp) > self.ttl
    
    def age(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.timestamp


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    updates: int = 0
    evictions: int = 0
    last_refresh: Optional[float] = None
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "updates": self.updates,
            "evictions": self.evictions,
            "hit_rate": f"{self.hit_rate:.1f}%",
            "last_refresh": datetime.fromtimestamp(self.last_refresh).isoformat() if self.last_refresh else None
        }


class StatusCache:
    """
    Granular caching for MCP service status and capabilities
    Supports per-service TTL and partial updates
    """
    
    def __init__(self, default_ttl: float = 60.0):
        """
        Initialize cache with configurable TTL
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
        
        # Per-service caches
        self.service_status: Dict[str, CacheEntry] = {}
        self.service_tools: Dict[str, CacheEntry] = {}
        self.service_prompts: Dict[str, CacheEntry] = {}
        self.service_resources: Dict[str, CacheEntry] = {}
        self.oauth_status: Dict[str, CacheEntry] = {}
        
        # Complete status cache (single entry for all data)
        self.complete_status: Optional[CacheEntry] = None
        
        # Aggregate counters (updated incrementally)
        self.total_enabled = 0
        self.total_disabled = 0
        
        # Cache statistics
        self.stats = CacheStatistics()
        
        # Lock for thread-safe operations
        self.lock = asyncio.Lock()
        
        logger.info(f"StatusCache initialized with {default_ttl}s TTL")
    
    async def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get cached service status"""
        async with self.lock:
            entry = self.service_status.get(service_name)
            
            if entry and not entry.is_expired():
                self.stats.hits += 1
                logger.debug(f"Cache hit for service status: {service_name} (age: {entry.age():.1f}s)")
                return entry.data
            
            self.stats.misses += 1
            logger.debug(f"Cache miss for service status: {service_name}")
            return None
    
    async def set_service_status(self, service_name: str, status: Dict[str, Any], ttl: Optional[float] = None):
        """Cache service status"""
        async with self.lock:
            # Update aggregate counters based on status change
            old_entry = self.service_status.get(service_name)
            if old_entry:
                old_enabled = old_entry.data.get("enabled", False)
                new_enabled = status.get("enabled", False)
                
                if old_enabled != new_enabled:
                    if new_enabled:
                        self.total_enabled += 1
                        self.total_disabled -= 1
                    else:
                        self.total_enabled -= 1
                        self.total_disabled += 1
            else:
                # New service
                if status.get("enabled", False):
                    self.total_enabled += 1
                else:
                    self.total_disabled += 1
            
            self.service_status[service_name] = CacheEntry(
                data=status,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )
            self.stats.updates += 1
            logger.debug(f"Cached service status for {service_name}")
    
    async def remove_service(self, service_name: str):
        """Remove service from all caches"""
        async with self.lock:
            # Update counters if service was in status cache
            if service_name in self.service_status:
                entry = self.service_status[service_name]
                if entry.data.get("enabled", False):
                    self.total_enabled -= 1
                else:
                    self.total_disabled -= 1
                del self.service_status[service_name]
                self.stats.evictions += 1
            
            # Remove from other caches
            for cache in [self.service_tools, self.service_prompts, 
                         self.service_resources, self.oauth_status]:
                if service_name in cache:
                    del cache[service_name]
                    self.stats.evictions += 1
            
            logger.debug(f"Removed service from cache: {service_name}")
    
    async def get_service_tools(self, service_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached service tools"""
        async with self.lock:
            entry = self.service_tools.get(service_name)
            
            if entry and not entry.is_expired():
                self.stats.hits += 1
                logger.debug(f"Cache hit for service tools: {service_name}")
                return entry.data.get("tools", [])
            
            self.stats.misses += 1
            return None
    
    async def set_service_tools(self, service_name: str, tools: List[Dict[str, Any]], ttl: Optional[float] = None):
        """Cache service tools"""
        async with self.lock:
            self.service_tools[service_name] = CacheEntry(
                data={"tools": tools, "count": len(tools)},
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )
            self.stats.updates += 1
            logger.debug(f"Cached {len(tools)} tools for {service_name}")
    
    async def get_all_services_status(self) -> Optional[Dict[str, Any]]:
        """Get complete status for all cached services"""
        async with self.lock:
            # Check if we have any valid cache entries
            valid_entries = {
                name: entry.data 
                for name, entry in self.service_status.items() 
                if not entry.is_expired()
            }
            
            if not valid_entries:
                self.stats.misses += 1
                return None
            
            self.stats.hits += 1
            return {
                "services": valid_entries,
                "total_enabled": self.total_enabled,
                "total_disabled": self.total_disabled
            }
    
    async def get_oauth_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get cached OAuth status for service"""
        async with self.lock:
            entry = self.oauth_status.get(service_name)
            
            if entry and not entry.is_expired():
                self.stats.hits += 1
                return entry.data
            
            self.stats.misses += 1
            return None
    
    async def set_oauth_status(self, service_name: str, status: Dict[str, Any], ttl: Optional[float] = None):
        """Cache OAuth status for service"""
        async with self.lock:
            self.oauth_status[service_name] = CacheEntry(
                data=status,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )
            self.stats.updates += 1
    
    async def invalidate_service(self, service_name: str):
        """Invalidate all cache entries for a specific service"""
        async with self.lock:
            invalidated = 0
            for cache in [self.service_status, self.service_tools, 
                         self.service_prompts, self.service_resources, 
                         self.oauth_status]:
                if service_name in cache:
                    del cache[service_name]
                    invalidated += 1
            
            if invalidated > 0:
                self.stats.evictions += invalidated
                logger.debug(f"Invalidated {invalidated} cache entries for {service_name}")
    
    async def clear_expired(self):
        """Remove all expired entries from cache"""
        async with self.lock:
            expired_count = 0
            
            for cache in [self.service_status, self.service_tools, 
                         self.service_prompts, self.service_resources, 
                         self.oauth_status]:
                expired_keys = [
                    key for key, entry in cache.items() 
                    if entry.is_expired()
                ]
                for key in expired_keys:
                    del cache[key]
                    expired_count += 1
            
            if expired_count > 0:
                self.stats.evictions += expired_count
                logger.debug(f"Cleared {expired_count} expired cache entries")
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics"""
        async with self.lock:
            # Count valid entries
            valid_status = sum(1 for e in self.service_status.values() if not e.is_expired())
            valid_tools = sum(1 for e in self.service_tools.values() if not e.is_expired())
            valid_oauth = sum(1 for e in self.oauth_status.values() if not e.is_expired())
            
            return {
                "statistics": self.stats.to_dict(),
                "cache_counts": {
                    "service_status": valid_status,
                    "service_tools": valid_tools,
                    "oauth_status": valid_oauth,
                    "total_entries": valid_status + valid_tools + valid_oauth
                },
                "ttl_settings": {
                    "default_ttl": self.default_ttl,
                    "oldest_entry": self._get_oldest_entry_age()
                }
            }
    
    def _get_oldest_entry_age(self) -> Optional[float]:
        """Get age of oldest valid cache entry"""
        oldest_timestamp = None
        
        for cache in [self.service_status, self.service_tools, 
                     self.service_prompts, self.service_resources, 
                     self.oauth_status]:
            for entry in cache.values():
                if not entry.is_expired():
                    if oldest_timestamp is None or entry.timestamp < oldest_timestamp:
                        oldest_timestamp = entry.timestamp
        
        return time.time() - oldest_timestamp if oldest_timestamp else None
    
    async def warm_cache(self, services: List[str]):
        """Pre-populate cache markers for services to track"""
        async with self.lock:
            for service_name in services:
                if service_name not in self.service_status:
                    # Create placeholder entry to track service
                    self.service_status[service_name] = CacheEntry(
                        data={"placeholder": True},
                        timestamp=0,  # Ancient timestamp to force refresh
                        ttl=0
                    )
            logger.info(f"Cache warmed with {len(services)} services to track")
    
    async def get_cached_data(self, key: str, ttl: Optional[float] = None) -> Optional[Any]:
        """Get cached data by key"""
        if not hasattr(self, 'generic_cache'):
            self.generic_cache = {}
        
        if key in self.generic_cache:
            entry = self.generic_cache[key]
            if ttl is not None:
                entry.ttl = ttl
            if not entry.is_expired():
                self.stats.hits += 1
                return entry.data
            else:
                # Remove expired entry
                del self.generic_cache[key]
        
        self.stats.misses += 1
        return None
    
    async def set_cached_data(self, key: str, data: Any, ttl: Optional[float] = None) -> None:
        """Set cached data by key"""
        if not hasattr(self, 'generic_cache'):
            self.generic_cache = {}
        
        cache_ttl = ttl if ttl is not None else self.default_ttl
        self.generic_cache[key] = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=cache_ttl
        )
        logger.debug(f"Cached data for key: {key} (TTL: {cache_ttl}s)")
    
    async def invalidate_cached_data(self, key: str) -> None:
        """Invalidate cached data by key"""
        if not hasattr(self, 'generic_cache'):
            self.generic_cache = {}
        
        if key in self.generic_cache:
            del self.generic_cache[key]
            logger.debug(f"Invalidated cached data for key: {key}")
            self.stats.evictions += 1