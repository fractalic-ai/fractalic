#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thin wrapper for Model-Context-Protocol discovery/execution.
You can swap this for an official SDK later.
"""
from __future__ import annotations
import requests
import time
from typing import Dict, Any, List, Optional

# Simple caching to reduce repeated HTTP calls to MCP servers
_list_tools_cache: Dict[str, tuple] = {}      # server -> (response, timestamp)
_list_prompts_cache: Dict[str, tuple] = {}    # server -> (response, timestamp)
_list_resources_cache: Dict[str, tuple] = {}  # server -> (response, timestamp)
_CACHE_DURATION = 30  # Cache for 30 seconds

def clear_cache(server: str | None = None):
    """Clear cached MCP discovery data.

    Args:
        server: Optional server base URL. If provided, only that server's cached
                entries (tools/prompts/resources) are cleared; otherwise all
                cached data is purged.
    """
    global _list_tools_cache, _list_prompts_cache, _list_resources_cache
    if server:
        _list_tools_cache.pop(server, None)
        _list_prompts_cache.pop(server, None)
        _list_resources_cache.pop(server, None)
    else:
        _list_tools_cache.clear()
        _list_prompts_cache.clear()
        _list_resources_cache.clear()

def list_tools(server: str) -> List[Dict[str, Any]]:
    """List tools from MCP server with caching to reduce load."""
    current_time = time.time()
    
    # Check if we have a recent cached response
    if server in _list_tools_cache:
        cached_response, timestamp = _list_tools_cache[server]
        if current_time - timestamp < _CACHE_DURATION:
            return cached_response
    
    # Fetch fresh data and cache it
    try:
        response = requests.get(f"{server.rstrip('/')}/list_tools", timeout=15).json()
        _list_tools_cache[server] = (response, current_time)
        return response
    except Exception as e:
        # If there's an error and we have cached data, return it (even if stale)
        if server in _list_tools_cache:
            cached_response, _ = _list_tools_cache[server]
            return cached_response
        raise e

def _cached_get(server: str, path: str, cache: Dict[str, tuple], ttl: int) -> Optional[Dict[str, Any]]:
    """Generic cached GET helper.

    Returns parsed JSON dict (or list) or None on hard failure.
    """
    now = time.time()
    # Serve fresh-enough cache
    if server in cache:
        data, ts = cache[server]
        if now - ts < ttl:
            return data
    try:
        resp = requests.get(f"{server.rstrip('/')}/{path.lstrip('/')}", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        cache[server] = (data, now)
        return data
    except Exception:
        # Return stale if exists
        if server in cache:
            return cache[server][0]
        return None

def list_prompts(server: str, service: Optional[str] = None) -> Dict[str, Any]:
    """List prompts from MCP manager.

    Server endpoint: GET /list_prompts -> { serviceA: {prompts:[...]}, ... }
    If service is specified, filter to that service only (empty dict if none).
    """
    data = _cached_get(server, "/list_prompts", _list_prompts_cache, _CACHE_DURATION)
    if not data or not isinstance(data, dict):
        return {}
    if service:
        svc = data.get(service)
        return {service: svc} if svc else {}
    return data

def get_prompt(server: str, service: str, prompt: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Retrieve a specific prompt template/content.

    POST /prompt/{service}/{prompt}  body: {"arguments": {...}}
    Returns prompt description + message content structure.
    """
    try:
        resp = requests.post(
            f"{server.rstrip('/')}/prompt/{service}/{prompt}",
            json={"arguments": arguments or {}},
            timeout=15,
        )
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text}", "isError": True}
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}", "isError": True}

def list_resources(server: str, service: Optional[str] = None) -> Dict[str, Any]:
    """List resources from MCP manager.

    GET /list_resources -> { serviceA: {resources:[...]}, ... }
    If service provided, filter accordingly.
    """
    data = _cached_get(server, "/list_resources", _list_resources_cache, _CACHE_DURATION)
    if not data or not isinstance(data, dict):
        return {}
    if service:
        svc = data.get(service)
        return {service: svc} if svc else {}
    return data

def read_resource(server: str, service: str, uri: str) -> Dict[str, Any]:
    """Read a resource's content from MCP manager.

    POST /resource/{service}/read  body: {"uri": "..."}
    Returns structured contents (text/blob entries).
    """
    try:
        resp = requests.post(
            f"{server.rstrip('/')}/resource/{service}/read",
            json={"uri": uri},
            timeout=20,
        )
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text}", "isError": True}
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}", "isError": True}

def call_tool(server: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Call a tool on an MCP server with better error handling."""
    try:
        # Handle server URL resolution
        if not server.startswith(('http://', 'https://')):
            # Default to local MCP manager for service names
            server = "http://127.0.0.1:5859"
        
        # Parse tool name to extract service and tool parts
        # For tools like "memory.read_graph", service="memory", tool="read_graph"
        if "." in name:
            service, tool = name.split(".", 1)
        else:
            # For tools without service prefix, try to infer from server or use the original name
            if server == "http://127.0.0.1:5859":
                # When calling MCP manager, use the tool name as service if no prefix
                service = name.split("_")[0] if "_" in name else "unknown"
                tool = name
            else:
                service = "unknown" 
                tool = name
            
        # Use the SDK v2 endpoint format: /call/{service}/{tool}
        url = f"{server.rstrip('/')}/call/{service}/{tool}"
        
        # Increased timeout for complex operations (especially Replicate API calls)
        # and add retry logic for network issues
        import time
        for attempt in range(3):
            try:
                response = requests.post(url,
                                       json={"arguments": args},  # SDK v2 expects just arguments
                                       timeout=90)  # Increased from 30s to 90s
                break  # Success, exit retry loop
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ConnectionResetError) as e:
                if attempt < 2:  # Only retry for first 2 attempts
                    print(f"⚠️  Network issue on attempt {attempt + 1}/3: {e}. Retrying in 2s...")
                    time.sleep(2)
                    continue
                else:
                    return {
                        "error": f"Network timeout after 3 attempts: {str(e)}",
                        "isError": True
                    }
        
        # Check HTTP status
        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}: {response.text}",
                "isError": True
            }
        
        # Try to parse JSON
        try:
            return response.json()
        except ValueError as e:
            return {
                "error": f"Invalid JSON response: {str(e)}. Raw response: {response.text[:200]}",
                "isError": True
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Network error: {str(e)}",
            "isError": True
        }
