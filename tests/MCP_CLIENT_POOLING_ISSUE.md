# MCP Manager Client Pooling Issue - Root Cause Analysis

**Date**: 2025-11-06
**Issue**: MCP servers occasionally "die" and stop responding, requiring server restart

## Problem

MCP Manager caches `Client` instances in `_client_pool` (fastmcp_manager.py:55-57), but:

1. **FastMCP Client lifecycle**: Requires `async with` context manager for proper connection management
2. **Current usage**: We store Client in pool, then use `async with client` for each request
3. **Result**: Each `async with` re-opens connection, so pooling provides NO benefit
4. **Side effect**: Subprocess connections (STDIO servers) can "die" or hang while Client sits in pool

## Evidence

### FastMCP Documentation
- "All client operations require using the `async with` context manager for proper connection lifecycle management"
- Connection is opened on `async with` entry, closed on exit
- Pooling Client instances WITHOUT active context provides no performance benefit

### Current Code Pattern

```python
# fastmcp_manager.py:184-205
# STDIO/bearer servers: pooled (PROBLEM)
if service_name in self._client_pool:
    return self._client_pool[service_name]  # Reuse pooled Client

# Later usage (line 239):
async with client as c:  # ← Opens NEW connection each time!
    tools = await c.list_tools()
# ← Closes connection
```

**Problem**: Client in pool has no active connection. Each `async with` opens/closes fresh connection anyway.

### Observed Symptoms

1. `/tools` endpoint hanging for 60s (timeout)
2. Server completely freezing, not responding to any requests
3. Requires full restart to recover
4. Happens intermittently over time

### Test Results

- **Cold start**: 18.3s for all 9 servers (parallel)  ← ACCEPTABLE
- **After restart**: Works initially, then degrades
- **Problem**: Pooled clients accumulate "dead" subprocess connections

## Root Cause

**Client pooling is fundamentally incompatible with FastMCP's connection model:**

1. Client stores subprocess/connection config, NOT an open connection
2. `async with` manages actual connection lifecycle
3. Pooling Client provides zero benefit (connection reopens each time)
4. Pooled Clients can accumulate stale subprocess references
5. When subprocess dies/hangs, Client in pool becomes unusable

## Solution

### Remove Client Pooling Entirely

**Current (WRONG)**:
```python
# Pool clients
self._client_pool: Dict[str, Client] = {}

# Reuse from pool
if service_name in self._client_pool:
    return self._client_pool[service_name]

# Later:
async with client as c:  # Opens connection
    await c.list_tools()
```

**Correct approach**:
```python
# NO client pool

# Create fresh Client every time
client = self.create_fastmcp_client(service_name, context)

# Use immediately with context manager
async with client as c:  # Opens connection
    await c.list_tools()
# Client discarded after use
```

### What To Cache Instead

**Cache results, not connections:**

```python
# CACHE these (already implemented):
await self.cache.set_service_tools(service_name, tools)        # ✅
await self.cache.set_service_status(service_name, status)      # ✅
await self.cache.set_cached_data(f"prompts_{service_name}")    # ✅

# DO NOT cache these:
self._client_pool[service_name] = client  # ❌ REMOVE
```

## Implementation Plan

### 1. Remove Client Pooling Infrastructure

**File**: `mcp_manager/fastmcp_manager.py`

```python
# REMOVE these (lines 55-57):
self._client_pool: Dict[str, Client] = {}
self._client_locks: Dict[str, asyncio.Lock] = {}
self._pool_lock = asyncio.Lock()
```

### 2. Simplify `get_or_create_client()` Method

**Current** (lines 175-205): Complex pooling logic with locks

**Replace with**:
```python
async def get_or_create_client(self, service_name: str, context: Dict[str, Any] | None = None) -> Client:
    """Create FRESH client every time (no pooling)"""
    config = self.service_configs.get(service_name)
    if not config:
        return None

    # Simply create and return new client
    return self.create_fastmcp_client(service_name, context)
```

### 3. Remove `cleanup_clients()` Method

No longer needed - clients are created/destroyed with `async with`, no cleanup required.

### 4. Keep OAuth Pool (Optional)

OAuth instances can be pooled since they only hold tokens, not connections:

```python
# OAuth pooling is OK (lines 62, 80-93):
self._oauth_pool: Dict[str, OAuth] = {}  # ✅ Keep this
```

But even OAuth pooling might not be necessary - consider removing it too for simplicity.

### 5. Verify Cache Layer

Ensure all expensive operations are cached:

- ✅ Tools (TTL: 300s) - `cache.set_service_tools()`
- ✅ Status (TTL: 300s) - `cache.set_service_status()`
- ✅ Prompts (TTL: 60s) - `cache.set_cached_data()`
- ✅ Resources (TTL: 60s) - `cache.set_cached_data()`
- ✅ All tools collection (TTL: 30s) - `cache.set_cached_data("all_tools")`

## Expected Results

**After fix:**

1. **No more hanging**: Fresh Client = fresh subprocess, can't accumulate stale connections
2. **Same performance**: Cache provides speed, not client pooling
3. **Simpler code**: ~50 lines removed (pooling logic, locks, cleanup)
4. **More reliable**: Follows FastMCP's intended usage pattern

**Performance impact:**

- **First request**: Same (18s cold start already acceptable)
- **Cached requests**: Same (cache handles speed, not pooling)
- **Long-running**: Better (no connection degradation over time)

## Testing Plan

1. Remove client pooling code
2. Test cold start performance (should be ~18s like before)
3. Test cached requests (should be <100ms from cache)
4. **Stress test**: Run repeated requests over 1 hour, verify no hangs
5. Test all 9 servers individually after stress test
6. Verify no timeouts or "dead" servers

## References

- FastMCP docs: https://gofastmcp.com/clients/client
- Client lifecycle: "All client operations require using the `async with` context manager"
- Our issue: Servers freeze intermittently, require restart
- Root cause: Client pooling incompatible with FastMCP connection model
