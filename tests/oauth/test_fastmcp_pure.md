# Pure FastMCP Library Test

**Test File:** `test_fastmcp_pure.py`
**Created:** November 5, 2025
**Purpose:** Validate FastMCP library OAuth functionality without Fractalic dependencies

## Overview

This test validates that the FastMCP library correctly handles OAuth authentication and token refresh for MCP servers. It is a **standalone test** that does not use any Fractalic modules.

## What This Test Does

1. **Loads MCP Server Configuration**
   - Reads `mcp_servers.json` from project root
   - Identifies OAuth-enabled servers (those with `url` and `oauth_callback_port`)

2. **Creates OAuth Clients with Auto-Refresh Workaround**
   - Uses `NoTTLDiskStore` to bypass TTL expiration checks
   - Uses `FixedExpiryOAuth` to correctly detect token expiry
   - Enables automatic refresh token usage (no browser popup on token expiry)

3. **Connects to Each Server**
   - Creates FastMCP Client with OAuth authentication
   - Measures connection time
   - Handles both fresh OAuth flow and token reuse

4. **Lists Available Tools**
   - Retrieves tool list from each server
   - Displays first 5 tools
   - Reports tool counts

5. **Generates Test Report**
   - Prints summary to console
   - Saves detailed JSON report to `test_fastmcp_pure_results.json`

## Test Structure

```
test_fastmcp_pure.py
├── OAuth Workaround Classes (inline)
│   ├── NoTTLDiskStore        # Bypasses TTL checks
│   ├── FixedExpiryOAuth      # Correct expiry detection
│   └── create_oauth_with_auto_refresh()
│
├── Test Functions
│   ├── load_mcp_config()     # Load mcp_servers.json
│   ├── get_oauth_servers()   # Extract OAuth servers
│   ├── test_server()         # Test single server
│   └── main()                # Orchestrate tests
```

## Dependencies

**Required Libraries:**
- `fastmcp` - Main MCP client library
- `key-value-aio` - Storage backend (DiskStore)

**Standard Libraries:**
- `asyncio` - Async execution
- `json` - Config and results
- `sqlite3` - Direct database access
- `pathlib` - Path handling
- `time` - Timing measurements

**No Fractalic Imports!** This test is completely standalone.

## OAuth Auto-Refresh Workaround

### The Problem

FastMCP has a bug where refresh tokens are deleted when access tokens expire:

```python
# In fastmcp/client/auth/oauth.py
await storage.put(
    key="token",
    value={access_token, refresh_token},  # Both tokens
    ttl=access_token.expires_in            # Only 1 hour!
)
```

When `access_token` expires, DiskStore deletes the **entire entry** including `refresh_token`.

### The Solution

**Part 1: NoTTLDiskStore**
- Wraps DiskStore
- Queries SQLite directly, ignoring `expire_time`
- Returns tokens even if expired
- Allows FastMCP to access `refresh_token`

**Part 2: FixedExpiryOAuth**
- Extends FastMCP's OAuth class
- Reads actual `expire_time` from database
- Doesn't recalculate expiry (which makes expired tokens appear fresh)
- Enables correct expiry detection

**Result:** FastMCP can now:
1. Load expired `access_token` + valid `refresh_token`
2. Detect that `access_token` is expired
3. Use `refresh_token` to get new `access_token`
4. Continue without browser popup

## Usage

### Run the test:

```bash
cd /Users/dima/fractalic/fractalic
/Users/dima/fractalic/fractalic/.venv/bin/python3 tests/oauth/test_fastmcp_pure.py
```

### First run (fresh OAuth):
- Browser opens for each server
- User authorizes
- Tokens saved
- Tools retrieved
- Takes 15-20s per server

### Subsequent runs (token reuse):
- No browser popup
- Tokens loaded from cache
- Tools retrieved
- Takes 5-7s per server

### Expired token run (auto-refresh):
- No browser popup
- Expired `access_token` detected
- `refresh_token` used automatically
- New `access_token` obtained
- Tools retrieved
- Takes 5-7s per server

## Expected Results

### Successful Test Output

```
======================================================================
PURE FASTMCP LIBRARY TEST
Testing OAuth servers with auto-refresh workaround
======================================================================

Config: /Users/dima/fractalic/fractalic/mcp_servers.json
Cache: /Users/dima/.fastmcp/oauth-test-pure

Loading MCP server configuration...
Found 2 OAuth-enabled servers:
  - notion (port 58100)
  - Replicate (port 58101)

======================================================================
Testing: notion
URL: https://mcp.notion.com/mcp
Callback Port: 58100
======================================================================

  1. Creating OAuth client with auto-refresh...
  2. Connecting to notion...
    ✓ OAuth token loaded: 2847s remaining
    ✓ Connected in 4.23s
  3. Listing tools...
    ✓ Got 15 tools

    First 5 tools:
      - search
      - create_page
      - update_page
      - get_page
      - get_database

  ✅ SUCCESS: notion - 15 tools in 5.12s

[Similar output for Replicate]

======================================================================
TEST RESULTS SUMMARY
======================================================================

Overall: 2/2 servers successful

✅ notion            15 tools |   5.12s
✅ Replicate         12 tools |   6.34s

📊 Results saved to: /Users/dima/fractalic/fractalic/test_fastmcp_pure_results.json
```

### Results JSON Structure

```json
{
  "timestamp": 1730860234.567,
  "summary": {
    "total": 2,
    "success": 2,
    "failed": 0
  },
  "results": [
    {
      "name": "notion",
      "url": "https://mcp.notion.com/mcp",
      "success": true,
      "tools": [
        {"name": "search"},
        {"name": "create_page"},
        {"name": "update_page"},
        {"name": "get_page"},
        {"name": "get_database"}
      ],
      "tool_count": 15,
      "error": null,
      "connect_time": 4.23,
      "total_time": 5.12
    }
  ]
}
```

## Verification

This test verifies:

- ✅ **FastMCP Client creation** - Can instantiate clients with OAuth
- ✅ **OAuth authentication** - Can authenticate via OAuth flow
- ✅ **Token persistence** - Tokens saved and reloaded correctly
- ✅ **Token reuse** - Valid tokens used without re-auth
- ✅ **Auto-refresh** - Expired tokens refreshed automatically (with workaround)
- ✅ **Tool listing** - Can retrieve tool schemas from servers
- ✅ **Multi-server** - Works with multiple OAuth providers

## Known Issues

### FastMCP Bug #1863
**Issue:** OAuth token refresh does not work out-of-the-box
**Status:** Open (as of November 2025)
**Workaround:** This test uses NoTTLDiskStore + FixedExpiryOAuth
**Link:** https://github.com/jlowin/fastmcp/issues/1863

### Provider-Specific Behavior
Some OAuth providers (like Notion) may accept expired tokens within a grace period. True auto-refresh behavior depends on provider strictly enforcing expiry.

## Testing Token Refresh

To test auto-refresh, manually expire a token:

```bash
# Age the token in database (make it expired 1 hour ago)
sqlite3 ~/.fastmcp/oauth-test-pure/notion/cache.db \
  "UPDATE Cache SET expire_time = unixepoch('now') - 3600 \
   WHERE key LIKE '%tokens'"

# Run test - should auto-refresh, not open browser
python3 tests/oauth/test_fastmcp_pure.py
```

Expected: `⚠️ OAuth token EXPIRED: 3600s ago (will auto-refresh)`

## Files

- `test_fastmcp_pure.py` - Main test script (420 lines)
- `test_fastmcp_pure.md` - This documentation
- `test_fastmcp_pure_results.json` - Test results (generated)

## Maintenance

### When to Update

1. **FastMCP library updates** - Check if auto-refresh bug is fixed
2. **New OAuth servers** - Added to `mcp_servers.json`
3. **Workaround improvements** - Better handling of edge cases

### Related Files

- `/mcp_servers.json` - Server configuration
- `/oauth_workaround_complete.py` - Workaround reference implementation
- `/OAUTH_WORKAROUND_FINAL_REPORT.md` - Detailed analysis
- `/FASTMCP_OAUTH_ROOT_CAUSE_ANALYSIS.md` - Bug analysis

## Insights

`★ Insight ─────────────────────────────────────`
1. **Why inline workaround classes?** - To maintain test independence from Fractalic codebase
2. **Why direct SQLite access?** - DiskStore provides no API for expired entries
3. **Why separate cache directories?** - Prevents interference with production tokens
`─────────────────────────────────────────────────`
