#!/usr/bin/env python3
"""
Pure FastMCP Library Test
=========================
This test validates FastMCP library functionality WITHOUT using any Fractalic modules.

Goals:
1. Load MCP server configs from mcp_servers.json
2. Connect to OAuth-enabled servers using FastMCP Client
3. Use the OAuth auto-refresh workaround (NoTTLDiskStore + FixedExpiryOAuth)
4. List tools from each server
5. Report results

This test is completely standalone - it only uses:
- fastmcp library
- Standard Python libraries (json, pathlib, time, asyncio)
- The OAuth workaround classes (copied inline to avoid Fractalic imports)
"""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional, Dict, List

from fastmcp import Client
from fastmcp.client.auth import OAuth
from key_value.aio.stores.disk import DiskStore


# ============================================================================
# OAuth Workaround Classes (from oauth_workaround_complete.py)
# ============================================================================

class NoTTLDiskStore:
    """
    DiskStore wrapper that IGNORES TTL for .get() operations.

    Problem: DiskStore.get() returns None for expired entries
    Solution: Query SQLite directly, bypassing TTL check

    This allows refresh_token to be retrieved even when
    access_token has expired in the database.
    """

    def __init__(self, directory: str | Path):
        self._directory = Path(directory)
        self._store = DiskStore(directory=str(directory))

    async def get(self, key: str, collection: Optional[str] = None) -> Optional[Any]:
        """
        Get value from database IGNORING expire_time.

        Returns expired entries that DiskStore would normally filter out.
        """
        # Handle collection prefix
        if collection:
            key = f"{collection}::{key}"

        db_path = self._directory / "cache.db"
        if not db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Query WITHOUT checking expire_time
            cursor.execute(
                "SELECT value FROM Cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            # Parse nested JSON structure
            value_json = row[0]
            data = json.loads(value_json)
            return data.get('value')

        except Exception as e:
            print(f"  ⚠️  Error in NoTTLDiskStore.get: {e}")
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
        # Call parent to load tokens and client info
        await super()._initialize()

        # If we have tokens, get the ACTUAL expiry from database
        if self.context.current_tokens:
            try:
                # Get token cache key (same as TokenStorageAdapter)
                token_key = f"{self.server_base_url}/tokens"

                # Access underlying DiskStore to read expire_time
                disk_store = self.token_storage_adapter._key_value_store

                # Check if it's our NoTTLDiskStore
                if hasattr(disk_store, '_directory'):
                    db_path = disk_store._directory / "cache.db"
                else:
                    # Fallback for regular DiskStore
                    db_path = Path(disk_store.directory if hasattr(disk_store, 'directory') else str(disk_store._directory)) / "cache.db"

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
                            print(f"    ✓ OAuth token loaded: {time_remaining:.0f}s remaining")
                        else:
                            print(f"    ⚠️  OAuth token EXPIRED: {-time_remaining:.0f}s ago (will auto-refresh)")

                        return

            except Exception as e:
                print(f"    ⚠️  Could not read database expire_time: {e}")

            # Fallback to parent behavior if database read fails
            if self.context.current_tokens.expires_in:
                self.context.update_token_expiry(self.context.current_tokens)
                print(f"    ⚠️  Using fallback expiry calculation (may be incorrect)")


def create_oauth_with_auto_refresh(
    mcp_url: str,
    cache_dir: str | Path,
    callback_port: Optional[int] = None
) -> FixedExpiryOAuth:
    """
    Create OAuth client with working auto-refresh.

    Args:
        mcp_url: MCP server URL
        cache_dir: Directory for token storage
        callback_port: Fixed port for OAuth callback (optional)

    Returns:
        FixedExpiryOAuth instance with working auto-refresh
    """
    # Create NoTTLDiskStore for token storage
    token_storage = NoTTLDiskStore(directory=cache_dir)

    # Create FixedExpiryOAuth with NoTTLDiskStore
    oauth = FixedExpiryOAuth(
        mcp_url=mcp_url,
        token_storage=token_storage,
        callback_port=callback_port
    )

    return oauth


# ============================================================================
# Test Functions
# ============================================================================

def load_mcp_config(config_path: Path) -> Dict:
    """Load MCP server configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def get_oauth_servers(config: Dict) -> List[tuple]:
    """
    Extract ALL URL-based servers from config (OAuth and potentially OAuth).

    Returns:
        List of (name, url, callback_port) tuples
        callback_port may be None for servers without explicit OAuth config
    """
    oauth_servers = []

    for name, server_config in config.get("mcpServers", {}).items():
        # Check if server has URL (HTTP/HTTPS based, not stdio)
        if "url" in server_config:
            enabled = server_config.get("enabled", True)

            if enabled:
                # Get callback port if specified, otherwise None
                callback_port = server_config.get("oauth_callback_port", None)

                oauth_servers.append((
                    name,
                    server_config["url"],
                    callback_port
                ))

    return oauth_servers


def get_stdio_servers(config: Dict) -> List[tuple]:
    """
    Extract STDIO-based servers from config.

    Returns:
        List of (name, command, args, env) tuples
    """
    stdio_servers = []

    for name, server_config in config.get("mcpServers", {}).items():
        # Check if server has command (stdio-based, not URL)
        if "command" in server_config:
            enabled = server_config.get("enabled", True)

            if enabled:
                command = server_config["command"]
                args = server_config.get("args", [])
                env = server_config.get("env", {})

                stdio_servers.append((
                    name,
                    command,
                    args,
                    env
                ))

    return stdio_servers


async def test_server(
    name: str,
    url: str,
    callback_port: Optional[int],
    cache_dir: Path
) -> Dict:
    """
    Test a single MCP server (OAuth or potentially OAuth).

    Args:
        name: Server name
        url: Server URL
        callback_port: OAuth callback port (None for auto-detect)
        cache_dir: Directory for OAuth cache

    Returns:
        Dict with test results
    """
    result = {
        "name": name,
        "url": url,
        "success": False,
        "tools": [],
        "tool_count": 0,
        "error": None,
        "connect_time": 0,
        "total_time": 0,
        "oauth_type": "explicit" if callback_port else "auto"
    }

    start_time = time.time()

    try:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"URL: {url}")
        if callback_port:
            print(f"Callback Port: {callback_port} (explicit OAuth)")
        else:
            print(f"OAuth: Auto-detect (no callback port specified)")
        print(f"{'='*70}")

        # Create client with appropriate auth
        if callback_port:
            # Explicit OAuth with workaround
            print("\n  1. Creating OAuth client with auto-refresh workaround...")
            oauth = create_oauth_with_auto_refresh(
                mcp_url=url,
                cache_dir=cache_dir / name,
                callback_port=callback_port
            )
            client = Client(url, auth=oauth)
        else:
            # Try with auto OAuth (FastMCP will handle it)
            print("\n  1. Creating client with auto OAuth...")
            client = Client(url, auth="oauth")

        # Create FastMCP client
        print(f"  2. Connecting to {name}...")

        connect_start = time.time()

        async with client:
            result["connect_time"] = time.time() - connect_start
            print(f"    ✓ Connected in {result['connect_time']:.2f}s")

            # List tools - FIRST REQUEST
            print(f"  3. Listing tools (1st request)...")
            first_request_start = time.time()
            tools_response = await client.list_tools()
            first_request_time = time.time() - first_request_start

            # Handle different response types
            if isinstance(tools_response, list):
                tools = tools_response
            elif hasattr(tools_response, 'tools'):
                tools = tools_response.tools
            else:
                tools = []

            result["tools"] = [
                {"name": t.name if hasattr(t, 'name') else str(t)}
                for t in tools[:5]  # First 5 tools only
            ]
            result["tool_count"] = len(tools) if tools else 0
            result["success"] = True

            print(f"    ✓ Got {result['tool_count']} tools in {first_request_time:.3f}s")

            # List tools - SECOND REQUEST (test caching)
            print(f"  4. Listing tools (2nd request - testing cache)...")
            second_request_start = time.time()
            tools_response2 = await client.list_tools()
            second_request_time = time.time() - second_request_start

            print(f"    ✓ Got {result['tool_count']} tools in {second_request_time:.3f}s")

            # Calculate cache speedup
            if first_request_time > 0:
                speedup = ((first_request_time - second_request_time) / first_request_time) * 100
                result["first_request_time"] = first_request_time
                result["second_request_time"] = second_request_time
                result["cache_speedup_percent"] = speedup
                print(f"    📊 Cache speedup: {speedup:.1f}%")

            if result["tools"]:
                print(f"\n    First {min(5, len(result['tools']))} tools:")
                for tool in result["tools"]:
                    print(f"      - {tool['name']}")

        result["total_time"] = time.time() - start_time

        print(f"\n  ✅ SUCCESS: {name} - {result['tool_count']} tools in {result['total_time']:.2f}s")

    except Exception as e:
        result["error"] = str(e)
        result["total_time"] = time.time() - start_time
        print(f"\n  ❌ FAILED: {name} - {str(e)[:100]}")

    return result


async def test_stdio_server(
    name: str,
    command: str,
    args: List[str],
    env: Dict[str, str]
) -> Dict:
    """
    Test a single STDIO MCP server.

    Args:
        name: Server name
        command: Command to run (e.g., 'npx')
        args: Command arguments
        env: Environment variables

    Returns:
        Dict with test results
    """
    result = {
        "name": name,
        "command": f"{command} {' '.join(args)}",
        "success": False,
        "tools": [],
        "tool_count": 0,
        "error": None,
        "connect_time": 0,
        "total_time": 0,
        "server_type": "stdio"
    }

    start_time = time.time()

    try:
        print(f"\n{'='*70}")
        print(f"Testing: {name} (STDIO)")
        print(f"Command: {command} {' '.join(args)}")
        print(f"{'='*70}")

        # Create stdio client
        print(f"\n  1. Creating STDIO client with config...")

        from fastmcp import Client

        # Create config in FastMCP format
        config = {
            "mcpServers": {
                name: {
                    "transport": "stdio",
                    "command": command,
                    "args": args,
                    "env": env if env else {}
                }
            }
        }

        # Create client with config
        client = Client(config)

        print(f"  2. Connecting to {name}...")

        connect_start = time.time()

        async with client:
            result["connect_time"] = time.time() - connect_start
            print(f"    ✓ Connected in {result['connect_time']:.2f}s")

            # List tools - FIRST REQUEST
            print(f"  3. Listing tools (1st request)...")
            first_request_start = time.time()
            tools_response = await client.list_tools()
            first_request_time = time.time() - first_request_start

            # Handle different response types
            if isinstance(tools_response, list):
                tools = tools_response
            elif hasattr(tools_response, 'tools'):
                tools = tools_response.tools
            else:
                tools = []

            result["tools"] = [
                {"name": t.name if hasattr(t, 'name') else str(t)}
                for t in tools[:5]  # First 5 tools only
            ]
            result["tool_count"] = len(tools) if tools else 0
            result["success"] = True

            print(f"    ✓ Got {result['tool_count']} tools in {first_request_time:.3f}s")

            # List tools - SECOND REQUEST (test caching)
            print(f"  4. Listing tools (2nd request - testing cache)...")
            second_request_start = time.time()
            tools_response2 = await client.list_tools()
            second_request_time = time.time() - second_request_start

            print(f"    ✓ Got {result['tool_count']} tools in {second_request_time:.3f}s")

            # Calculate cache speedup
            if first_request_time > 0:
                speedup = ((first_request_time - second_request_time) / first_request_time) * 100
                result["first_request_time"] = first_request_time
                result["second_request_time"] = second_request_time
                result["cache_speedup_percent"] = speedup
                print(f"    📊 Cache speedup: {speedup:.1f}%")

            if result["tools"]:
                print(f"\n    First {min(5, len(result['tools']))} tools:")
                for tool in result["tools"]:
                    print(f"      - {tool['name']}")

        result["total_time"] = time.time() - start_time

        print(f"\n  ✅ SUCCESS: {name} - {result['tool_count']} tools in {result['total_time']:.2f}s")

    except Exception as e:
        result["error"] = str(e)
        result["total_time"] = time.time() - start_time
        print(f"\n  ❌ FAILED: {name} - {str(e)[:100]}")

    return result


async def main():
    """Main test function."""
    print("\n" + "="*70)
    print("PURE FASTMCP LIBRARY TEST")
    print("Testing ALL MCP servers (URL + STDIO)")
    print("="*70)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "mcp_servers.json"
    cache_dir = Path.home() / ".fastmcp" / "oauth-test-pure"

    print(f"\nConfig: {config_path}")
    print(f"Cache: {cache_dir}")

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"\nLoading MCP server configuration...")
    config = load_mcp_config(config_path)

    # Get URL-based servers
    oauth_servers = get_oauth_servers(config)
    print(f"\n📡 Found {len(oauth_servers)} URL-based servers:")
    for name, url, port in oauth_servers:
        port_str = f"port {port}" if port else "auto OAuth"
        print(f"  - {name} ({port_str})")

    # Get STDIO-based servers
    stdio_servers = get_stdio_servers(config)
    print(f"\n💻 Found {len(stdio_servers)} STDIO-based servers:")
    for name, command, args, env in stdio_servers:
        print(f"  - {name} ({command})")

    # Test all servers IN PARALLEL
    print(f"\n{'='*70}")
    print("TESTING ALL SERVERS IN PARALLEL")
    print(f"{'='*70}\n")

    # Create tasks for all servers
    tasks = []

    # Add URL-based server tasks
    for name, url, callback_port in oauth_servers:
        tasks.append(test_server(name, url, callback_port, cache_dir))

    # Add STDIO-based server tasks
    for name, command, args, env in stdio_servers:
        tasks.append(test_stdio_server(name, command, args, env))

    # Run all tests in parallel
    print(f"⏳ Starting {len(tasks)} tests in parallel...\n")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Create error result
            server_name = f"server_{i}"
            final_results.append({
                "name": server_name,
                "success": False,
                "error": str(result),
                "tool_count": 0,
                "total_time": 0
            })
        else:
            final_results.append(result)

    results = final_results

    # Print summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)

    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    total_tools = sum(r["tool_count"] for r in results if r["success"])

    print(f"\nOverall: {success_count}/{total_count} servers successful")
    print(f"Total tools available: {total_tools}\n")

    # Group results by type
    url_results = [r for r in results if r.get("oauth_type") or r.get("server_type") != "stdio"]
    stdio_results = [r for r in results if r.get("server_type") == "stdio"]

    if url_results:
        print("📡 URL-based servers:")
        for result in url_results:
            status = "✅" if result["success"] else "❌"
            tools = result["tool_count"]
            time_str = f"{result['total_time']:.2f}s"
            print(f"   {status} {result['name']:20} {tools:3} tools | {time_str:8}")
            if result["error"]:
                print(f"      Error: {result['error'][:60]}")

    if stdio_results:
        print("\n💻 STDIO-based servers:")
        for result in stdio_results:
            status = "✅" if result["success"] else "❌"
            tools = result["tool_count"]
            time_str = f"{result['total_time']:.2f}s"
            print(f"   {status} {result['name']:20} {tools:3} tools | {time_str:8}")
            if result["error"]:
                print(f"      Error: {result['error'][:60]}")

    # Save results to JSON
    results_file = project_root / "test_fastmcp_pure_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.time(),
            "summary": {
                "total": total_count,
                "success": success_count,
                "failed": total_count - success_count
            },
            "results": results
        }, f, indent=2)

    print(f"\n📊 Results saved to: {results_file}")
    print("\n" + "="*70)

    # Return exit code
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    import sys
    try:
        exit_code = asyncio.run(main())
        # Force immediate exit to avoid cleanup delays
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
