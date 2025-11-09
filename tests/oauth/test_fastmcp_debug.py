#!/usr/bin/env python3
"""Debug version of FastMCP test - sequential with detailed logging"""

import asyncio
import json
from pathlib import Path
from fastmcp import Client

async def test_one_server(name: str, url: str):
    """Test single server with debug output"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    try:
        print(f"  1. Creating client...")
        client = Client(url)

        print(f"  2. Connecting...")
        async with client:
            print(f"  3. Connected! Calling list_tools()...")
            tools = await client.list_tools()
            print(f"  4. Got {len(tools)} tools")

            return {
                "name": name,
                "success": True,
                "tool_count": len(tools),
                "error": None
            }
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "name": name,
            "success": False,
            "tool_count": 0,
            "error": str(e)
        }

async def main():
    """Test servers SEQUENTIALLY"""
    print("DEBUG TEST - Sequential execution")
    print("="*60)

    # Load config
    config_path = Path("/Users/dima/fractalic/fractalic/mcp_servers.json")
    with open(config_path) as f:
        config = json.load(f)

    # Get URL servers
    servers = []
    for name, server_config in config.get("mcpServers", {}).items():
        if "url" in server_config and server_config.get("enabled", True):
            servers.append((name, server_config["url"]))

    print(f"\nFound {len(servers)} URL servers")
    for name, url in servers:
        print(f"  - {name}: {url}")

    # Test SEQUENTIALLY
    print("\n" + "="*60)
    print("STARTING SEQUENTIAL TESTS")
    print("="*60)

    results = []
    for i, (name, url) in enumerate(servers, 1):
        print(f"\n[{i}/{len(servers)}] Testing {name}...")
        result = await test_one_server(name, url)
        results.append(result)
        print(f"Result: {'✅ Success' if result['success'] else '❌ Failed'}")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    success = sum(1 for r in results if r["success"])
    print(f"\nSuccess: {success}/{len(results)}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} {r['name']:20} {r['tool_count']:3} tools")
        if r["error"]:
            print(f"   Error: {r['error'][:80]}")

    return 0

if __name__ == "__main__":
    import sys
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
        sys.exit(1)
