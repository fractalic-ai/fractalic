#!/usr/bin/env python3
"""
Test Replicate MCP without /sse suffix to use Streamable HTTP transport
"""
import asyncio
import time
from fastmcp.client import Client
import sys
sys.path.insert(0, '.')
from mcp_manager.oauth_helper import create_custom_oauth_client


async def test_replicate_without_sse():
    print("=" * 70)
    print("REPLICATE TEST - BASE URL (Streamable HTTP)")
    print("=" * 70)
    print()

    # Test with BASE URL (no /sse) to trigger StreamableHttpTransport
    mcp_url = "https://mcp.replicate.com"

    print(f"1. Creating OAuth client for Replicate (base URL)...")
    print(f"   URL: {mcp_url}")
    print(f"   OAuth cache: oauth-cache")
    start = time.time()

    try:
        oauth = create_custom_oauth_client(
            mcp_url=mcp_url,
            callback_port=58101
        )
        print(f"   ✓ OAuth client created\n")

        print(f"2. Connecting to Replicate...")
        client = Client(mcp_url, auth=oauth)

        connection_start = time.time()
        async with client:
            connection_time = time.time() - connection_start
            print(f"   ✓ Connected in {connection_time:.2f}s\n")

            print(f"3. Fetching tools...")
            tools_start = time.time()
            tools = await client.list_tools()
            tools_time = time.time() - tools_start

            tool_count = len(tools) if isinstance(tools, list) else len(tools.tools) if hasattr(tools, 'tools') else 0

            total_time = time.time() - start

            print()
            print(f"✅ SUCCESS in {total_time:.2f}s total")
            print(f"   Tools found: {tool_count}")
            print(f"   Connection: {connection_time:.2f}s")
            print(f"   List tools: {tools_time:.2f}s")

            if tool_count > 0:
                print(f"\n   First 3 tools:")
                tools_list = tools if isinstance(tools, list) else tools.tools
                for i, tool in enumerate(tools_list[:3], 1):
                    print(f"   {i}. {tool.name}: {tool.description[:60]}...")

    except Exception as e:
        elapsed = time.time() - start
        print()
        print(f"❌ ERROR after {elapsed:.2f}s")
        print(f"   Error: {e}")
        print(f"   Type: {type(e).__name__}")
        raise


if __name__ == "__main__":
    asyncio.run(test_replicate_without_sse())
