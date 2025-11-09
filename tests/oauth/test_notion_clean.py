#!/usr/bin/env python3
"""
Clean Notion OAuth Test - No Fractalic modules
Uses only fastmcp + our OAuth workaround
"""
import asyncio
import time
import sys
from pathlib import Path

# Add current dir to path for oauth_helper import
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import Client
from mcp_manager.oauth_helper import create_custom_oauth_client

async def test_notion_clean():
    """Test Notion with clean OAuth setup"""

    print("=" * 70)
    print("CLEAN NOTION OAUTH TEST")
    print("=" * 70)

    # OAuth cache directory
    oauth_cache_dir = Path(__file__).parent / "oauth-cache"

    print(f"\n1. Creating OAuth client for Notion...")
    print(f"   OAuth cache: {oauth_cache_dir}")

    # Create OAuth with our workaround
    oauth = create_custom_oauth_client(
        mcp_url="https://mcp.notion.com/mcp",
        callback_port=58100
    )

    print(f"   ✓ OAuth client created")

    # Create FastMCP client
    client = Client("https://mcp.notion.com/mcp", auth=oauth)

    print(f"\n2. Connecting to Notion...")
    start = time.time()

    try:
        async with client:
            connect_time = time.time() - start
            print(f"   ✓ Connected in {connect_time:.2f}s")

            print(f"\n3. Listing tools...")
            tools_start = time.time()

            tools = await client.list_tools()
            tools = tools if isinstance(tools, list) else []

            tools_time = time.time() - tools_start
            print(f"   ✓ Got {len(tools)} tools in {tools_time:.2f}s")

            if tools:
                print(f"\n   First 5 tools:")
                for tool in tools[:5]:
                    print(f"     - {tool.name}")

            print(f"\n4. Listing prompts...")
            try:
                prompts = await client.list_prompts()
                prompts = prompts if isinstance(prompts, list) else []
                print(f"   ✓ Got {len(prompts)} prompts")
            except Exception as e:
                prompts = []
                print(f"   ⚠️  Prompts not supported: {e}")

            print(f"\n5. Listing resources...")
            try:
                resources = await client.list_resources()
                resources = resources if isinstance(resources, list) else []
                print(f"   ✓ Got {len(resources)} resources")
            except Exception as e:
                resources = []
                print(f"   ⚠️  Resources not supported: {e}")

            total_time = time.time() - start

            print("\n" + "=" * 70)
            print("RESULT")
            print("=" * 70)
            print(f"✅ SUCCESS")
            print(f"   Tools: {len(tools)}")
            print(f"   Prompts: {len(prompts)}")
            print(f"   Resources: {len(resources)}")
            print(f"   Total time: {total_time:.2f}s")
            print("=" * 70)

            return True

    except Exception as e:
        error_time = time.time() - start
        print(f"\n❌ ERROR after {error_time:.2f}s")
        print(f"   Error: {e}")
        print(f"   Type: {type(e).__name__}")

        import traceback
        print("\nTraceback:")
        traceback.print_exc()

        return False

if __name__ == "__main__":
    success = asyncio.run(test_notion_clean())
    sys.exit(0 if success else 1)
