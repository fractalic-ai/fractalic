#!/usr/bin/env python3
"""
PURE FastMCP OAuth Test (No Fractalic Modules)
==============================================
Clean test using ONLY FastMCP 2.13+ API according to official documentation.

Tests:
1. OAuth service (notion) - token persistence & refresh
2. Stdio service (chrome-devtools) - verify non-OAuth works
3. Timing measurements for cold start performance

NO Fractalic imports - only FastMCP + standard library.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# FastMCP imports - ONLY FastMCP, no Fractalic modules!
from fastmcp import Client
from fastmcp.client.auth import OAuth
from key_value.aio.stores.disk import DiskStore


async def test_oauth_service(service_name: str, url: str, oauth_cache_dir: Path) -> Dict[str, Any]:
    """Test OAuth service (notion) with timing."""
    print(f"\n{'='*60}")
    print(f"Testing OAuth Service: {service_name}")
    print(f"URL: {url}")
    print(f"OAuth cache: {oauth_cache_dir}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Create DiskStore for persistent token storage (FastMCP 2.13+)
        token_storage = DiskStore(directory=str(oauth_cache_dir))

        # Create OAuth client with persistent storage
        oauth = OAuth(
            mcp_url=url,
            token_storage=token_storage,
            callback_port=58100  # Fixed port
        )

        # Create FastMCP Client with OAuth
        client = Client(url, auth=oauth)

        connect_start = time.time()
        await client.__aenter__()
        connect_time = time.time() - connect_start
        print(f"✓ Connected in {connect_time:.2f}s")

        # Get tools
        tools_start = time.time()
        tools_response = await client.list_tools()
        tools_time = time.time() - tools_start

        # FastMCP returns list directly, not an object with .tools attribute
        if isinstance(tools_response, list):
            tools = tools_response
        elif hasattr(tools_response, 'tools'):
            tools = tools_response.tools
        else:
            tools = []

        tool_count = len(tools)

        print(f"✓ Got {tool_count} tools in {tools_time:.2f}s")

        # Print first 3 tool names
        if tools:
            print(f"  Sample tools:")
            for tool in tools[:3]:
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                print(f"    - {tool_name}")
        else:
            print(f"  ⚠️  Warning: 0 tools returned!")

        await client.__aexit__(None, None, None)

        total_time = time.time() - start_time

        return {
            'service': service_name,
            'type': 'oauth',
            'success': True,
            'tool_count': tool_count,
            'connect_time': connect_time,
            'tools_time': tools_time,
            'total_time': total_time
        }

    except Exception as e:
        total_time = time.time() - start_time
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        return {
            'service': service_name,
            'type': 'oauth',
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}",
            'total_time': total_time
        }


async def test_stdio_service(service_name: str, command: str, args: List[str]) -> Dict[str, Any]:
    """Test stdio service (chrome-devtools) with timing."""
    print(f"\n{'='*60}")
    print(f"Testing Stdio Service: {service_name}")
    print(f"Command: {command} {' '.join(args)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Create FastMCP Client for stdio using config dict
        # FastMCP Client accepts dict with command/args for stdio transport
        config = {
            "command": command,
            "args": args
        }
        client = Client(config)

        connect_start = time.time()
        await client.__aenter__()
        connect_time = time.time() - connect_start
        print(f"✓ Connected in {connect_time:.2f}s")

        # Get tools
        tools_start = time.time()
        tools_response = await client.list_tools()
        tools_time = time.time() - tools_start

        # FastMCP returns list directly, not an object with .tools attribute
        if isinstance(tools_response, list):
            tools = tools_response
        elif hasattr(tools_response, 'tools'):
            tools = tools_response.tools
        else:
            tools = []

        tool_count = len(tools)

        print(f"✓ Got {tool_count} tools in {tools_time:.2f}s")

        # Print first 3 tool names
        if tools:
            print(f"  Sample tools:")
            for tool in tools[:3]:
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                print(f"    - {tool_name}")

        await client.__aexit__(None, None, None)

        total_time = time.time() - start_time

        return {
            'service': service_name,
            'type': 'stdio',
            'success': True,
            'tool_count': tool_count,
            'connect_time': connect_time,
            'tools_time': tools_time,
            'total_time': total_time
        }

    except Exception as e:
        total_time = time.time() - start_time
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        return {
            'service': service_name,
            'type': 'stdio',
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}",
            'total_time': total_time
        }


async def main():
    """Run all tests."""
    print("=" * 70)
    print("PURE FASTMCP OAUTH TEST (No Fractalic modules)")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup OAuth cache directory
    oauth_cache_dir = Path(__file__).parent / "oauth-cache-test"
    oauth_cache_dir.mkdir(exist_ok=True)
    print(f"\nOAuth cache directory: {oauth_cache_dir}")

    # Load test config
    config_path = Path(__file__).parent / "mcp_servers.json"
    with open(config_path) as f:
        config = json.load(f)

    servers = config.get('mcpServers', {})
    print(f"Loaded {len(servers)} services from config")

    results = []
    overall_start = time.time()

    # Test each service
    for name, spec in servers.items():
        if not spec.get('enabled', True):
            print(f"\n⊘ Skipping disabled service: {name}")
            continue

        if 'url' in spec:
            # OAuth service
            result = await test_oauth_service(name, spec['url'], oauth_cache_dir)
        elif 'command' in spec:
            # Stdio service
            result = await test_stdio_service(name, spec['command'], spec.get('args', []))
        else:
            print(f"\n⊘ Skipping unknown service type: {name}")
            continue

        results.append(result)

    overall_time = time.time() - overall_start

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        service_info = f"{result['service']} ({result['type']})"

        print(f"\n{status}: {service_info}")

        if result['success']:
            print(f"  Tools: {result['tool_count']}")
            print(f"  Connect time: {result['connect_time']:.2f}s")
            print(f"  Tools time: {result['tools_time']:.2f}s")
            print(f"  Total time: {result['total_time']:.2f}s")
        else:
            print(f"  Error: {result['error']}")
            print(f"  Total time: {result['total_time']:.2f}s")

    print(f"\nOverall execution time: {overall_time:.2f}s")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Save results to JSON
    output_path = Path(__file__).parent / "test_pure_fastmcp_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_time': overall_time,
            'results': results
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
