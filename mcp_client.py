#!/usr/bin/env python3
"""
MCP Manager HTTP Client
Pure HTTP client for status, test, tools, and kill commands
"""

import asyncio
import json
import argparse


async def cli_status(port: int = 5859):
    """CLI command to get status via HTTP"""
    import aiohttp
    
    url = f"http://localhost:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(json.dumps(data, indent=2))
                else:
                    print(f"❌ Status request failed: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to server at port {port}: {e}")
    except Exception as e:
        print(f"❌ Error getting status: {e}")


async def cli_test_service(service_name: str, port: int = 5859):
    """CLI command to test a service via HTTP"""
    import aiohttp
    
    url = f"http://localhost:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{url}/test/{service_name}") as response:
                if response.status == 200:
                    data = await response.json()
                    success = data.get("success", False)
                    print(f"Service {service_name}: {'✅ Connected' if success else '❌ Failed'}")
                else:
                    print(f"❌ Test request failed: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to server at port {port}: {e}")
    except Exception as e:
        print(f"❌ Error testing service: {e}")


async def cli_get_tools(service_name: str, port: int = 5859):
    """CLI command to get tools via HTTP"""
    import aiohttp
    
    url = f"http://localhost:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/tools/{service_name}") as response:
                if response.status == 200:
                    data = await response.json()
                    tools = data.get("tools", [])
                    print(f"Tools for {service_name}:")
                    for tool in tools:
                        print(f"  - {tool['name']}: {tool['description']}")
                else:
                    print(f"❌ Tools request failed: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to server at port {port}: {e}")
    except Exception as e:
        print(f"❌ Error getting tools: {e}")


async def cli_kill(port: int = 5859):
    """CLI command to kill running server via HTTP"""
    import aiohttp
    
    url = f"http://localhost:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{url}/kill") as response:
                if response.status == 200:
                    print("✅ Kill command sent successfully")
                else:
                    print(f"❌ Kill command failed: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to server at port {port}: {e}")
    except Exception as e:
        print(f"❌ Error sending kill command: {e}")


def main():
    """Main entry point for CLI HTTP client"""
    parser = argparse.ArgumentParser(description='MCP Manager HTTP Client')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get services status')
    status_parser.add_argument('--port', type=int, default=5859, help='Port of server to query (default: 5859)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test service connection')
    test_parser.add_argument('service', help='Service name to test')
    test_parser.add_argument('--port', type=int, default=5859, help='Port of server to query (default: 5859)')
    
    # Tools command
    tools_parser = subparsers.add_parser('tools', help='Get tools for service')
    tools_parser.add_argument('service', help='Service name')
    tools_parser.add_argument('--port', type=int, default=5859, help='Port of server to query (default: 5859)')
    
    # Kill command
    kill_parser = subparsers.add_parser('kill', help='Kill running server')
    kill_parser.add_argument('--port', type=int, default=5859, help='Port of server to kill (default: 5859)')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        asyncio.run(cli_status(args.port))
    elif args.command == 'test':
        asyncio.run(cli_test_service(args.service, args.port))
    elif args.command == 'tools':
        asyncio.run(cli_get_tools(args.service, args.port))
    elif args.command == 'kill':
        asyncio.run(cli_kill(args.port))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
