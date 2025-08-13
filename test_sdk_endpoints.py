#!/usr/bin/env python3
"""Quick test script for SDK MCP manager endpoints"""

import asyncio
import json
import aiohttp
import subprocess
import time
import signal
import sys

async def test_endpoints():
    """Test all SDK endpoints"""
    
    # Start server
    print("Starting SDK MCP manager...")
    proc = subprocess.Popen([
        sys.executable, "fractalic_mcp_manager_sdk.py", 
        "serve", "--port", "5866"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        # Wait for server to start
        await asyncio.sleep(8)
        
        async with aiohttp.ClientSession() as session:
            # Test status endpoint
            print("Testing /status endpoint...")
            async with session.get("http://localhost:5866/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Status: success={data.get('success', False)}, services={len(data.get('services', []))}")
                else:
                    print(f"❌ Status failed: {resp.status}")
            
            # Test add_server endpoint
            print("Testing /add_server endpoint...")
            add_data = {
                "name": "test-sdk-server",
                "config": {"command": "echo", "args": ["test"]}
            }
            async with session.post("http://localhost:5866/add_server", 
                                   json=add_data) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Add server: success={data.get('success', False)}")
                else:
                    print(f"❌ Add server failed: {resp.status}")
            
            # Test delete_server endpoint  
            print("Testing /delete_server endpoint...")
            delete_data = {"name": "test-sdk-server"}
            async with session.post("http://localhost:5866/delete_server",
                                   json=delete_data) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Delete server: success={data.get('success', False)}")
                else:
                    print(f"❌ Delete server failed: {resp.status}")
                    
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        # Stop server
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("SDK test completed")

if __name__ == "__main__":
    asyncio.run(test_endpoints())
