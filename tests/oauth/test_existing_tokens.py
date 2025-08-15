#!/usr/bin/env python3
"""
Test existing tokens functionality
"""

import asyncio
import logging
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from seamless_oauth_manager import SeamlessOAuthManager

logging.basicConfig(level=logging.INFO)

async def test_existing_tokens():
    """Test that existing tokens are recognized and used"""
    manager = SeamlessOAuthManager("replicate", "https://mcp.replicate.com")
    
    # Test getting valid token (should use existing tokens)
    print("Testing with existing tokens...")
    access_token = await manager.get_valid_access_token()
    
    if access_token:
        print(f"✅ Got valid access token: {access_token[:30]}...")
        
        # Test MCP connection
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
            
            headers = manager.get_bearer_header(access_token)
            headers.update({
                "User-Agent": "MCP-Client/1.0",
                "MCP-Protocol-Version": "2025-06-18"
            })
            
            print("Testing MCP connection...")
            async with sse_client("https://mcp.replicate.com/sse", headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.list_tools()
                    tools = result.tools if hasattr(result, 'tools') else []
                    
                    print(f"🎉 SUCCESS! Retrieved {len(tools)} tools using existing tokens!")
                    return True
                    
        except Exception as e:
            print(f"❌ MCP connection failed: {e}")
            return False
    else:
        print(f"❌ No valid access token found")
        return False

if __name__ == "__main__":
    asyncio.run(test_existing_tokens())
