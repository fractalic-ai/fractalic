#!/usr/bin/env python3

"""
Test the OAuth tokens we just obtained
"""

import asyncio
import json
import logging
from mcp import ClientSession
from mcp.client.sse import sse_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_saved_tokens():
    """Test the saved OAuth tokens"""
    
    # Load tokens
    try:
        with open('oauth_tokens_simple.json', 'r') as f:
            tokens = json.load(f)
            
        access_token = tokens['access_token']
        logger.info(f"Testing access token: {access_token[:30]}...")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "MCP-Client/1.0",
            "MCP-Protocol-Version": "2025-06-18"
        }
        
        logger.info("Connecting to Replicate MCP with OAuth Bearer token...")
        
        async with sse_client("https://mcp.replicate.com/sse", headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Request tools
                result = await session.list_tools()
                tools = result.tools if hasattr(result, 'tools') else []
                
                logger.info(f"🎉 SUCCESS! Retrieved {len(tools)} tools from Replicate MCP!")
                
                if tools:
                    logger.info("Sample tools:")
                    for i, tool in enumerate(tools[:5]):
                        logger.info(f"  {i+1}. {tool.name}: {tool.description}")
                
                return len(tools)
                
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return 0

if __name__ == "__main__":
    tools_count = asyncio.run(test_saved_tokens())
    if tools_count > 0:
        print(f"\n✅ OAuth flow is working! Got {tools_count} tools from Replicate MCP!")
    else:
        print(f"\n❌ OAuth flow failed!")
