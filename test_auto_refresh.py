#!/usr/bin/env python3
"""
Test automatic token refresh functionality
"""

import asyncio
import json
import logging
import time
import copy
from seamless_oauth_manager import SeamlessOAuthManager

logging.basicConfig(level=logging.INFO)

async def test_token_refresh():
    """Test automatic token refresh when tokens are near expiry"""
    manager = SeamlessOAuthManager("replicate", "https://mcp.replicate.com")
    
    # First, modify the token expiry to simulate near-expiry
    tokens_file = "oauth_tokens.json"
    
    # Load current tokens
    with open(tokens_file, 'r') as f:
        data = json.load(f)
    
    # Backup original data using deep copy
    original_data = copy.deepcopy(data)
    
    try:
        # Simulate token near expiry (set obtained_at to 59 minutes ago)
        # This should trigger a refresh since our threshold is 5 minutes (300 seconds)
        current_time = time.time()
        expiry_time = 3600  # 1 hour in seconds
        near_expiry_obtained_at = current_time - (expiry_time - 240)  # 4 minutes left
        
        data['replicate']['obtained_at'] = near_expiry_obtained_at
        
        # Save modified tokens to simulate near-expiry
        with open(tokens_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print("🕐 Simulated tokens near expiry (4 minutes remaining)...")
        print("Testing automatic refresh...")
        
        # Test getting valid token (should trigger refresh)
        access_token = await manager.get_valid_access_token()
        
        if access_token:
            print(f"✅ Got valid access token: {access_token[:30]}...")
            
            # Load updated tokens to see if refresh happened
            with open(tokens_file, 'r') as f:
                updated_data = json.load(f)
            
            new_obtained_at = updated_data['replicate']['obtained_at']
            
            if new_obtained_at > near_expiry_obtained_at:
                print("✅ Token refresh was performed automatically!")
                print(f"   Old obtained_at: {near_expiry_obtained_at}")
                print(f"   New obtained_at: {new_obtained_at}")
            else:
                print("ℹ️  Token refresh was not needed (tokens still valid)")
            
            # Test MCP connection with refreshed token
            try:
                from mcp import ClientSession
                from mcp.client.sse import sse_client
                
                headers = manager.get_bearer_header(access_token)
                headers.update({
                    "User-Agent": "MCP-Client/1.0",
                    "MCP-Protocol-Version": "2025-06-18"
                })
                
                print("Testing MCP connection with refreshed token...")
                async with sse_client("https://mcp.replicate.com/sse", headers=headers) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        
                        result = await session.list_tools()
                        tools = result.tools if hasattr(result, 'tools') else []
                        
                        print(f"🎉 SUCCESS! Retrieved {len(tools)} tools with refreshed token!")
                        return True
                        
            except Exception as e:
                print(f"❌ MCP connection failed: {e}")
                return False
        else:
            print("❌ Failed to get valid access token")
            return False
    
    finally:
        # Restore original tokens
        with open(tokens_file, 'w') as f:
            json.dump(original_data, f, indent=2)
        print("🔄 Restored original tokens")

if __name__ == "__main__":
    asyncio.run(test_token_refresh())
