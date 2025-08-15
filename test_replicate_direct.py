#!/usr/bin/env python3
"""
Direct test of Replicate MCP connection using the latest SDK v1.13.0
This isolates the connection issue from our MCP manager implementation
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp.client.auth import OAuthClientProvider, OAuthToken

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTokenStorage:
    """Simple token storage for testing"""
    
    def __init__(self, file_path: str, service_name: str):
        self.file_path = Path(file_path)
        self.service_name = service_name
    
    async def get_tokens(self) -> OAuthToken | None:
        """Get tokens from file"""
        if not self.file_path.exists():
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            service_data = data.get(self.service_name, {})
            if not service_data or 'access_token' not in service_data:
                return None
                
            return OAuthToken(
                access_token=service_data['access_token'],
                refresh_token=service_data.get('refresh_token'),
                expires_in=service_data.get('expires_in', 3600),
                scope=service_data.get('scope')
            )
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            return None
    
    async def save_tokens(self, tokens: OAuthToken) -> None:
        """Save tokens - not implemented for test"""
        pass

async def test_replicate_connection():
    """Test direct connection to Replicate MCP server"""
    
    # Load tokens
    token_file = "/Users/marina/llexem-jan-25-deploy/llexem_deploy_2025/fractalic/oauth_tokens.json"
    storage = SimpleTokenStorage(token_file, "replicate")
    tokens = await storage.get_tokens()
    
    if not tokens:
        print("❌ No tokens found for replicate")
        return False
    
    print(f"✅ Loaded tokens, access_token starts with: {tokens.access_token[:20]}...")
    
    # Test connection without OAuth provider first (basic bearer auth)
    url = "https://mcp.replicate.com/sse"
    headers = {
        'MCP-Protocol-Version': '2025-06-18',
        'Authorization': f'Bearer {tokens.access_token}'
    }
    timeout = 30.0
    
    print(f"🔄 Connecting to {url} with timeout {timeout}s...")
    print("🔐 Using direct Bearer token authentication...")
    
    try:
        async with sse_client(url, headers=headers, timeout=timeout) as (read, write):
            print("✅ SSE connection established")
            
            async with ClientSession(read, write) as session:
                print("🔄 Initializing MCP session...")
                await session.initialize()
                print("✅ MCP session initialized")
                
                print("🔄 Listing tools...")
                tools_result = await session.list_tools()
                print("✅ Tools retrieved")
                
                tools = getattr(tools_result, 'tools', [])
                print(f"🎉 SUCCESS: Got {len(tools)} tools from replicate")
                
                for i, tool in enumerate(tools[:3]):
                    print(f"  Tool {i+1}: {tool.name} - {tool.description[:80]}...")
                
                return True
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🧪 Testing direct Replicate MCP connection...")
    print("📦 MCP SDK version: 1.13.0")
    print("🔗 Transport: SSE")
    print("🔐 Auth: OAuth with existing tokens")
    print("-" * 50)
    
    success = await test_replicate_connection()
    
    print("-" * 50)
    if success:
        print("🎉 Test PASSED: Replicate MCP connection working")
    else:
        print("💥 Test FAILED: Replicate MCP connection broken")

if __name__ == "__main__":
    asyncio.run(main())
