#!/usr/bin/env python3
"""
Direct test of Replicate MCP service using official SDK
"""

import asyncio
import json
import logging
from pathlib import Path

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SimpleTokenStorage:
    """Simple token storage for testing"""
    
    def __init__(self, service_name: str = "replicate"):
        self.service_name = service_name
        self.tokens_file = Path("oauth_tokens.json")
    
    async def get_tokens(self) -> OAuthToken | None:
        """Load tokens from file"""
        if not self.tokens_file.exists():
            return None
        
        try:
            with open(self.tokens_file, 'r') as f:
                data = json.load(f)
            
            # Load service-specific token
            if self.service_name in data:
                token_data = data[self.service_name]
                return OAuthToken(
                    access_token=token_data['access_token'],
                    token_type=token_data.get('token_type', 'Bearer'),
                    expires_in=token_data.get('expires_in'),
                    refresh_token=token_data.get('refresh_token'),
                    scope=token_data.get('scope')
                )
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
        
        return None
    
    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Save tokens to file"""
        try:
            data = {}
            if self.tokens_file.exists():
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
            
            data[self.service_name] = {
                'access_token': tokens.access_token,
                'token_type': tokens.token_type,
                'expires_in': tokens.expires_in,
                'refresh_token': tokens.refresh_token,
                'scope': tokens.scope
            }
            
            with open(self.tokens_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Tokens saved for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
    
    async def get_client_info(self):
        return None
    
    async def set_client_info(self, client_info):
        pass

async def test_replicate_mcp():
    """Test Replicate MCP service directly"""
    
    # Configuration
    url = "https://mcp.replicate.com/sse"
    headers = {"MCP-Protocol-Version": "2025-06-18"}
    
    # Create OAuth provider with dummy handlers (we have existing tokens)
    token_storage = SimpleTokenStorage("replicate")
    
    async def dummy_redirect_handler(auth_url: str):
        logger.info(f"Would redirect to: {auth_url}")
    
    async def dummy_callback_handler():
        logger.info("Callback handler called")
        return ("dummy_code", "dummy_state")
    
    oauth_provider = OAuthClientProvider(
        server_url="https://mcp.replicate.com",
        client_metadata=OAuthClientMetadata(
            client_name="Test MCP Client",
            redirect_uris=[AnyUrl("http://localhost:5859/oauth/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="read write",
        ),
        storage=token_storage,
        redirect_handler=dummy_redirect_handler,
        callback_handler=dummy_callback_handler,
    )
    
    logger.info("Starting Replicate MCP test...")
    
    try:
        # Test with different timeout configurations
        timeouts_to_test = [
            (30.0, 60.0),    # 30s connection, 60s read
            (30.0, 120.0),   # 30s connection, 120s read  
            (60.0, 300.0),   # 60s connection, 300s read
        ]
        
        for connection_timeout, sse_read_timeout in timeouts_to_test:
            logger.info(f"Testing with timeouts: connection={connection_timeout}s, sse_read={sse_read_timeout}s")
            
            try:
                async with sse_client(
                    url,
                    auth=oauth_provider,
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream):
                    logger.info("✅ SSE connection established")
                    
                    async with ClientSession(read_stream, write_stream) as session:
                        logger.info("📡 Starting MCP session initialization...")
                        
                        # Initialize with timeout
                        await asyncio.wait_for(session.initialize(), timeout=30.0)
                        logger.info("✅ MCP session initialized successfully")
                        
                        # List tools with timeout
                        logger.info("📋 Requesting tools list...")
                        tools_result = await asyncio.wait_for(session.list_tools(), timeout=60.0)
                        
                        logger.info(f"✅ SUCCESS: Received {len(tools_result.tools)} tools")
                        
                        # Show first few tools
                        for i, tool in enumerate(tools_result.tools[:3]):
                            logger.info(f"  Tool {i+1}: {tool.name} - {tool.description[:100]}...")
                        
                        if len(tools_result.tools) > 3:
                            logger.info(f"  ... and {len(tools_result.tools) - 3} more tools")
                        
                        # Calculate token count (simplified)
                        tools_json = json.dumps([{
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        } for tool in tools_result.tools])
                        
                        estimated_tokens = len(tools_json) // 4  # Rough estimate
                        logger.info(f"📊 Estimated token count: {estimated_tokens}")
                        
                        return True
                        
            except asyncio.TimeoutError as e:
                logger.error(f"❌ Timeout with settings {connection_timeout}s/{sse_read_timeout}s: {e}")
                continue
            except Exception as e:
                logger.error(f"❌ Error with settings {connection_timeout}s/{sse_read_timeout}s: {e}")
                continue
        
        logger.error("❌ All timeout configurations failed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting direct MCP SDK test for Replicate service")
    
    success = await test_replicate_mcp()
    
    if success:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("💥 Test failed!")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
