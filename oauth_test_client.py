#!/usr/bin/env python3

"""
OAuth Test Client - Complete OAuth flow with MCP testing
Uses authlib for reliable OAuth 2.0 implementation
"""

import asyncio
import json
import logging
import webbrowser
from pathlib import Path
from urllib.parse import parse_qsl, urlparse
from typing import Optional, Dict, Any

import aiohttp
from aiohttp import web
import requests
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.oauth2.rfc6749.errors import OAuth2Error

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplicateOAuthClient:
    def __init__(self):
        self.base_url = "https://mcp.replicate.com"
        self.redirect_uri = "http://localhost:5860/oauth/callback"
        self.tokens_file = "oauth_tokens_simple.json"
        
        # Client info (will be loaded or registered)
        self.client_id = None
        self.client_secret = None
        
        # OAuth endpoints (will be discovered)
        self.auth_endpoint = None
        self.token_endpoint = None
        
        # Runtime state
        self.auth_code = None
        self.callback_received = asyncio.Event()
        
    def discover_oauth_endpoints(self):
        """Discover OAuth 2.0 endpoints"""
        try:
            # Try standard OAuth discovery
            discovery_url = f"{self.base_url}/.well-known/oauth-authorization-server"
            response = requests.get(discovery_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.auth_endpoint = data.get('authorization_endpoint')
                self.token_endpoint = data.get('token_endpoint')
            else:
                # Fallback to known endpoints
                self.auth_endpoint = f"{self.base_url}/authorize"
                self.token_endpoint = f"{self.base_url}/token"
            
            logger.info(f"OAuth endpoints discovered:")
            logger.info(f"  Authorization: {self.auth_endpoint}")
            logger.info(f"  Token: {self.token_endpoint}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to discover OAuth endpoints: {e}")
            # Use fallback endpoints
            self.auth_endpoint = f"{self.base_url}/authorize"
            self.token_endpoint = f"{self.base_url}/token"
            return True
    
    def load_existing_client(self):
        """Load existing client credentials"""
        try:
            # Try our simple tokens file first
            if Path(self.tokens_file).exists():
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
                    if 'client_id' in data and 'client_secret' in data:
                        self.client_id = data['client_id']
                        self.client_secret = data['client_secret']
                        logger.info(f"Loaded client from {self.tokens_file}: {self.client_id}")
                        return True
            
            # Try the main oauth_tokens.json file
            oauth_tokens_file = "oauth_tokens.json"
            if Path(oauth_tokens_file).exists():
                with open(oauth_tokens_file, 'r') as f:
                    data = json.load(f)
                    if 'replicate' in data:
                        replicate_data = data['replicate']
                        if 'client_info' in replicate_data:
                            client_info = replicate_data['client_info']
                            self.client_id = client_info.get('client_id')
                            self.client_secret = client_info.get('client_secret')
                            if self.client_id and self.client_secret:
                                logger.info(f"Loaded client from {oauth_tokens_file}: {self.client_id}")
                                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load existing client: {e}")
            return False
    
    def register_new_client(self):
        """Register a new OAuth client"""
        try:
            registration_url = f"{self.base_url}/client-register"
            
            registration_data = {
                "client_name": "Fractalic MCP Test Client",
                "redirect_uris": [self.redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "client_secret_post"
            }
            
            response = requests.post(registration_url, json=registration_data, timeout=30)
            
            if response.status_code == 201:
                client_info = response.json()
                self.client_id = client_info['client_id']
                self.client_secret = client_info['client_secret']
                
                logger.info(f"Registered new client: {self.client_id}")
                
                # Save client credentials
                self.save_client_info()
                return True
            else:
                logger.error(f"Client registration failed: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register client: {e}")
            return False
    
    def save_client_info(self):
        """Save client credentials"""
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'redirect_uri': self.redirect_uri
            }
            
            with open(self.tokens_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Client info saved to {self.tokens_file}")
            
        except Exception as e:
            logger.error(f"Failed to save client info: {e}")
    
    def save_tokens(self, tokens):
        """Save access tokens"""
        try:
            # Load existing data
            data = {}
            if Path(self.tokens_file).exists():
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
            
            # Add token info
            data.update({
                'access_token': tokens.get('access_token'),
                'refresh_token': tokens.get('refresh_token'),
                'token_type': tokens.get('token_type', 'Bearer'),
                'expires_in': tokens.get('expires_in'),
                'scope': tokens.get('scope')
            })
            
            with open(self.tokens_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Tokens saved to {self.tokens_file}")
            logger.info(f"Access token: {tokens.get('access_token', 'N/A')[:20]}...")
            
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
    
    def load_tokens(self):
        """Load existing access tokens"""
        try:
            if Path(self.tokens_file).exists():
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
                    if 'access_token' in data:
                        logger.info("Loaded existing access token")
                        return data
            return None
            
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            return None
    
    async def callback_handler(self, request):
        """Handle OAuth callback"""
        try:
            query = request.query
            
            if 'error' in query:
                logger.error(f"OAuth error: {query['error']} - {query.get('error_description', '')}")
                return web.Response(text="Authorization failed", status=400)
            
            if 'code' in query:
                self.auth_code = query['code']
                self.callback_received.set()
                logger.info("✅ Authorization code received")
                
                return web.Response(text="Authorization successful! You can close this window.", status=200)
            else:
                logger.error("No authorization code in callback")
                return web.Response(text="No authorization code received", status=400)
                
        except Exception as e:
            logger.error(f"Callback handler error: {e}")
            return web.Response(text="Internal error", status=500)
    
    async def exchange_code_for_tokens(self):
        """Exchange authorization code for access tokens"""
        try:
            token_data = {
                'grant_type': 'authorization_code',
                'code': self.auth_code,
                'redirect_uri': self.redirect_uri,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            response = requests.post(self.token_endpoint, data=token_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                tokens = response.json()
                logger.info("✅ Token exchange successful!")
                self.save_tokens(tokens)
                return tokens
            else:
                logger.error(f"Token exchange failed: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return None
    
    async def test_mcp_connection(self, access_token):
        """Test MCP connection with access token"""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "MCP-Client/1.0",
                "MCP-Protocol-Version": "2025-06-18"
            }
            
            logger.info("Testing MCP connection with Bearer token...")
            
            async with sse_client("https://mcp.replicate.com/sse", headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Request tools
                    result = await session.list_tools()
                    tools = result.tools if hasattr(result, 'tools') else []
                    
                    logger.info(f"✅ Successfully retrieved {len(tools)} tools from Replicate MCP!")
                    
                    if tools:
                        logger.info("First few tools:")
                        for i, tool in enumerate(tools[:5]):
                            logger.info(f"  {i+1}. {tool.name}: {tool.description}")
                    
                    return len(tools)
                    
        except Exception as e:
            logger.error(f"❌ MCP connection failed: {e}")
            return 0
    
    async def do_oauth_flow(self):
        """Complete OAuth flow"""
        try:
            # Step 1: Ensure we have OAuth endpoints
            if not self.discover_oauth_endpoints():
                return None
            
            # Step 2: Ensure we have client credentials
            if not self.load_existing_client():
                if not self.register_new_client():
                    logger.error("Failed to get client credentials")
                    return None
            
            # Step 3: Check for existing valid tokens
            existing_tokens = self.load_tokens()
            if existing_tokens and 'access_token' in existing_tokens:
                logger.info("Testing existing tokens...")
                tools_count = await self.test_mcp_connection(existing_tokens['access_token'])
                if tools_count > 0:
                    logger.info("✅ Existing tokens work! No need to reauthorize.")
                    return existing_tokens
                else:
                    logger.info("Existing tokens don't work, need to reauthorize...")
            
            # Step 4: Start callback server
            app = web.Application()
            app.router.add_get('/oauth/callback', self.callback_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, 'localhost', 5860)
            await site.start()
            
            logger.info("OAuth callback server started on http://localhost:5860")
            
            # Step 5: Build authorization URL
            auth_params = {
                'response_type': 'code',
                'client_id': self.client_id,
                'redirect_uri': self.redirect_uri,
                'scope': 'mcp',
                'state': 'test_state_123'
            }
            
            auth_url = self.auth_endpoint + '?' + '&'.join([f"{k}={v}" for k, v in auth_params.items()])
            
            logger.info(f"Opening authorization URL: {auth_url}")
            webbrowser.open(auth_url)
            
            # Step 6: Wait for callback
            logger.info("Waiting for OAuth callback (complete authorization in your browser)...")
            
            try:
                await asyncio.wait_for(self.callback_received.wait(), timeout=120)
            except asyncio.TimeoutError:
                logger.error("OAuth callback timeout - authorization not completed")
                await runner.cleanup()
                return None
            
            # Step 7: Exchange code for tokens
            tokens = await self.exchange_code_for_tokens()
            
            # Cleanup callback server
            await runner.cleanup()
            
            if tokens and 'access_token' in tokens:
                # Step 8: Test the tokens
                logger.info("Testing new tokens with MCP...")
                tools_count = await self.test_mcp_connection(tokens['access_token'])
                
                if tools_count > 0:
                    logger.info(f"🎉 OAuth flow complete! Successfully retrieved {tools_count} tools from Replicate MCP!")
                    return tokens
                else:
                    logger.error("Tokens received but MCP connection failed")
                    return None
            else:
                logger.error("Failed to get access tokens")
                return None
                
        except Exception as e:
            logger.error(f"OAuth flow error: {e}")
            return None

async def main():
    """Main test function"""
    client = ReplicateOAuthClient()
    
    logger.info("🚀 Starting Replicate OAuth flow test...")
    
    tokens = await client.do_oauth_flow()
    
    if tokens:
        logger.info("✅ OAuth flow completed successfully!")
        logger.info(f"Access token: {tokens.get('access_token', 'N/A')[:30]}...")
        logger.info(f"Token type: {tokens.get('token_type', 'N/A')}")
        logger.info(f"Expires in: {tokens.get('expires_in', 'N/A')} seconds")
    else:
        logger.error("❌ OAuth flow failed")

if __name__ == "__main__":
    asyncio.run(main())
