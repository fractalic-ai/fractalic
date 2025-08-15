#!/usr/bin/env python3
"""
Simple OAuth 2.0 Client using authlib
Bypasses MCP SDK OAuth provider issues
"""

import asyncio
import json
import logging
import secrets
import webbrowser
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

import aiohttp
from authlib.integrations.requests_client import OAuth2Session
from aiohttp import web

logger = logging.getLogger(__name__)

class SimpleOAuthClient:
    """Simple OAuth 2.0 client that actually works"""
    
    def __init__(self, service_name: str, server_url: str, callback_port: int = 5860):
        self.service_name = service_name
        self.server_url = server_url.rstrip('/')
        self.callback_port = callback_port
        self.redirect_uri = f"http://localhost:{callback_port}/oauth/callback"
        
        # Token storage
        self.token_file = Path("oauth_tokens_simple.json")
        
        # OAuth endpoints (will be discovered)
        self.auth_endpoint = None
        self.token_endpoint = None
        self.client_id = None
        self.client_secret = None
        
        # Callback state
        self.auth_code = None
        self.callback_received = asyncio.Event()
        
    async def discover_endpoints(self):
        """Discover OAuth endpoints from server"""
        try:
            async with aiohttp.ClientSession() as session:
                # Try OAuth authorization server metadata
                async with session.get(f"{self.server_url}/.well-known/oauth-authorization-server") as resp:
                    if resp.status == 200:
                        metadata = await resp.json()
                        self.auth_endpoint = metadata.get("authorization_endpoint")
                        self.token_endpoint = metadata.get("token_endpoint")
                        logger.info(f"Discovered OAuth endpoints: auth={self.auth_endpoint}, token={self.token_endpoint}")
                        return True
                        
        except Exception as e:
            logger.error(f"Failed to discover OAuth endpoints: {e}")
            return False
            
        return False
    
    async def register_client(self):
        """Register OAuth client with the server"""
        if not self.auth_endpoint or not self.token_endpoint:
            if not await self.discover_endpoints():
                raise Exception("Failed to discover OAuth endpoints")
        
        # Try to register client
        registration_url = f"{self.server_url}/.well-known/oauth-authorization-server"
        try:
            async with aiohttp.ClientSession() as session:
                # Get registration endpoint
                async with session.get(registration_url) as resp:
                    if resp.status == 200:
                        metadata = await resp.json()
                        reg_endpoint = metadata.get("registration_endpoint")
                        
                        if reg_endpoint:
                            # Register client
                            client_data = {
                                "client_name": f"Simple OAuth Client - {self.service_name}",
                                "redirect_uris": [self.redirect_uri],
                                "grant_types": ["authorization_code", "refresh_token"],
                                "response_types": ["code"]
                            }
                            
                            async with session.post(reg_endpoint, json=client_data) as reg_resp:
                                if reg_resp.status in (200, 201):
                                    reg_data = await reg_resp.json()
                                    self.client_id = reg_data.get("client_id")
                                    self.client_secret = reg_data.get("client_secret")
                                    logger.info(f"Client registered: {self.client_id}")
                                    return True
                                    
        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            
        # Fallback: try to load existing client info
        tokens = self.load_tokens()
        if tokens and "client_info" in tokens:
            client_info = tokens["client_info"]
            self.client_id = client_info.get("client_id")
            self.client_secret = client_info.get("client_secret")
            if self.client_id:
                logger.info(f"Using existing client: {self.client_id}")
                return True
                
        raise Exception("Failed to register or load OAuth client")
    
    def load_tokens(self) -> Optional[Dict]:
        """Load tokens from file"""
        if not self.token_file.exists():
            return None
            
        try:
            with open(self.token_file, 'r') as f:
                data = json.load(f)
                return data.get(self.service_name)
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            return None
    
    def save_tokens(self, tokens: Dict):
        """Save tokens to file"""
        try:
            data = {}
            if self.token_file.exists():
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
            
            data[self.service_name] = tokens
            
            with open(self.token_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Tokens saved for {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
    
    async def callback_handler(self, request):
        """Handle OAuth callback"""
        query = request.query
        
        if 'code' in query:
            self.auth_code = query['code']
            self.callback_received.set()
            logger.info("OAuth callback received with authorization code")
            
            return web.Response(
                text="""
                <!DOCTYPE html>
                <html><head><title>Authorization Complete</title></head>
                <body style="font-family: Arial; text-align: center; margin-top: 50px;">
                    <h2 style="color: green;">✅ Authorization Complete</h2>
                    <p>You can now close this window and return to the application.</p>
                </body></html>
                """,
                content_type='text/html'
            )
        
        elif 'error' in query:
            error = query['error']
            logger.error(f"OAuth error: {error}")
            self.callback_received.set()
            
            return web.Response(
                text=f"""
                <!DOCTYPE html>
                <html><head><title>Authorization Error</title></head>
                <body style="font-family: Arial; text-align: center; margin-top: 50px;">
                    <h2 style="color: red;">❌ Authorization Error</h2>
                    <p>Error: {error}</p>
                </body></html>
                """,
                content_type='text/html'
            )
        
        return web.Response(text="Invalid callback", status=400)
    
    async def start_callback_server(self):
        """Start callback server"""
        app = web.Application()
        app.router.add_get('/oauth/callback', self.callback_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.callback_port)
        await site.start()
        
        logger.info(f"OAuth callback server started on http://localhost:{self.callback_port}")
        return runner
    
    async def get_access_token(self) -> Optional[str]:
        """Get valid access token (handles refresh if needed)"""
        tokens = self.load_tokens()
        if not tokens:
            return await self.do_oauth_flow()
        
        access_token = tokens.get("access_token")
        if not access_token:
            return await self.do_oauth_flow()
        
        # TODO: Check if token is expired and refresh if needed
        # For now, just return the token
        return access_token
    
    async def do_oauth_flow(self) -> Optional[str]:
        """Perform complete OAuth flow"""
        try:
            # Step 1: Register client
            await self.register_client()
            
            # Step 2: Start callback server
            runner = await self.start_callback_server()
            
            try:
                # Step 3: Generate authorization URL
                auth_params = {
                    "response_type": "code",
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "scope": "read write",
                    "state": secrets.token_urlsafe(32)
                }
                
                # Add PKCE if supported
                code_verifier = secrets.token_urlsafe(32)
                code_challenge = code_verifier  # Simplified for now
                auth_params.update({
                    "code_challenge": code_challenge,
                    "code_challenge_method": "plain"
                })
                
                auth_url = f"{self.auth_endpoint}?" + "&".join([f"{k}={v}" for k, v in auth_params.items()])
                
                # Step 4: Open browser
                logger.info(f"Opening authorization URL: {auth_url}")
                webbrowser.open(auth_url)
                
                # Step 5: Wait for callback
                logger.info("Waiting for OAuth callback...")
                await asyncio.wait_for(self.callback_received.wait(), timeout=300)
                
                if not self.auth_code:
                    raise Exception("No authorization code received")
                
                # Step 6: Exchange code for tokens
                token_data = {
                    "grant_type": "authorization_code",
                    "code": self.auth_code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": code_verifier
                }
                
                if self.client_secret:
                    token_data["client_secret"] = self.client_secret
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.token_endpoint, data=token_data) as resp:
                        if resp.status == 200:
                            tokens = await resp.json()
                            
                            # Save tokens
                            token_info = {
                                "access_token": tokens["access_token"],
                                "token_type": tokens.get("token_type", "Bearer"),
                                "refresh_token": tokens.get("refresh_token"),
                                "expires_in": tokens.get("expires_in"),
                                "scope": tokens.get("scope"),
                                "client_info": {
                                    "client_id": self.client_id,
                                    "client_secret": self.client_secret
                                }
                            }
                            
                            self.save_tokens(token_info)
                            
                            logger.info("OAuth flow completed successfully!")
                            return tokens["access_token"]
                        else:
                            error_text = await resp.text()
                            raise Exception(f"Token exchange failed: {resp.status} {error_text}")
                
            finally:
                # Cleanup callback server
                await runner.cleanup()
                
        except Exception as e:
            logger.error(f"OAuth flow failed: {e}")
            return None
    
    async def get_bearer_header(self) -> Optional[Dict[str, str]]:
        """Get Bearer authorization header"""
        token = await self.get_access_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return None


# Test function
async def test_replicate_oauth():
    """Test OAuth flow with Replicate"""
    client = SimpleOAuthClient("replicate", "https://mcp.replicate.com")
    
    # Get access token (will trigger OAuth flow if needed)
    token = await client.get_access_token()
    
    if token:
        print(f"✅ Got access token: {token[:50]}...")
        
        # Test making an authenticated request
        bearer_header = await client.get_bearer_header()
        print(f"Bearer header: {bearer_header}")
        
        return True
    else:
        print("❌ Failed to get access token")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_replicate_oauth())
