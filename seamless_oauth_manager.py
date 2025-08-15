#!/usr/bin/env python3
"""
Integrated OAuth Manager - Seamless token management with authlib
Provides automatic token refresh without user intervention
"""

import asyncio
import json
import logging
import time
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any

import aiohttp
from aiohttp import web
import requests
from authlib.oauth2.rfc6749.errors import OAuth2Error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeamlessOAuthManager:
    """
    OAuth manager that handles initial authorization and automatic token refresh
    without requiring user intervention after initial setup
    """
    
    def __init__(self, service_name: str, base_url: str, tokens_file: str = "oauth_tokens.json"):
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')
        self.tokens_file = Path(tokens_file)
        self.redirect_uri = "http://localhost:5860/oauth/callback"
        
        # OAuth endpoints (will be discovered)
        self.auth_endpoint = None
        self.token_endpoint = None
        
        # Client credentials (will be loaded or registered)
        self.client_id = None
        self.client_secret = None
        
        # Runtime state for authorization flow
        self.auth_code = None
        self.callback_received = asyncio.Event()
        self.callback_server = None
        
        # Token refresh settings
        self.token_refresh_threshold = 300  # Refresh when <5 minutes remaining
        self.last_refresh_attempt = 0
        self.refresh_cooldown = 60  # Don't retry refresh for 60 seconds
        
    def discover_oauth_endpoints(self) -> bool:
        """Discover OAuth 2.0 endpoints"""
        try:
            discovery_url = f"{self.base_url}/.well-known/oauth-authorization-server"
            response = requests.get(discovery_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.auth_endpoint = data.get('authorization_endpoint')
                self.token_endpoint = data.get('token_endpoint')
            else:
                # Fallback to standard endpoints
                self.auth_endpoint = f"{self.base_url}/authorize"
                self.token_endpoint = f"{self.base_url}/token"
            
            logger.info(f"OAuth endpoints for {self.service_name}:")
            logger.info(f"  Authorization: {self.auth_endpoint}")
            logger.info(f"  Token: {self.token_endpoint}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to discover OAuth endpoints for {self.service_name}: {e}")
            # Use fallback endpoints
            self.auth_endpoint = f"{self.base_url}/authorize"
            self.token_endpoint = f"{self.base_url}/token"
            return True
    
    def load_client_credentials(self) -> bool:
        """Load existing client credentials from tokens file"""
        try:
            if not self.tokens_file.exists():
                return False
            
            with open(self.tokens_file, 'r') as f:
                data = json.load(f)
                
            service_data = data.get(self.service_name, {})
            client_info = service_data.get('client_info', {})
            
            self.client_id = client_info.get('client_id')
            self.client_secret = client_info.get('client_secret')
            
            if self.client_id and self.client_secret:
                logger.info(f"Loaded client credentials for {self.service_name}: {self.client_id}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to load client credentials for {self.service_name}: {e}")
            return False
    
    def register_new_client(self) -> bool:
        """Register a new OAuth client"""
        try:
            registration_url = f"{self.base_url}/client-register"
            
            registration_data = {
                "client_name": f"Fractalic MCP Client - {self.service_name}",
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
                
                logger.info(f"Registered new client for {self.service_name}: {self.client_id}")
                
                # Save client credentials
                self.save_client_credentials(client_info)
                return True
            else:
                logger.error(f"Client registration failed for {self.service_name}: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register client for {self.service_name}: {e}")
            return False
    
    def save_client_credentials(self, client_info: Dict):
        """Save client credentials to tokens file"""
        try:
            # Load existing data
            data = {}
            if self.tokens_file.exists():
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
            
            # Initialize service data if not exists
            if self.service_name not in data:
                data[self.service_name] = {}
            
            # Update client info
            data[self.service_name]['client_info'] = client_info
            
            # Save to file
            with open(self.tokens_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Client credentials saved for {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to save client credentials for {self.service_name}: {e}")
    
    def load_tokens(self) -> Optional[Dict]:
        """Load existing access tokens"""
        try:
            if not self.tokens_file.exists():
                return None
            
            with open(self.tokens_file, 'r') as f:
                data = json.load(f)
                
            service_data = data.get(self.service_name, {})
            
            # Check if we have the required token fields
            if ('access_token' in service_data and 
                'refresh_token' in service_data and
                'obtained_at' in service_data):
                
                logger.info(f"Loaded existing tokens for {self.service_name}")
                return service_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load tokens for {self.service_name}: {e}")
            return None
    
    def save_tokens(self, tokens: Dict):
        """Save access tokens with timestamp"""
        try:
            # Load existing data
            data = {}
            if self.tokens_file.exists():
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
            
            # Initialize service data if not exists
            if self.service_name not in data:
                data[self.service_name] = {}
            
            # Add timestamp for expiry calculations
            tokens['obtained_at'] = time.time()
            
            # Update token info (preserve client_info)
            client_info = data[self.service_name].get('client_info', {})
            data[self.service_name].update(tokens)
            if client_info:
                data[self.service_name]['client_info'] = client_info
            
            # Save to file
            with open(self.tokens_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Tokens saved for {self.service_name}")
            logger.info(f"Access token: {tokens.get('access_token', 'N/A')[:20]}...")
            
        except Exception as e:
            logger.error(f"Failed to save tokens for {self.service_name}: {e}")
    
    def are_tokens_valid(self, tokens: Dict) -> bool:
        """Check if tokens are still valid (not expired)"""
        if not tokens or 'access_token' not in tokens:
            return False
        
        # If no expiry info, assume valid
        if 'expires_in' not in tokens or 'obtained_at' not in tokens:
            return True
        
        # Calculate remaining time
        age = time.time() - tokens['obtained_at']
        remaining = tokens['expires_in'] - age
        
        # Consider valid if more than threshold remaining
        return remaining > self.token_refresh_threshold
    
    def should_refresh_tokens(self, tokens: Dict) -> bool:
        """Check if tokens should be refreshed"""
        if not tokens or 'refresh_token' not in tokens:
            return False
        
        # Check cooldown
        if time.time() - self.last_refresh_attempt < self.refresh_cooldown:
            return False
        
        # If no expiry info, don't refresh
        if 'expires_in' not in tokens or 'obtained_at' not in tokens:
            return False
        
        # Calculate remaining time
        age = time.time() - tokens['obtained_at']
        remaining = tokens['expires_in'] - age
        
        # Refresh if close to expiry
        return remaining <= self.token_refresh_threshold
    
    async def refresh_tokens(self, tokens: Dict) -> Optional[Dict]:
        """Refresh access tokens using refresh token"""
        try:
            self.last_refresh_attempt = time.time()
            
            # Ensure we have OAuth endpoints
            if not self.token_endpoint:
                if not self.discover_oauth_endpoints():
                    logger.error(f"Failed to discover OAuth endpoints for {self.service_name}")
                    return None
            
            # Ensure we have client credentials
            if not self.client_id or not self.client_secret:
                if not self.load_client_credentials():
                    logger.error(f"No client credentials for token refresh: {self.service_name}")
                    return None
            
            refresh_data = {
                'grant_type': 'refresh_token',
                'refresh_token': tokens['refresh_token'],
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            logger.info(f"Refreshing tokens for {self.service_name}...")
            
            response = requests.post(self.token_endpoint, data=refresh_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                new_tokens = response.json()
                
                # Merge with existing tokens (preserve refresh_token if not provided)
                if 'refresh_token' not in new_tokens and 'refresh_token' in tokens:
                    new_tokens['refresh_token'] = tokens['refresh_token']
                
                self.save_tokens(new_tokens)
                logger.info(f"✅ Successfully refreshed tokens for {self.service_name}")
                return new_tokens
            else:
                logger.error(f"Token refresh failed for {self.service_name}: {response.status_code} {response.text}")
                
                # Handle specific error cases
                try:
                    error_data = response.json()
                    error_type = error_data.get('error', '')
                    
                    # If refresh token is invalid, clear tokens to trigger re-auth
                    if error_type in ['invalid_grant', 'invalid_request']:
                        logger.warning(f"Refresh token invalid for {self.service_name}, will require re-authorization")
                        return None
                        
                except:
                    pass
                
                return None
                
        except Exception as e:
            logger.error(f"Token refresh error for {self.service_name}: {e}")
            return None
    
    async def get_valid_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary"""
        try:
            # Load existing tokens
            tokens = self.load_tokens()
            
            if tokens:
                # Check if tokens are still valid
                if self.are_tokens_valid(tokens):
                    return tokens['access_token']
                
                # Try to refresh if needed and possible
                if self.should_refresh_tokens(tokens):
                    refreshed_tokens = await self.refresh_tokens(tokens)
                    if refreshed_tokens:
                        return refreshed_tokens['access_token']
                
                # If refresh failed, tokens might still be usable for a bit
                if 'access_token' in tokens:
                    logger.warning(f"Using potentially expired token for {self.service_name}")
                    return tokens['access_token']
            
            # No valid tokens available, need to re-authorize
            logger.info(f"No valid tokens for {self.service_name}, authorization required")
            return None
            
        except Exception as e:
            logger.error(f"Error getting access token for {self.service_name}: {e}")
            return None
    
    def get_bearer_header(self, access_token: str) -> Dict[str, str]:
        """Get Bearer authorization header"""
        return {"Authorization": f"Bearer {access_token}"}
    
    async def callback_handler(self, request):
        """Handle OAuth callback"""
        try:
            query = request.query
            
            if 'error' in query:
                logger.error(f"OAuth error for {self.service_name}: {query['error']} - {query.get('error_description', '')}")
                return web.Response(text="Authorization failed", status=400)
            
            if 'code' in query:
                self.auth_code = query['code']
                self.callback_received.set()
                logger.info(f"✅ Authorization code received for {self.service_name}")
                
                return web.Response(text="Authorization successful! You can close this window.", status=200)
            else:
                logger.error(f"No authorization code in callback for {self.service_name}")
                return web.Response(text="No authorization code received", status=400)
                
        except Exception as e:
            logger.error(f"Callback handler error for {self.service_name}: {e}")
            return web.Response(text="Internal error", status=500)
    
    async def exchange_code_for_tokens(self) -> Optional[Dict]:
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
                logger.info(f"✅ Token exchange successful for {self.service_name}!")
                self.save_tokens(tokens)
                return tokens
            else:
                logger.error(f"Token exchange failed for {self.service_name}: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Token exchange error for {self.service_name}: {e}")
            return None
    
    async def ensure_valid_tokens(self) -> Optional[str]:
        """Ensure we have valid tokens, performing full OAuth flow if necessary"""
        try:
            # First try to get valid tokens without user intervention
            access_token = await self.get_valid_access_token()
            if access_token:
                return access_token
            
            # Need to perform full OAuth flow
            logger.info(f"Performing OAuth authorization flow for {self.service_name}...")
            
            # Step 1: Ensure we have OAuth endpoints
            if not self.discover_oauth_endpoints():
                return None
            
            # Step 2: Ensure we have client credentials
            if not self.load_client_credentials():
                if not self.register_new_client():
                    logger.error(f"Failed to get client credentials for {self.service_name}")
                    return None
            
            # Step 3: Start callback server
            app = web.Application()
            app.router.add_get('/oauth/callback', self.callback_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, 'localhost', 5860)
            await site.start()
            self.callback_server = runner
            
            logger.info(f"OAuth callback server started for {self.service_name}")
            
            # Step 4: Build authorization URL
            auth_params = {
                'response_type': 'code',
                'client_id': self.client_id,
                'redirect_uri': self.redirect_uri,
                'scope': 'mcp',
                'state': f'{self.service_name}_auth_state'
            }
            
            auth_url = self.auth_endpoint + '?' + '&'.join([f"{k}={v}" for k, v in auth_params.items()])
            
            logger.info(f"Opening authorization URL for {self.service_name}: {auth_url}")
            webbrowser.open(auth_url)
            
            # Step 5: Wait for callback
            logger.info(f"Waiting for OAuth callback for {self.service_name} (complete authorization in your browser)...")
            
            try:
                await asyncio.wait_for(self.callback_received.wait(), timeout=120)
            except asyncio.TimeoutError:
                logger.error(f"OAuth callback timeout for {self.service_name} - authorization not completed")
                await runner.cleanup()
                return None
            
            # Step 6: Exchange code for tokens
            tokens = await self.exchange_code_for_tokens()
            
            # Cleanup callback server
            await runner.cleanup()
            self.callback_server = None
            
            if tokens and 'access_token' in tokens:
                logger.info(f"🎉 OAuth flow complete for {self.service_name}!")
                return tokens['access_token']
            else:
                logger.error(f"Failed to get access tokens for {self.service_name}")
                return None
                
        except Exception as e:
            logger.error(f"OAuth flow error for {self.service_name}: {e}")
            if self.callback_server:
                await self.callback_server.cleanup()
                self.callback_server = None
            return None

# Test function
async def test_seamless_oauth():
    """Test seamless OAuth with automatic refresh"""
    manager = SeamlessOAuthManager("replicate", "https://mcp.replicate.com")
    
    # Test getting valid token (will refresh or re-auth as needed)
    access_token = await manager.ensure_valid_tokens()
    
    if access_token:
        print(f"✅ Got valid access token for replicate: {access_token[:30]}...")
        
        # Test MCP connection with the token
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
            
            headers = manager.get_bearer_header(access_token)
            headers.update({
                "User-Agent": "MCP-Client/1.0",
                "MCP-Protocol-Version": "2025-06-18"
            })
            
            async with sse_client("https://mcp.replicate.com/sse", headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.list_tools()
                    tools = result.tools if hasattr(result, 'tools') else []
                    
                    print(f"🎉 Successfully retrieved {len(tools)} tools from Replicate MCP!")
                    return True
                    
        except Exception as e:
            print(f"❌ MCP connection failed: {e}")
            return False
    else:
        print(f"❌ Failed to get valid access token for replicate")
        return False

if __name__ == "__main__":
    asyncio.run(test_seamless_oauth())
