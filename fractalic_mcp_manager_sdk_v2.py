#!/usr/bin/env python3
"""
MCP Manager using official Python SDK v1.12.4
Per-Request Session Pattern (No Persistent Connections)
"""

import asyncio
import json
import logging
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from aiohttp import web
from pydantic import AnyUrl

# MCP SDK imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    transport: str  # stdio, sse, http
    spec: Dict[str, Any]
    has_oauth: bool = False
    
    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> 'ServiceConfig':
        return cls(
            name=name,
            transport=config.get('transport', 'stdio'),
            spec=config,
            has_oauth=config.get('has_oauth', False)
        )

class FileTokenStorage(TokenStorage):
    """File-based token storage for OAuth"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
    
    async def get_tokens(self) -> Optional[OAuthToken]:
        """Load tokens from file"""
        if not self.file_path.exists():
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # Load default token (simple implementation)
            if 'default' in data:
                token_data = data['default']
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
            if self.file_path.exists():
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
            
            # Save as default token (simple implementation)
            data['default'] = {
                'access_token': tokens.access_token,
                'token_type': tokens.token_type,
                'expires_in': tokens.expires_in,
                'refresh_token': tokens.refresh_token,
                'scope': tokens.scope
            }
            
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("OAuth tokens saved successfully")
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
    
    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        return None
    
    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        pass

# Global callback data for OAuth
oauth_callback_data = {}

class MCPSupervisorV2:
    """MCP Service Manager using official Python SDK - Per-Request Sessions"""
    
    def __init__(self):
        self.config = self.load_config()
        self.oauth_providers: Dict[str, OAuthClientProvider] = {}
        self.token_storage = FileTokenStorage("oauth_tokens.json")
        self.service_states: Dict[str, str] = {}
        
        # Initialize OAuth providers for services that need them
        self._init_oauth_providers()
        
        # Set initial states
        for service_name in self.config:
            self.service_states[service_name] = "stopped"
    
    def load_config(self) -> Dict[str, ServiceConfig]:
        """Load MCP servers configuration"""
        config_file = Path("mcp_servers.json")
        if not config_file.exists():
            logger.warning(f"Config file {config_file} not found")
            return {}
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            services = {}
            for name, config in data.get('mcpServers', {}).items():
                services[name] = ServiceConfig.from_dict(name, config)
            
            logger.info(f"Loaded {len(services)} services from config")
            return services
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _init_oauth_providers(self):
        """Initialize OAuth providers for services that support OAuth"""
        for name, service in self.config.items():
            if service.has_oauth:
                provider = OAuthClientProvider(
                    server_url=service.spec.get('oauth_server_url', service.spec.get('url', '')),
                    client_metadata=OAuthClientMetadata(
                        client_name=f"Fractalic MCP Client - {name}",
                        redirect_uris=[AnyUrl("http://localhost:5859/oauth/callback")],
                        grant_types=["authorization_code", "refresh_token"],
                        response_types=["code"],
                        scope=service.spec.get('oauth_scope', 'read write'),
                    ),
                    storage=self.token_storage,
                    redirect_handler=self._handle_oauth_redirect,
                    callback_handler=self._handle_oauth_callback,
                )
                self.oauth_providers[name] = provider
    
    def _handle_oauth_redirect(self, auth_url: str) -> None:
        """Handle OAuth redirect by opening browser"""
        logger.info(f"Opening OAuth authorization URL: {auth_url}")
        
        def open_browser():
            webbrowser.open(auth_url)
        
        threading.Thread(target=open_browser, daemon=True).start()
        
    async def _handle_oauth_callback(self) -> tuple[str, str | None]:
        """Handle OAuth callback by waiting for the callback server"""
        # Clear any previous callback data
        oauth_callback_data.clear()
        
        # Wait for the callback with timeout
        timeout = 300  # 5 minutes
        logger.info("Waiting for OAuth callback...")
        
        for _ in range(timeout * 10):  # Check every 100ms
            if oauth_callback_data.get("received"):
                break
            await asyncio.sleep(0.1)
        else:
            logger.error("OAuth callback timeout after 5 minutes")
            return ("", None)
        
        if oauth_callback_data.get("error"):
            logger.error(f"OAuth error: {oauth_callback_data['error']}")
            return ("", None)
        
        if not oauth_callback_data.get("code"):
            logger.error("No authorization code received")
            return ("", None)
        
        logger.info("OAuth authorization code received successfully")
        return (oauth_callback_data["code"], oauth_callback_data.get("state"))
    
    async def _create_session_for_service(self, service: ServiceConfig) -> tuple[ClientSession, Any]:
        """Create a new session for a service (per-request pattern)"""
        oauth_provider = self.oauth_providers.get(service.name)
        
        if service.transport == "stdio":
            # STDIO transport
            server_params = StdioServerParameters(
                command=service.spec['command'],
                args=service.spec.get('args', []),
                env=service.spec.get('env', {})
            )
            connection = stdio_client(server_params)
            read_stream, write_stream = await connection.__aenter__()
            
        elif service.transport == "sse":
            # SSE transport with optional OAuth
            url = service.spec['url']
            connection = sse_client(url, auth=oauth_provider)
            read_stream, write_stream = await connection.__aenter__()
            
        elif service.transport == "http":
            # HTTP transport (streamable) with optional OAuth
            url = service.spec['url']
            connection = streamablehttp_client(url, auth=oauth_provider)
            read_stream, write_stream, _ = await connection.__aenter__()
            
        else:
            raise ValueError(f"Unsupported transport: {service.transport}")
        
        # Create and initialize session
        session = ClientSession(read_stream, write_stream)
        await session.initialize()
        
        return session, connection
    
    async def get_tools_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get tools for a service using per-request session"""
        if service_name not in self.config:
            return []
        
        service = self.config[service_name]
        
        try:
            session, connection = await self._create_session_for_service(service)
            try:
                # List tools
                tools_result = await session.list_tools()
                tools = []
                for tool in tools_result.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    })
                return tools
            finally:
                # Clean up connection
                if hasattr(connection, '__aexit__'):
                    await connection.__aexit__(None, None, None)
                    
        except Exception as e:
            logger.error(f"Failed to get tools for {service_name}: {e}")
            return []
    
    async def call_tool_for_service(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a service using per-request session"""
        if service_name not in self.config:
            raise ValueError(f"Service {service_name} not found")
        
        service = self.config[service_name]
        
        session, connection = await self._create_session_for_service(service)
        try:
            # Call tool
            result = await session.call_tool(tool_name, arguments)
            
            # Format response
            response = {
                "content": [],
                "isError": getattr(result, 'isError', False)
            }
            
            # Handle content
            for content in result.content:
                if hasattr(content, 'text'):
                    response["content"].append({
                        "type": "text",
                        "text": content.text
                    })
                elif hasattr(content, 'data'):
                    response["content"].append({
                        "type": "image",
                        "data": content.data,
                        "mimeType": getattr(content, 'mimeType', 'image/png')
                    })
            
            # Handle structured content
            if hasattr(result, 'structuredContent') and result.structuredContent:
                response["structuredContent"] = result.structuredContent
            
            return response
            
        finally:
            # Clean up connection
            if hasattr(connection, '__aexit__'):
                await connection.__aexit__(None, None, None)
    
    async def test_service_connection(self, service_name: str) -> bool:
        """Test connection to a service (no persistent sessions)"""
        if service_name not in self.config:
            return False
        
        service = self.config[service_name]
        try:
            session, connection = await self._create_session_for_service(service)
            try:
                # Test connection by listing tools
                await session.list_tools()
                self.service_states[service_name] = "running"
                logger.info(f"Service {service_name} connection tested successfully")
                return True
            finally:
                if hasattr(connection, '__aexit__'):
                    await connection.__aexit__(None, None, None)
        except Exception as e:
            self.service_states[service_name] = "error"
            logger.error(f"Failed to connect to {service_name}: {e}")
            return False
    
    async def status(self) -> Dict[str, Any]:
        """Get status of all services"""
        services = {}
        total_tools = 0
        
        for name, service in self.config.items():
            # Get current tools count by testing connection
            tools = await self.get_tools_for_service(name)
            tools_count = len(tools)
            total_tools += tools_count
            
            # Update state based on successful tool retrieval
            if tools_count > 0:
                state = "running"
                connected = True
                self.service_states[name] = "running"
            else:
                state = self.service_states.get(name, "stopped")
                connected = False
            
            services[name] = {
                "status": state,
                "connected": connected,
                "tools_count": tools_count,
                "has_oauth": service.has_oauth,
                "transport": service.transport
            }
        
        return {
            "services": services,
            "total_services": len(self.config),
            "running_services": sum(1 for s in services.values() if s["connected"]),
            "oauth_enabled": len(self.oauth_providers) > 0,
            "total_tools": total_tools
        }

# Global supervisor instance
supervisor = MCPSupervisorV2()

# OAuth callback handler for web server
async def oauth_callback_handler(request):
    """Handle OAuth callback from the authorization server"""
    global oauth_callback_data
    
    query = request.query
    
    if 'code' in query:
        oauth_callback_data.update({
            'received': True,
            'code': query['code'],
            'state': query.get('state'),
            'error': None
        })
        logger.info("OAuth authorization code received")
        
        return web.Response(
            text="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>OAuth Authorization Complete</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                    .success { color: #4CAF50; }
                    .container { max-width: 500px; margin: 0 auto; padding: 20px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="success">✅ Authorization Complete</h2>
                    <p>You have successfully authorized the application.</p>
                    <p>You can now close this window and return to the application.</p>
                </div>
            </body>
            </html>
            """,
            content_type='text/html'
        )
    
    elif 'error' in query:
        error = query['error']
        oauth_callback_data.update({
            'received': True,
            'code': None,
            'state': None,
            'error': error
        })
        logger.error(f"OAuth authorization error: {error}")
        
        return web.Response(
            text=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>OAuth Authorization Error</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                    .error {{ color: #f44336; }}
                    .container {{ max-width: 500px; margin: 0 auto; padding: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="error">❌ Authorization Error</h2>
                    <p>Authorization failed: {error}</p>
                    <p>Please try again or check your configuration.</p>
                </div>
            </body>
            </html>
            """,
            content_type='text/html'
        )
    
    else:
        oauth_callback_data.update({
            'received': True,
            'code': None,
            'state': None,
            'error': 'invalid_request'
        })
        
        return web.Response(
            text="Invalid OAuth callback request",
            status=400
        )

# REST API endpoints
async def status_handler(request):
    """GET /status - Get services status"""
    try:
        status = await supervisor.status()
        return web.json_response(status)
    except Exception as e:
        logger.error(f"Status error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def test_service_handler(request):
    """POST /test/{name} - Test service connection"""
    service_name = request.match_info['name']
    try:
        success = await supervisor.test_service_connection(service_name)
        return web.json_response({"success": success, "service": service_name})
    except Exception as e:
        logger.error(f"Test service error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def get_tools_handler(request):
    """GET /tools/{name} - Get tools for a service"""
    service_name = request.match_info['name']
    try:
        tools = await supervisor.get_tools_for_service(service_name)
        return web.json_response({"tools": tools, "count": len(tools)})
    except Exception as e:
        logger.error(f"Get tools error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def call_tool_handler(request):
    """POST /call/{service}/{tool} - Call a tool"""
    service_name = request.match_info['service']
    tool_name = request.match_info['tool']
    
    try:
        data = await request.json()
        arguments = data.get('arguments', {})
        
        result = await supervisor.call_tool_for_service(service_name, tool_name, arguments)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Call tool error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def oauth_start_handler(request):
    """POST /oauth/start/{service} - Start OAuth flow for a service"""
    service_name = request.match_info['service']
    
    if service_name not in supervisor.oauth_providers:
        return web.json_response({"error": "Service does not support OAuth"}, status=400)
    
    try:
        # OAuth flow will be triggered when the service is first accessed
        # For now, just test the service connection which will trigger OAuth if needed
        success = await supervisor.test_service_connection(service_name)
        return web.json_response({"oauth_started": True, "service": service_name, "success": success})
    except Exception as e:
        logger.error(f"OAuth start error: {e}")
        return web.json_response({"error": str(e)}, status=500)

def create_app():
    """Create aiohttp application"""
    app = web.Application()
    
    # API routes
    app.router.add_get('/status', status_handler)
    app.router.add_post('/test/{name}', test_service_handler)
    app.router.add_get('/tools/{name}', get_tools_handler)
    app.router.add_post('/call/{service}/{tool}', call_tool_handler)
    app.router.add_post('/oauth/start/{service}', oauth_start_handler)
    
    # OAuth callback route
    app.router.add_get('/oauth/callback', oauth_callback_handler)
    
    return app

async def serve():
    """Run the HTTP server"""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, 'localhost', 5859)
    await site.start()
    
    logger.info("MCP Manager V2 started on http://localhost:5859")
    logger.info("Per-request session pattern - no persistent connections")
    
    try:
        # Keep server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await runner.cleanup()

# CLI interface
async def cli_status():
    """CLI command to get status"""
    status = await supervisor.status()
    print(json.dumps(status, indent=2))

async def cli_test_service(service_name: str):
    """CLI command to test a service"""
    success = await supervisor.test_service_connection(service_name)
    print(f"Service {service_name}: {'✅ Connected' if success else '❌ Failed'}")

async def cli_get_tools(service_name: str):
    """CLI command to get tools"""
    tools = await supervisor.get_tools_for_service(service_name)
    print(f"Tools for {service_name}:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MCP Manager V2 - Per-Request Sessions')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Serve command
    subparsers.add_parser('serve', help='Start HTTP server')
    
    # Status command
    subparsers.add_parser('status', help='Get services status')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test service connection')
    test_parser.add_argument('service', help='Service name to test')
    
    # Tools command
    tools_parser = subparsers.add_parser('tools', help='Get tools for service')
    tools_parser.add_argument('service', help='Service name')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        asyncio.run(serve())
    elif args.command == 'status':
        asyncio.run(cli_status())
    elif args.command == 'test':
        asyncio.run(cli_test_service(args.service))
    elif args.command == 'tools':
        asyncio.run(cli_get_tools(args.service))
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
