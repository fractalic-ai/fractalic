#!/usr/bin/env python3
"""
MCP Manager - SDK-Compatible Implementation
Maintains compatibility with Fractalic ecosystem while using MCP SDK
"""

import argparse
import os
import json
import asyncio
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pydantic import AnyUrl

# Core MCP SDK imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.shared.context import RequestContext
from mcp.shared.exceptions import McpError
from mcp.types import Tool, CallToolRequest, CallToolResult

# HTTP client (if available)
try:
    from mcp.client.streamable_http import streamablehttp_client
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    HTTP_CLIENT_AVAILABLE = False
    streamablehttp_client = None

# OAuth imports (MCP 1.12.4+)
try:
    from mcp.client.auth import OAuthClientProvider, TokenStorage
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    # Type stubs for when OAuth isn't available
    class OAuthClientProvider: pass
    class OAuthClientInformationFull: pass
    class OAuthClientMetadata: pass
    class OAuthToken: 
        def model_dump(self): return {}
    class TokenStorage: pass
    logging.warning("OAuth not available in this MCP version")

# Web server for management API
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OAuth token storage
OAUTH_STORAGE_PATH = os.getenv('OAUTH_STORAGE_PATH', '/tmp/mcp_oauth_tokens.json')

@dataclass
@dataclass
class ServiceConfig:
    """Minimal service configuration"""
    name: str
    transport: str  # stdio, sse, http
    spec: Dict[str, Any]
    has_oauth: bool = False
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None

class FileTokenStorage(TokenStorage):
    """File-based token storage implementation for OAuth tokens"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.environ.get('OAUTH_STORAGE_PATH', 'oauth_tokens.json')
    
    async def get_tokens(self) -> Optional[OAuthToken]:
        """Get stored tokens for any client"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    token_data = json.load(f)
                    if token_data:
                        # Return the first token found (simplified for now)
                        for client_id, data in token_data.items():
                            return OAuthToken(**data)
            return None
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            return None
    
    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store tokens (simplified - store under default key)"""
        try:
            os.makedirs(os.path.dirname(self.storage_path) if os.path.dirname(self.storage_path) else '.', exist_ok=True)
            
            # Load existing tokens
            existing = {}
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    existing = json.load(f)
            
            # Store under default key for now
            existing['default'] = tokens.model_dump()
            
            with open(self.storage_path, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to store tokens: {e}")
    
    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        """Get stored client information (not implemented for file storage)"""
        return None
    
    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information (not implemented for file storage)"""
        pass
    
    async def remove_tokens(self, client_id: str) -> None:
        """Remove tokens for client"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    token_data = json.load(f)
                
                if client_id in token_data:
                    del token_data[client_id]
                    
                    with open(self.storage_path, 'w') as f:
                        json.dump(token_data, f, indent=2)
                    
                    logger.info(f"Removed tokens for {client_id}")
        except Exception as e:
            logger.error(f"Failed to remove tokens for {client_id}: {e}")

class MCPSupervisor:
    """SDK-based MCP service supervisor using per-request sessions"""
    
    def __init__(self):
        # Remove persistent session storage - using per-request pattern
        self.service_states: Dict[str, str] = {}  # running, stopped, error
        self.token_storage = FileTokenStorage() if OAUTH_AVAILABLE else None
        self.oauth_providers: Dict[str, OAuthClientProvider] = {}
        
        # Load service configurations synchronously
        self.services_config: List[ServiceConfig] = self._load_config_sync()
        
        # Initialize OAuth providers for services that support OAuth
        self._init_oauth_providers()
    
    def _load_config_sync(self, config_path: str = "mcp_servers.json") -> List[ServiceConfig]:
        """Load service configurations synchronously"""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}")
            return []
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        services = []
        for name, spec in config_data.get('mcpServers', {}).items():
            # Determine transport type
            if 'command' in spec:
                transport = 'stdio'
            elif 'url' in spec:
                if spec.get('transport') == 'sse':
                    transport = 'sse'
                else:
                    transport = 'http'
            else:
                logger.warning(f"Cannot determine transport for {name}")
                continue
            
            has_oauth = spec.get('has_oauth', False)
            
            service = ServiceConfig(
                name=name,
                transport=transport,
                spec=spec,
                has_oauth=has_oauth
            )
            services.append(service)
            logger.info(f"Loaded service: {name} ({transport})")
        
        logger.info(f"Loaded {len(services)} services total")
        return services
    
    def _init_oauth_providers(self):
        """Initialize OAuth providers for services that support OAuth"""
        if not OAUTH_AVAILABLE:
            return
        
        for service in self.services_config:
            if service.has_oauth:
                provider = OAuthClientProvider(
                    server_url=service.spec.get('oauth_server_url', service.spec.get('url', '')),
                    client_metadata=OAuthClientMetadata(
                        client_name=f"Fractalic MCP Client - {service.name}",
                        redirect_uris=[AnyUrl("http://localhost:5859/oauth/callback")],
                        grant_types=["authorization_code", "refresh_token"],
                        response_types=["code"],
                        scope=service.spec.get('oauth_scope', 'read write'),
                    ),
                    storage=self.token_storage,
                    redirect_handler=self._handle_oauth_redirect,
                    callback_handler=self._handle_oauth_callback,
                )
                self.oauth_providers[service.name] = provider
                logger.info(f"Initialized OAuth provider for {service.name}")
        
    async def setup_oauth_provider(self, service: ServiceConfig) -> Optional[OAuthClientProvider]:
        """Setup OAuth provider for service if needed"""
        if not OAUTH_AVAILABLE:
            return None
        
        # Setup OAuth for HTTP or SSE services that might require authentication
        if service.transport in ['http', 'sse'] and 'url' in service.spec:
            url = service.spec['url']
            # Check if this looks like an OAuth-enabled service (including replicate.com)
            if any(domain in url for domain in ['zapier.com', 'oauth', 'auth', 'replicate.com']):
                try:
                    # Create OAuth provider with correct MCP SDK classes
                    provider = OAuthClientProvider(
                        server_url=url,
                        client_metadata=OAuthClientMetadata(
                            client_name=f"Fractalic MCP Client - {service.name}",
                            redirect_uris=["http://localhost:5859/oauth/callback"],
                            grant_types=["authorization_code", "refresh_token"],
                            response_types=["code"],
                            scope="read write"
                        ),
                        storage=self.token_storage,
                        redirect_handler=self._handle_oauth_redirect,
                        callback_handler=self._handle_oauth_callback
                    )
                    
                    self.oauth_providers[service.name] = provider
                    logger.info(f"OAuth provider setup for {service.name}")
                    return provider
                    
                except Exception as e:
                    logger.error(f"Failed to setup OAuth for {service.name}: {e}")
                    return None
        
        return None
    
    async def _handle_oauth_redirect(self, auth_url: str) -> None:
        """Handle OAuth redirect by opening browser"""
        import webbrowser
        import threading
        
        logger.info(f"Opening OAuth authorization URL in browser: {auth_url}")
        
        # Start a simple HTTP server to catch the callback
        self._oauth_callback_received = False
        self._oauth_callback_data = {}
        
        # Open the authorization URL in the default browser
        def open_browser():
            webbrowser.open(auth_url)
        
        # Run browser opening in a separate thread to avoid blocking
        threading.Thread(target=open_browser, daemon=True).start()
        
    async def _handle_oauth_callback(self) -> tuple[str, str | None]:
        """Handle OAuth callback by waiting for the callback server"""
        import asyncio
        
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
    
    async def start(self, service_name: str = None) -> bool:
        """Test service connections (no persistent sessions)"""
        if service_name:
            # Test specific service
            service = next((s for s in self.services_config if s.name == service_name), None)
            if not service:
                logger.error(f"Service not found: {service_name}")
                return False
            
            return await self._start_service(service)
        else:
            # Test all services (don't fail if some fail)
            success_count = 0
            for service in self.services_config:
                try:
                    if await self._start_service(service):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to test {service.name}: {e}")
            
            logger.info(f"Tested {success_count}/{len(self.services_config)} services successfully")
            return True  # Always return True so web server starts
    
    async def _start_service(self, service: ServiceConfig) -> bool:
        """Start a single service using appropriate SDK client"""
        try:
            # Setup OAuth if needed
            oauth_provider = await self.setup_oauth_provider(service)
            
            if service.transport == 'stdio':
                await self._connect_stdio(service, oauth_provider)
            elif service.transport == 'sse':
                await self._connect_sse(service, oauth_provider)
            elif service.transport == 'http':
                await self._connect_http(service, oauth_provider)
            else:
                logger.error(f"Unsupported transport: {service.transport}")
                return False
                
            logger.info(f"Started {service.name} via {service.transport}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {service.name}: {e}")
            return False
    
    async def stop(self, service_name: Optional[str] = None) -> bool:
        """Mark services as stopped (no persistent sessions to close)"""
        if service_name:
            if service_name in self.service_states:
                self.service_states[service_name] = "stopped"
                logger.info(f"Marked {service_name} as stopped")
                return True
            return False
        else:
            # Mark all as stopped
            for service in self.services_config:
                self.service_states[service.name] = "stopped"
            logger.info("Marked all services as stopped")
            return True

    async def restart(self, service_name: str) -> bool:
        """Restart a service (test connection)"""
        await self.stop(service_name)
        return await self._start_service(next((s for s in self.services_config if s.name == service_name), None))

    async def disconnect_all(self):
        """Mark all services as disconnected (no persistent sessions)"""
        for service in self.services_config:
            self.service_states[service.name] = "stopped"
        logger.info("Marked all services as disconnected")
    
    async def status(self) -> Dict[str, Any]:
        """Get status using per-request sessions"""
        services = {}
        total_tools = 0
        
        for service in self.services_config:
            # Get current tools count by testing connection
            tools = await self.get_tools_for_service(service.name)
            tools_count = len(tools)
            total_tools += tools_count
            
            # Update state based on successful tool retrieval
            if tools_count > 0:
                state = "running"
                connected = True
                self.service_states[service.name] = "running"
            else:
                state = self.service_states.get(service.name, "stopped")
                connected = False
            
            services[service.name] = {
                "status": state,
                "connected": connected,
                "tools_count": tools_count,
                "has_oauth": service.name in self.oauth_providers,
                "transport": service.transport
            }
        
        return {
            "services": services,
            "total_services": len(self.services_config),
            "running_services": sum(1 for s in services.values() if s["connected"]),
            "oauth_enabled": OAUTH_AVAILABLE,
            "total_tools": total_tools
        }
    
    async def tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get tools from all services using per-request sessions"""
        all_tools = {}
        
        for service in self.services_config:
            tools = await self.get_tools_for_service(service.name)
            all_tools[service.name] = tools
        
        return all_tools
    
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
            
        elif service.transport == "sse":
            # SSE transport with optional OAuth
            url = service.spec['url']
            connection = sse_client(url, auth=oauth_provider)
            
        elif service.transport == "http":
            # HTTP transport (streamable) with optional OAuth
            url = service.spec['url']
            connection = streamablehttp_client(url, auth=oauth_provider)
            
        else:
            raise ValueError(f"Unsupported transport: {service.transport}")
        
        # Use async context manager properly
        if service.transport == "http":
            read_stream, write_stream, _ = await connection.__aenter__()
        else:
            read_stream, write_stream = await connection.__aenter__()
        
        # Create and initialize session
        session = ClientSession(read_stream, write_stream)
        await session.initialize()
        
        return session, connection

    async def _start_service(self, service: ServiceConfig) -> bool:
        """Test service connection (no persistent sessions)"""
        try:
            session, connection = await self._create_session_for_service(service)
            try:
                # Test connection by listing tools
                await session.list_tools()
                self.service_states[service.name] = "running"
                logger.info(f"Service {service.name} connection tested successfully")
                return True
            finally:
                # Clean up connection
                if hasattr(connection, '__aexit__'):
                    await connection.__aexit__(None, None, None)
        except Exception as e:
            self.service_states[service.name] = "error"
            logger.error(f"Failed to connect to {service.name}: {e}")
            return False
    
    async def _connect_sse(self, service: ServiceConfig, oauth_provider: Optional[OAuthClientProvider]):
        """Connect via SSE transport"""
        url = service.spec['url']
        
        try:
            # Use the MCP SDK SSE client with OAuth as context manager
            connection = sse_client(url, auth=oauth_provider)
            
            # Start the connection in a task so it doesn't block
            async def connect_task():
                async with connection as (read_stream, write_stream):
                    # Create client session
                    session = ClientSession(read_stream, write_stream)
                    await session.initialize()
                    
                    # Store the session
                    self.sessions[service.name] = session
                    self.service_states[service.name] = "running"
                    logger.info(f"Connected {service.name} via SSE with auth: {oauth_provider is not None}")
                    
                    # Keep connection alive
                    try:
                        while service.name in self.sessions:
                            await asyncio.sleep(1)
                    except asyncio.CancelledError:
                        logger.info(f"SSE connection for {service.name} cancelled")
                        raise
            
            # Store the task to manage the connection
            task = asyncio.create_task(connect_task())
            self.clients[service.name] = task
            
            # Wait a bit for connection to establish
            await asyncio.sleep(0.5)
            
            # Check if connection was successful
            if service.name not in self.sessions:
                self.service_states[service.name] = "failed"
                raise Exception("Failed to establish SSE connection")
            
        except Exception as e:
            self.service_states[service.name] = "failed"
            logger.error(f"Failed to connect {service.name} via SSE: {e}")
            raise
    
    async def _connect_http(self, service: ServiceConfig, oauth_provider: Optional[OAuthClientProvider]):
        """Connect via HTTP transport"""
        if not HTTP_CLIENT_AVAILABLE:
            raise RuntimeError("HTTP client not available in this MCP version")
            
        url = service.spec['url']
        
        try:
            # Use the MCP SDK streamable HTTP client
            connection = streamablehttp_client(url, auth=oauth_provider)
            read_stream, write_stream, get_session_id = await connection.__aenter__()
            
            # Create client session
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            
            # Store the connection and session
            self.clients[service.name] = connection
            self.sessions[service.name] = session
            
            self.service_states[service.name] = "running"
            logger.info(f"Connected to {service.name} via HTTP")
            
        except Exception as e:
            logger.error(f"Failed to connect to {service.name} via HTTP: {e}")
            self.service_states[service.name] = "error"
            # Don't raise - just mark as error state
    
    async def list_tools(self, service_name: str = None) -> Dict[str, List[Tool]]:
        """List tools from services"""
        if service_name:
            if service_name not in self.sessions:
                return {}
            
            try:
                session = self.sessions[service_name]
                result = await session.list_tools()
                tools = result.tools if hasattr(result, 'tools') else []
                self.tools_cache[service_name] = tools
                return {service_name: tools}
            except Exception as e:
                logger.error(f"Failed to list tools from {service_name}: {e}")
                return {}
        
        # List from all services
        all_tools = {}
        for name, session in self.sessions.items():
            try:
                result = await session.list_tools()
                tools = result.tools if hasattr(result, 'tools') else []
                all_tools[name] = tools
                self.tools_cache[name] = tools
            except Exception as e:
                logger.error(f"Failed to list tools from {name}: {e}")
                all_tools[name] = []
        
        return all_tools
    
    async def get_tools_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get tools for a service using per-request session"""
        service = next((s for s in self.services_config if s.name == service_name), None)
        if not service:
            return []
        
        try:
            # Add timeout to prevent hanging
            return await asyncio.wait_for(self._get_tools_with_timeout(service), timeout=10)
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting tools for {service_name}")
            return []
        except Exception as e:
            logger.error(f"Failed to get tools for {service_name}: {e}")
            return []
    
    async def _get_tools_with_timeout(self, service):
        """Helper method to get tools with proper cleanup"""
        oauth_provider = self.oauth_providers.get(service.name)
        
        if service.transport == "stdio":
            # STDIO transport
            server_params = StdioServerParameters(
                command=service.spec['command'],
                args=service.spec.get('args', []),
                env=service.spec.get('env', {})
            )
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    return [{"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema} for tool in tools_result.tools]
                    
        elif service.transport == "sse":
            # SSE transport with optional OAuth
            url = service.spec['url']
            async with sse_client(url, auth=oauth_provider) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    return [{"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema} for tool in tools_result.tools]
                    
        elif service.transport == "http":
            # HTTP transport (streamable) with optional OAuth
            url = service.spec['url']
            async with streamablehttp_client(url, auth=oauth_provider) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    return [{"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema} for tool in tools_result.tools]
        
        else:
            raise ValueError(f"Unsupported transport: {service.transport}")
        
        return []

    async def call_tool(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a service using per-request session"""
        service = next((s for s in self.services_config if s.name == service_name), None)
        if not service:
            raise ValueError(f"Service {service_name} not found")
        
        oauth_provider = self.oauth_providers.get(service.name)
        
        if service.transport == "stdio":
            # STDIO transport
            server_params = StdioServerParameters(
                command=service.spec['command'],
                args=service.spec.get('args', []),
                env=service.spec.get('env', {})
            )
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await self._format_tool_result(await session.call_tool(tool_name, arguments))
                    
        elif service.transport == "sse":
            # SSE transport with optional OAuth
            url = service.spec['url']
            async with sse_client(url, auth=oauth_provider) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await self._format_tool_result(await session.call_tool(tool_name, arguments))
                    
        elif service.transport == "http":
            # HTTP transport (streamable) with optional OAuth
            url = service.spec['url']
            async with streamablehttp_client(url, auth=oauth_provider) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await self._format_tool_result(await session.call_tool(tool_name, arguments))
        
        else:
            raise ValueError(f"Unsupported transport: {service.transport}")
    
    async def _format_tool_result(self, result):
        """Format tool call result to match expected API"""
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
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {
            "connected_services": list(self.sessions.keys()),
            "oauth_enabled": OAUTH_AVAILABLE,
            "total_tools": sum(len(tools) for tools in self.tools_cache.values()),
            "services": {}
        }
        
        for name in self.sessions.keys():
            status["services"][name] = {
                "connected": name in self.sessions,
                "tools_count": len(self.tools_cache.get(name, [])),
                "has_oauth": name in self.oauth_providers
            }
        
        return status
    
    async def disconnect_all(self):
        """Disconnect all services"""
        for name in list(self.sessions.keys()):
            try:
                # Cancel SSE tasks if they exist
                if name in self.clients:
                    client = self.clients[name]
                    if isinstance(client, asyncio.Task):
                        client.cancel()
                        try:
                            await client
                        except asyncio.CancelledError:
                            pass
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
        
        self.sessions.clear()
        self.clients.clear()
        self.tools_cache.clear()
        
        # Reset all service states
        for name in self.service_states:
            self.service_states[name] = "stopped"

# Global supervisor instance
supervisor = MCPSupervisor()

# REST API handlers compatible with original
async def _json_response(data):
    """Return JSON response"""
    return web.json_response(data)

async def _status_handler(request):
    """GET /status - Compatible with original API"""
    status = await supervisor.status()
    return await _json_response(status)

async def _tools_handler(request):
    """GET /tools and /list_tools - Compatible with original API"""
    tools = await supervisor.tools()
    return await _json_response(tools)

async def _start_handler(request):
    """POST /start/{name} - Start specific service"""
    service_name = request.match_info.get('n')
    success = await supervisor.start(service_name)
    return await _json_response({"success": success})

async def _stop_handler(request):
    """POST /stop/{name} - Stop specific service"""
    service_name = request.match_info.get('n')
    success = await supervisor.stop(service_name)
    return await _json_response({"success": success})

async def _restart_handler(request):
    """POST /restart/{name} - Restart specific service"""
    service_name = request.match_info.get('n')
    success = await supervisor.restart(service_name)
    return await _json_response({"success": success})

async def _call_tool_handler(request):
    """POST /call_tool - Compatible with original API"""
    try:
        data = await request.json()
        
        # Backward compatibility: handle original format {"name": "tool_name"}
        if 'name' in data and 'service' not in data and 'server_name' not in data:
            # Original format - name is the full tool identifier
            tool_name = data['name']
            service_name = None  # Will be resolved by supervisor
        else:
            # New format with separate service and tool
            service_name = data.get('service') or data.get('server_name')
            tool_name = data.get('tool') or data.get('tool_name')
        
        arguments = data.get('arguments', {})
        
        if not tool_name:
            return await _json_response({
                "error": "Missing tool name",
                "success": False
            })
        
        # Handle backward compatibility for original format
        if service_name:
            # New format: explicit service and tool
            result = await supervisor.call_tool(service_name, tool_name, arguments)
        else:
            # Original format: tool_name contains full identifier, let supervisor resolve
            result = await supervisor.call_tool_legacy(tool_name, arguments)
        
        return await _json_response(result)
        
    except Exception as e:
        return await _json_response({
            "error": str(e),
            "success": False
        })

async def _add_server_handler(request):
    """POST /add_server - Add new server configuration"""
    try:
        body = await request.json()
        
        # Handle case where frontend sends JSON as a string in jsonConfig field
        if "jsonConfig" in body and isinstance(body["jsonConfig"], str):
            try:
                body = json.loads(body["jsonConfig"])
            except json.JSONDecodeError:
                return await _json_response({
                    "success": False, 
                    "error": "Fractalic MCP manager: Invalid JSON in jsonConfig field"
                })
        
        # Handle different JSON formats from frontend
        if "mcpServers" in body and isinstance(body["mcpServers"], dict):
            # Format: {"mcpServers": {"server-name": {...}}}
            servers = body["mcpServers"]
            if len(servers) != 1:
                return await _json_response({
                    "success": False, 
                    "error": "Fractalic MCP manager: When using mcpServers format, exactly one server must be provided"
                })
            
            server_name, server_config = next(iter(servers.items()))
            name = server_name
            config = server_config
        else:
            # Standard format with name and config at top level
            name = body.get("name")
            config = body.get("config", {})
            
            if not name:
                return await _json_response({
                    "success": False, 
                    "error": "Fractalic MCP manager: Server name is required"
                })
        
        # Read current configuration
        config_path = Path("mcp_servers.json")
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    current_config = json.load(f)
            else:
                current_config = {"mcpServers": {}}
        except Exception as e:
            return await _json_response({
                "success": False, 
                "error": f"Fractalic MCP manager: Failed to read server configuration: {str(e)}"
            })
        
        # Check for duplicate names
        if name in current_config.get("mcpServers", {}):
            return await _json_response({
                "success": False, 
                "error": f"Fractalic MCP manager: Server with name '{name}' already exists"
            })
        
        # Add server to configuration
        current_config.setdefault("mcpServers", {})[name] = config
        
        # Save updated configuration
        try:
            with open(config_path, 'w') as f:
                json.dump(current_config, f, indent=2)
        except Exception as e:
            return await _json_response({
                "success": False, 
                "error": f"Fractalic MCP manager: Failed to save server configuration: {str(e)}"
            })
        
        # Reload services configuration and try to start the new service
        supervisor.load_config()
        
        try:
            success = await supervisor.start(name)
            if success:
                logger.info(f"Fractalic MCP manager: Launched new server '{name}' dynamically")
            else:
                logger.warning(f"Fractalic MCP manager: Failed to launch '{name}' dynamically")
        except Exception as e:
            logger.error(f"Fractalic MCP manager: Failed to launch '{name}' dynamically: {e}")
        
        return await _json_response({
            "success": True,
            "message": "Fractalic MCP manager: Server added successfully",
            "server": {"name": name, "config": config}
        })
        
    except Exception as e:
        return await _json_response({
            "success": False, 
            "error": f"Fractalic MCP manager: Internal server error: {str(e)}"
        })

async def _delete_server_handler(request):
    """POST /delete_server - Delete server configuration"""
    try:
        body = await request.json()
        name = body.get("name")
        
        if not name:
            return await _json_response({
                "success": False, 
                "error": "Fractalic MCP manager: Server name is required"
            })
        
        # Read current configuration
        config_path = Path("mcp_servers.json")
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    current_config = json.load(f)
            else:
                current_config = {"mcpServers": {}}
        except Exception as e:
            return await _json_response({
                "success": False, 
                "error": f"Fractalic MCP manager: Failed to read server configuration: {str(e)}"
            })
        
        # Check if server exists
        if name not in current_config.get("mcpServers", {}):
            return await _json_response({
                "success": False, 
                "error": f"Fractalic MCP manager: Server '{name}' not found"
            })
        
        # Stop the server if it's running
        try:
            if name in supervisor.sessions:
                await supervisor.stop(name)
                logger.info(f"Stopped running server '{name}' before deletion")
        except Exception as e:
            logger.warning(f"Warning: Failed to stop server '{name}' before deletion: {e}")
        
        # Remove server from configuration
        server_config = current_config["mcpServers"].pop(name)
        
        # Save updated configuration  
        try:
            with open(config_path, 'w') as f:
                json.dump(current_config, f, indent=2)
        except Exception as e:
            return await _json_response({
                "success": False, 
                "error": f"Fractalic MCP manager: Failed to save server configuration: {str(e)}"
            })
        
        # Reload services configuration
        supervisor.load_config()
        
        logger.info(f"Deleted MCP server '{name}' with config: {server_config}")
        
        return await _json_response({
            "success": True,
            "message": "Fractalic MCP manager: Server deleted successfully",
            "server": {"name": name, "config": server_config}
        })
        
    except Exception as e:
        return await _json_response({
            "success": False, 
            "error": f"Fractalic MCP manager: Internal server error: {str(e)}"
        })

async def _kill_handler(request):
    """POST /kill - Terminate the manager"""
    return await _json_response({"success": True, "message": "Fractalic MCP manager: Terminating..."})

# Global variable to store OAuth callback data
oauth_callback_data = {}

async def _oauth_callback_handler(request):
    """GET /oauth/callback - Handle OAuth callback from authorization servers"""
    try:
        # Extract authorization code and state from query parameters
        code = request.query.get('code')
        state = request.query.get('state')
        error = request.query.get('error')
        
        # Store the callback data
        oauth_callback_data.update({
            'code': code,
            'state': state,
            'error': error,
            'received': True
        })
        
        logger.info(f"OAuth callback received - Code: {'✓' if code else '✗'}, State: {state}, Error: {error}")
        
        # Return a proper HTML page
        if error:
            html_content = f"""
            <html>
            <head><title>OAuth Authorization Failed</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 50px;">
                <h1 style="color: #d32f2f;">Authorization Failed</h1>
                <p>Error: {error}</p>
                <p>You can close this window.</p>
            </body>
            </html>
            """
        else:
            html_content = """
            <html>
            <head><title>OAuth Authorization Successful</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 50px;">
                <h1 style="color: #4caf50;">Authorization Successful!</h1>
                <p>Your OAuth authorization was completed successfully.</p>
                <p>You can now close this window and return to your application.</p>
            </body>
            </html>
            """
        
        return web.Response(text=html_content, content_type='text/html')
        
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return web.Response(text="OAuth callback error", status=500)

# Global supervisor instance
supervisor = MCPSupervisor()

# Global supervisor instance
supervisor = MCPSupervisor()

def create_app():
    """Create web application with compatible API endpoints"""
    app = web.Application()
    
    # Enable CORS
    cors = cors_setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Compatible API routes
    app.router.add_get('/status', _status_handler)
    app.router.add_get('/tools', _tools_handler)
    app.router.add_get('/list_tools', _tools_handler)  # Alias
    app.router.add_post('/start/{n}', _start_handler)
    app.router.add_post('/stop/{n}', _stop_handler)
    app.router.add_post('/restart/{n}', _restart_handler)
    app.router.add_post('/call_tool', _call_tool_handler)
    app.router.add_post('/add_server', _add_server_handler)
    app.router.add_post('/delete_server', _delete_server_handler)
    app.router.add_post('/kill', _kill_handler)
    
    # OAuth callback route
    app.router.add_get('/oauth/callback', _oauth_callback_handler)
    
    return app

async def serve_http(port: int = 5859):
    """Start HTTP server on specified port"""
    # Services are loaded in supervisor.__init__() - no need to reload
    
    # Start web server first (don't block on service connections)
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"MCP Manager running on port {port}")
    
    # Try to start services in background (don't block the web server)
    try:
        await supervisor.start()  # Start all services
    except Exception as e:
        logger.error(f"Some services failed to start: {e}")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await supervisor.stop()
        await runner.cleanup()

async def cli_status(port: int = 5859):
    """CLI status command"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'http://localhost:{port}/status') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"Services: {data.get('running_services', 0)}/{data.get('total_services', 0)} running")
                    print(f"Total tools: {data.get('total_tools', 0)}")
                    
                    for name, info in data.get('services', {}).items():
                        status = info.get('status', 'unknown')
                        tools_count = info.get('tools_count', 0)
                        print(f"  {name}: {status} ({tools_count} tools)")
                else:
                    print(f"Error: HTTP {resp.status}")
    except Exception as e:
        print(f"Error connecting to manager: {e}")

async def cli_tools(port: int = 5859):
    """CLI tools command"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'http://localhost:{port}/tools') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    total_tools = 0
                    
                    for service_name, tools in data.items():
                        print(f"{service_name}: {len(tools)} tools")
                        for tool in tools:
                            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
                        total_tools += len(tools)
                    
                    print(f"\nTotal: {total_tools} tools across {len(data)} services")
                else:
                    print(f"Error: HTTP {resp.status}")
    except Exception as e:
        print(f"Error connecting to manager: {e}")

async def cli_start_service(service_name: str, port: int = 5859):
    """CLI start service command"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f'http://localhost:{port}/start/{service_name}') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('success'):
                        print(f"✅ Started {service_name}")
                    else:
                        print(f"❌ Failed to start {service_name}")
                else:
                    print(f"Error: HTTP {resp.status}")
    except Exception as e:
        print(f"Error connecting to manager: {e}")

async def cli_stop_service(service_name: str, port: int = 5859):
    """CLI stop service command"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f'http://localhost:{port}/stop/{service_name}') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('success'):
                        print(f"✅ Stopped {service_name}")
                    else:
                        print(f"❌ Failed to stop {service_name}")
                else:
                    print(f"Error: HTTP {resp.status}")
    except Exception as e:
        print(f"Error connecting to manager: {e}")

async def cli_restart_service(service_name: str, port: int = 5859):
    """CLI restart service command"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f'http://localhost:{port}/restart/{service_name}') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('success'):
                        print(f"✅ Restarted {service_name}")
                    else:
                        print(f"❌ Failed to restart {service_name}")
                else:
                    print(f"Error: HTTP {resp.status}")
    except Exception as e:
        print(f"Error connecting to manager: {e}")

def main():
    """Main CLI entry point - compatible with original"""
    parser = argparse.ArgumentParser(description='MCP Manager - SDK Compatible')
    parser.add_argument('command', nargs='?', default='serve',
                        choices=['serve', 'status', 'tools', 'start', 'stop', 'restart'],
                        help='Command to execute')
    parser.add_argument('name', nargs='?', help='Service name (for start/stop/restart)')
    parser.add_argument('--port', '-p', type=int, default=5859, help='Port number')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        asyncio.run(serve_http(args.port))
    elif args.command == 'status':
        asyncio.run(cli_status(args.port))
    elif args.command == 'tools':
        asyncio.run(cli_tools(args.port))
    elif args.command == 'start':
        if not args.name:
            print("Error: Service name required for start command")
            sys.exit(1)
        asyncio.run(cli_start_service(args.name, args.port))
    elif args.command == 'stop':
        if not args.name:
            print("Error: Service name required for stop command")
            sys.exit(1)
        asyncio.run(cli_stop_service(args.name, args.port))
    elif args.command == 'restart':
        if not args.name:
            print("Error: Service name required for restart command")
            sys.exit(1)
        asyncio.run(cli_restart_service(args.name, args.port))

if __name__ == "__main__":
    main()
