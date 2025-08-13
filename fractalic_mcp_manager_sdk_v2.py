#!/usr/bin/env python3
"""
MCP Manager using official Python SDK v1.12.4
Per-Request Session Pattern (No Persistent Connections)
"""

import asyncio
import json
import logging
import threading
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from aiohttp import web
from pydantic import AnyUrl

# Token counting support (aligned with legacy manager)
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    TOKENIZER = None

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

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

# MCP Protocol Version (Required for 2025-06-18)
MCP_PROTOCOL_VERSION = "2025-06-18"

@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    transport: str  # stdio, sse, http
    spec: Dict[str, Any]
    has_oauth: bool = False
    
    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> 'ServiceConfig':
        # Auto-detect transport if not specified
        transport = config.get('transport', 'stdio')
        if 'url' in config and transport == 'stdio':
            if '/sse' in config['url']:
                transport = 'sse'
            else:
                transport = 'http'  # Use streamable HTTP as default for URLs
        
        return cls(
            name=name,
            transport=transport,
            spec=config,
            has_oauth=config.get('has_oauth', False)
        )

class FileTokenStorage(TokenStorage):
    """File-based token storage for OAuth - service-specific"""
    
    def __init__(self, file_path: str, service_name: str = "default"):
        self.file_path = Path(file_path)
        self.service_name = service_name
    
    async def get_tokens(self) -> Optional[OAuthToken]:
        """Load tokens from file for specific service"""
        logger.info(f"Loading tokens for service: {self.service_name} from {self.file_path}")
        
        if not self.file_path.exists():
            logger.warning(f"Token file {self.file_path} not found")
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Token file loaded, available services: {list(data.keys())}")
            
            # Load service-specific token or fall back to default
            token_key = self.service_name if self.service_name in data else "default"
            if token_key in data:
                token_data = data[token_key]
                logger.info(f"Found tokens for {token_key}, access_token starts with: {token_data['access_token'][:20]}...")
                return OAuthToken(
                    access_token=token_data['access_token'],
                    token_type=token_data.get('token_type', 'Bearer'),
                    expires_in=token_data.get('expires_in'),
                    refresh_token=token_data.get('refresh_token'),
                    scope=token_data.get('scope')
                )
            else:
                logger.warning(f"No tokens found for service {self.service_name} or default")
        except Exception as e:
            logger.error(f"Failed to load tokens for {self.service_name}: {e}")
        
        return None
    
    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Save tokens to file for specific service"""
        try:
            data = {}
            if self.file_path.exists():
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
            
            # Save as service-specific token
            data[self.service_name] = {
                'access_token': tokens.access_token,
                'token_type': tokens.token_type,
                'expires_in': tokens.expires_in,
                'refresh_token': tokens.refresh_token,
                'scope': tokens.scope
            }
            
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"OAuth tokens saved successfully for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to save tokens for {self.service_name}: {e}")
    
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
        self.token_storages: Dict[str, FileTokenStorage] = {}  # Service-specific token storage
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
        """
        Initialize OAuth providers for services that support OAuth.
        The MCP SDK will automatically handle RFC 9728 OAuth discovery when needed.
        """
        for name, service in self.config.items():
            # Only create OAuth providers for services explicitly configured with OAuth
            if service.has_oauth:
                self._create_oauth_provider(name, service)
                logger.info(f"OAuth provider created for explicitly configured service: {name}")
        
        # For other services, the SDK will automatically discover OAuth requirements
        # when they return 401 + WWW-Authenticate headers (per RFC 9728)
    
    def _has_embedded_auth(self, url: str) -> bool:
        """
        Simple heuristic to detect embedded auth tokens in URLs.
        This is only used for informational purposes - the SDK handles OAuth discovery.
        """
        from urllib.parse import urlparse, parse_qs
        import re
        
        parsed = urlparse(url)
        
        # Look for long parameter values that might be tokens
        query_params = parse_qs(parsed.query)
        for values in query_params.values():
            for value in values:
                if len(value) > 15 and re.match(r'^[A-Za-z0-9+/=_-]+$', value):
                    return True
        
        # Look for long path segments that might be tokens
        path_segments = parsed.path.split('/')
        for segment in path_segments:
            if len(segment) >= 20 and re.match(r'^[A-Za-z0-9+/=_-]+$', segment):
                return True
        
        return False

    def _create_oauth_provider(self, name: str, service: ServiceConfig):
        """Create OAuth provider for a service"""
        try:
            server_url = service.spec.get('oauth_server_url', service.spec.get('url', ''))
            if server_url:
                # Create service-specific token storage
                token_storage = FileTokenStorage("oauth_tokens.json", name)
                self.token_storages[name] = token_storage
                
                provider = OAuthClientProvider(
                    server_url=server_url,
                    client_metadata=OAuthClientMetadata(
                        client_name=f"Fractalic MCP Client - {name}",
                        redirect_uris=[AnyUrl("http://localhost:5859/oauth/callback")],
                        grant_types=["authorization_code", "refresh_token"],
                        response_types=["code"],
                        scope=service.spec.get('oauth_scope', 'read write'),
                    ),
                    storage=token_storage,
                    redirect_handler=self._handle_oauth_redirect,
                    callback_handler=self._create_callback_handler_for_service(name),
                )
                self.oauth_providers[name] = provider
                logger.info(f"OAuth provider initialized for {name}")
        except Exception as e:
            logger.error(f"Failed to initialize OAuth for {name}: {e}")
    
    def _create_dynamic_oauth_provider(self, name: str, server_url: str) -> OAuthClientProvider:
        """Create OAuth provider dynamically for automatic OAuth discovery"""
        try:
            # Create service-specific token storage
            token_storage = FileTokenStorage("oauth_tokens.json", name)
            self.token_storages[name] = token_storage
            
            provider = OAuthClientProvider(
                server_url=server_url,
                client_metadata=OAuthClientMetadata(
                    client_name=f"Fractalic MCP Client - {name}",
                    redirect_uris=[AnyUrl("http://localhost:5859/oauth/callback")],
                    grant_types=["authorization_code", "refresh_token"],
                    response_types=["code"],
                    scope="read write",
                ),
                storage=token_storage,
                redirect_handler=self._handle_oauth_redirect,
                callback_handler=self._create_callback_handler_for_service(name),
            )
            # Store it for future use
            self.oauth_providers[name] = provider
            logger.info(f"Dynamic OAuth provider created for {name}")
            return provider
        except Exception as e:
            logger.error(f"Failed to create dynamic OAuth provider for {name}: {e}")
            return None
    
    async def _handle_oauth_redirect(self, auth_url: str) -> None:
        """Handle OAuth redirect by opening browser (async)"""
        logger.info(f"Opening OAuth authorization URL: {auth_url}")
        
        def open_browser():
            webbrowser.open(auth_url)
        
        # Run browser opening in thread to avoid blocking
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Small delay to ensure browser has time to start
        await asyncio.sleep(0.1)
    
    def _create_callback_handler_for_service(self, service_name: str):
        """Create a service-specific callback handler to avoid state mixing"""
        async def callback_handler() -> tuple[str, str | None]:
            """Service-specific OAuth callback handler"""
            # Create a unique callback key for this service's OAuth session
            callback_key = f"{service_name}_{id(asyncio.current_task())}"
            
            # Register this callback session
            oauth_callback_data[callback_key] = {"received": False}
            
            # Wait for the callback with timeout
            timeout = 300  # 5 minutes
            logger.info(f"Waiting for OAuth callback for service: {service_name}")
            
            try:
                for _ in range(timeout * 10):  # Check every 100ms
                    if oauth_callback_data[callback_key].get("received"):
                        break
                    await asyncio.sleep(0.1)
                else:
                    logger.error(f"OAuth callback timeout after 5 minutes for {service_name}")
                    return ("", None)
                
                callback_data = oauth_callback_data.get(callback_key, {})
                
                if callback_data.get("error"):
                    logger.error(f"OAuth error for {service_name}: {callback_data['error']}")
                    return ("", None)
                
                if not callback_data.get("code"):
                    logger.error(f"No authorization code received for {service_name}")
                    return ("", None)
                
                logger.info(f"OAuth authorization code received successfully for {service_name}")
                
                return (callback_data["code"], callback_data.get("state"))
                
            finally:
                # Clean up the callback data
                oauth_callback_data.pop(callback_key, None)
        
        return callback_handler
    
    async def _handle_oauth_redirect(self, auth_url: str) -> None:
        """Handle OAuth redirect by opening browser (async)"""
        logger.info(f"Opening OAuth authorization URL: {auth_url}")
        
        def open_browser():
            webbrowser.open(auth_url)
        
        # Run browser opening in thread to avoid blocking
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Small delay to ensure browser has time to start
        await asyncio.sleep(0.1)
    
    async def get_tools_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """
        Get tools for a service using per-request session with proper context management.
        
        For dynamic OAuth services (like Replicate):
        - Tools are only exposed after successful OAuth authentication
        - The SDK handles RFC 9728 OAuth Protected Resource Metadata discovery automatically
        - Extended timeouts are configured to handle OAuth flow + tool discovery
        - "Dynamic" authentication means tools can change based on user permissions/context
        """
        if service_name not in self.config:
            return []
        
        service = self.config[service_name]
        connection = None
        
        try:
            # Apply httpx low-and-slow attack protection using asyncio.wait_for
            # See: https://github.com/encode/httpx/issues/1450
            # This prevents indefinite hangs when SSE servers send data too slowly
            return await asyncio.wait_for(
                self._get_tools_for_service_impl(service_name, service), 
                timeout=60.0  # 60 second timeout - balances OAuth flows with hang protection
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout (60s) getting tools for {service_name} - SSE server experiencing slow response (httpx low-and-slow issue). Service may be temporarily unavailable.")
            return []
        except Exception as e:
            logger.error(f"Failed to get tools for {service_name}: {e}")
            # Log more details for debugging OAuth/dynamic tools issues
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Full traceback for {service_name}: {full_traceback}")
            return []
    
    async def _get_tools_for_service_impl(self, service_name: str, service: ServiceConfig) -> List[Dict[str, Any]]:
        """Internal implementation of get_tools_for_service with proper error handling"""
        try:
            if service.transport == "stdio":
                # STDIO transport
                server_params = StdioServerParameters(
                    command=service.spec['command'],
                    args=service.spec.get('args', []),
                    env=service.spec.get('env', {})
                )
                
                # Use proper context manager pattern as per SDK examples
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        # Initialize the connection
                        await session.initialize()
                        
                        # List tools
                        tools_result = await session.list_tools()
                        tools = []
                        
                        for tool in tools_result.tools:
                            tool_dict = {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            
                            # Include title if available (2025-06-18 feature)
                            if hasattr(tool, 'title') and tool.title:
                                tool_dict["title"] = tool.title
                            
                            # Include annotations if available (2025-06-18 feature)  
                            if hasattr(tool, 'annotations') and tool.annotations:
                                tool_dict["annotations"] = tool.annotations
                            
                            tools.append(tool_dict)
                        
                        return tools
                        
            elif service.transport == "sse":
                # SSE transport with automatic OAuth discovery
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                # Create OAuth provider dynamically if needed and not already exists
                oauth_provider = self.oauth_providers.get(service.name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    # Create OAuth provider for potential OAuth discovery
                    oauth_provider = self._create_dynamic_oauth_provider(service.name, url)
                
                # Configure timeouts for OAuth services (especially dynamic ones like Replicate)
                # SSE client expects float timeouts (seconds) - per SDK issue #936
                connection_timeout = 30.0  # Connection establishment timeout
                # Let SDK use default sse_read_timeout
                
                logger.info(f"Connecting to {service_name} SSE service with {connection_timeout}s connection timeout, default read timeout")
                
                # Store tools_result outside context manager to handle cleanup timeouts
                tools_result = None
                
                try:
                    # Note: Issue #936 documents timeout type inconsistencies in MCP SDK
                    # sse_client expects float, streamablehttp_client expects timedelta
                    async with sse_client(
                        url, 
                        auth=oauth_provider, 
                        headers=headers,
                        timeout=connection_timeout
                    ) as (read_stream, write_stream):
                        logger.info(f"SSE connection established for {service_name}, starting MCP session...")
                        async with ClientSession(read_stream, write_stream) as session:
                            # Initialize the connection
                            logger.info(f"Initializing MCP session for {service_name}...")
                            await session.initialize()
                            logger.info(f"MCP session initialized for {service_name}, listing tools...")
                            
                            # List tools
                            tools_result = await session.list_tools()
                            logger.info(f"Received {len(tools_result.tools)} tools from {service_name}")
                except Exception as cleanup_error:
                    # If we got tools but cleanup failed, still process the tools
                    if tools_result is not None:
                        logger.warning(f"Got tools successfully but cleanup failed for {service_name}: {cleanup_error}")
                    else:
                        # Re-raise if we didn't get tools
                        raise
                
                # Process tools outside context manager (works even if cleanup failed)
                if tools_result is not None:
                    tools = []
                    for tool in tools_result.tools:
                        tool_dict = {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        
                        # Include title if available (2025-06-18 feature)
                        if hasattr(tool, 'title') and tool.title:
                            tool_dict["title"] = tool.title
                        
                        # Include annotations if available (2025-06-18 feature)  
                        if hasattr(tool, 'annotations') and tool.annotations:
                            tool_dict["annotations"] = tool.annotations
                        
                        tools.append(tool_dict)
                    
                    return tools
                else:
                    return []
                        
            elif service.transport == "http":
                # HTTP transport (streamable) with automatic OAuth discovery
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                # Create OAuth provider dynamically if needed and not already exists
                oauth_provider = self.oauth_providers.get(service.name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    # Create OAuth provider for potential OAuth discovery
                    oauth_provider = self._create_dynamic_oauth_provider(service.name, url)
                
                # Configure timeouts for OAuth services
                # Streamable HTTP client expects timedelta timeouts (per SDK issue #936)
                from datetime import timedelta
                connection_timeout = timedelta(seconds=30)    # Connection establishment timeout
                sse_read_timeout = timedelta(seconds=45)      # Reduced read timeout for faster failure detection
                
                logger.info(f"Connecting to {service_name} HTTP service with {connection_timeout.total_seconds()}s connection timeout, {sse_read_timeout.total_seconds()}s read timeout")
                
                async with streamablehttp_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        # Initialize the connection
                        await session.initialize()
                        
                        # List tools
                        tools_result = await session.list_tools()
                        tools = []
                        
                        for tool in tools_result.tools:
                            tool_dict = {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            
                            # Include title if available (2025-06-18 feature)
                            if hasattr(tool, 'title') and tool.title:
                                tool_dict["title"] = tool.title
                            
                            # Include annotations if available (2025-06-18 feature)  
                            if hasattr(tool, 'annotations') and tool.annotations:
                                tool_dict["annotations"] = tool.annotations
                            
                            tools.append(tool_dict)
                        
                        return tools
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
        
        except Exception as e:
            logger.error(f"Error in _get_tools_for_service_impl for {service_name}: {e}")
            # Check if this is a timeout error for OAuth services
            if "timeout" in str(e).lower() or "ReadTimeout" in str(e):
                if service.has_oauth or service_name in self.oauth_providers:
                    logger.warning(f"Timeout detected for OAuth service {service_name}. This may be normal for dynamic services that perform OAuth discovery + tool listing.")
                else:
                    logger.warning(f"Timeout detected for {service_name}. Service may be slow or unresponsive.")
            raise  # Re-raise to be caught by the outer timeout handler
    
    async def get_tools_info_for_service(self, service_name: str) -> Dict[str, Any]:
        """
        Get tool count and token count for a specific service.
        Returns count information aligned with legacy MCP manager.
        """
        if service_name not in self.config:
            return {"tool_count": 0, "token_count": 0, "tools_error": "Service not found"}
        
        try:
            tools = await self.get_tools_for_service(service_name)
            
            if not tools:
                return {"tool_count": 0, "token_count": 0}
            
            tool_count = len(tools)
            
            # Calculate token count using available tokenizer
            if LITELLM_AVAILABLE:
                try:
                    # Convert MCP tools to OpenAI format for LiteLLM (aligned with legacy)
                    openai_tools = []
                    for tool in tools:
                        openai_tool = {
                            "type": "function", 
                            "function": {
                                "name": tool.get("name", "unknown"),
                                "description": tool.get("description", ""),
                                "parameters": tool.get("inputSchema", {})
                            }
                        }
                        openai_tools.append(openai_tool)
                    
                    # Use LiteLLM to count tokens (same as legacy manager)
                    token_count = litellm.token_counter(model="gpt-4", messages=[], tools=openai_tools)
                    
                except Exception as e:
                    # Fallback to tiktoken if LiteLLM fails
                    if TIKTOKEN_AVAILABLE:
                        schema_json = json.dumps(tools)
                        token_count = len(TOKENIZER.encode(schema_json))
                    else:
                        # Rough estimate if no tokenizer available
                        schema_json = json.dumps(tools)
                        token_count = len(schema_json) // 4  # Rough approximation
                        
            elif TIKTOKEN_AVAILABLE:
                # Direct tiktoken approach (same as legacy manager)
                schema_json = json.dumps(tools)
                token_count = len(TOKENIZER.encode(schema_json))
            else:
                # Rough estimate if no tokenizer available
                schema_json = json.dumps(tools)
                token_count = len(schema_json) // 4  # Rough approximation
            
            return {"tool_count": tool_count, "token_count": token_count}
            
        except Exception as e:
            logger.error(f"Failed to get tools info for {service_name}: {e}")
            return {"tool_count": 0, "token_count": 0, "tools_error": str(e)}
    
    async def call_tool_for_service(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a service using per-request session with proper context management"""
        if service_name not in self.config:
            raise ValueError(f"Service {service_name} not found")
        
        service = self.config[service_name]
        
        try:
            if service.transport == "stdio":
                # STDIO transport
                server_params = StdioServerParameters(
                    command=service.spec['command'],
                    args=service.spec.get('args', []),
                    env=service.spec.get('env', {})
                )
                
                # Use proper context manager pattern as per SDK examples
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        # Initialize the connection
                        await session.initialize()
                        
                        # Call tool
                        result = await session.call_tool(tool_name, arguments)
                        
                        # Format response with structured output support (2025-06-18)
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
                            elif hasattr(content, 'resource'):
                                # Handle embedded resources
                                response["content"].append({
                                    "type": "resource",
                                    "resource": {
                                        "uri": content.resource.uri,
                                        "text": getattr(content.resource, 'text', None),
                                        "blob": getattr(content.resource, 'blob', None)
                                    }
                                })
                        
                        # Handle structured content (2025-06-18 feature)
                        if hasattr(result, 'structuredContent') and result.structuredContent:
                            response["structuredContent"] = result.structuredContent
                        
                        # Handle tool annotations if present
                        if hasattr(result, '_meta') and result._meta:
                            response["_meta"] = result._meta
                        
                        return response
                        
            elif service.transport == "sse":
                # SSE transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = self.oauth_providers.get(service.name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    oauth_provider = self._create_dynamic_oauth_provider(service.name, url)
                
                # Configure timeouts for OAuth services (SSE client expects float)
                connection_timeout = 30.0   # Connection establishment timeout
                sse_read_timeout = 300.0     # Extended SSE read timeout for OAuth + tool calls (5 min)
                
                async with sse_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        
                        result = await session.call_tool(tool_name, arguments)
                        
                        # Format response (same as stdio)
                        response = {
                            "content": [],
                            "isError": getattr(result, 'isError', False)
                        }
                        
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
                            elif hasattr(content, 'resource'):
                                response["content"].append({
                                    "type": "resource",
                                    "resource": {
                                        "uri": content.resource.uri,
                                        "text": getattr(content.resource, 'text', None),
                                        "blob": getattr(content.resource, 'blob', None)
                                    }
                                })
                        
                        if hasattr(result, 'structuredContent') and result.structuredContent:
                            response["structuredContent"] = result.structuredContent
                        
                        if hasattr(result, '_meta') and result._meta:
                            response["_meta"] = result._meta
                        
                        return response
                        
            elif service.transport == "http":
                # HTTP transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = self.oauth_providers.get(service.name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    oauth_provider = self._create_dynamic_oauth_provider(service.name, url)
                
                # Configure timeouts for OAuth services (Streamable HTTP client expects timedelta)
                from datetime import timedelta
                connection_timeout = timedelta(seconds=30)    # Connection establishment timeout
                sse_read_timeout = timedelta(seconds=600)     # Extended read timeout for OAuth + tool calls
                
                async with streamablehttp_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        
                        result = await session.call_tool(tool_name, arguments)
                        
                        # Format response (same as stdio)
                        response = {
                            "content": [],
                            "isError": getattr(result, 'isError', False)
                        }
                        
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
                            elif hasattr(content, 'resource'):
                                response["content"].append({
                                    "type": "resource",
                                    "resource": {
                                        "uri": content.resource.uri,
                                        "text": getattr(content.resource, 'text', None),
                                        "blob": getattr(content.resource, 'blob', None)
                                    }
                                })
                        
                        if hasattr(result, 'structuredContent') and result.structuredContent:
                            response["structuredContent"] = result.structuredContent
                        
                        if hasattr(result, '_meta') and result._meta:
                            response["_meta"] = result._meta
                        
                        return response
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
                
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} for {service_name}: {e}")
            raise
    
    async def test_service_connection(self, service_name: str) -> bool:
        """Test connection to a service with proper context management"""
        if service_name not in self.config:
            return False
        
        service = self.config[service_name]
        
        
        try:
            return await self._test_service_connection_impl(service_name, service)
        except Exception as e:
            logger.error(f"Failed to test connection to {service_name}: {e}")
            self.service_states[service_name] = "error"
            return False
    
    async def _test_service_connection_impl(self, service_name: str, service: ServiceConfig) -> bool:
        """Internal implementation of test_service_connection with proper error handling"""
        try:
            if service.transport == "stdio":
                # STDIO transport
                server_params = StdioServerParameters(
                    command=service.spec['command'],
                    args=service.spec.get('args', []),
                    env=service.spec.get('env', {})
                )
                
                # Use proper context manager pattern
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        # Initialize and test connection
                        await session.initialize()
                        await session.list_tools()
                        
                        self.service_states[service_name] = "running"
                        logger.info(f"Service {service_name} connection tested successfully")
                        return True
                        
            elif service.transport == "sse":
                # SSE transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = self.oauth_providers.get(service_name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    oauth_provider = self._create_dynamic_oauth_provider(service_name, url)
                
                # Configure timeouts for OAuth services (SSE client expects float)
                connection_timeout = 15.0   # Shorter connection timeout for tests
                sse_read_timeout = 20.0     # Shorter SSE read timeout for tests
                
                logger.info(f"Testing {service_name} SSE connection with {connection_timeout}s connection timeout, {sse_read_timeout}s read timeout")
                
                async with sse_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        await session.list_tools()
                        
                        self.service_states[service_name] = "running"
                        logger.info(f"Service {service_name} connection tested successfully")
                        return True
                        
            elif service.transport == "http":
                # HTTP transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = self.oauth_providers.get(service_name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    oauth_provider = self._create_dynamic_oauth_provider(service_name, url)
                
                # Configure timeouts for OAuth services (Streamable HTTP client expects timedelta)
                from datetime import timedelta
                connection_timeout = timedelta(seconds=15)    # Shorter connection timeout for tests
                sse_read_timeout = timedelta(seconds=20)      # Shorter read timeout for tests
                
                logger.info(f"Testing {service_name} HTTP connection with {connection_timeout.total_seconds()}s connection timeout, {sse_read_timeout.total_seconds()}s read timeout")
                
                async with streamablehttp_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        await session.list_tools()
                        
                        self.service_states[service_name] = "running"
                        logger.info(f"Service {service_name} connection tested successfully")
                        return True
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
        
        except Exception as e:
            logger.error(f"Error in _test_service_connection_impl for {service_name}: {e}")
            if "timeout" in str(e).lower() or "ReadTimeout" in str(e):
                logger.warning(f"Timeout detected during connection test for {service_name}")
            raise  # Re-raise to be caught by the outer timeout handler
    
    async def status(self, include_tools_info: bool = False) -> Dict[str, Any]:
        """
        Get status of all services.
        If include_tools_info=True, fetches actual tools count and token count for each service.
        Otherwise, returns configuration status only (faster for health checks).
        """
        services = {}
        total_tools = 0
        total_tokens = 0
        
        for name, service in self.config.items():
            # Base configuration info
            service_info = {
                "status": self.service_states.get(name, "not_tested"),
                "connected": False,  # Per-request pattern - no persistent connections
                "has_oauth": service.has_oauth or name in self.oauth_providers,
                "transport": service.transport,
                "url": service.spec.get('url', None),
                "command": service.spec.get('command', None)
            }
            
            # Include tools information if requested (aligned with legacy manager)
            if include_tools_info:
                try:
                    tools_info = await self.get_tools_info_for_service(name)
                    service_info.update(tools_info)  # Adds tool_count, token_count, and any tools_error
                    
                    # Accumulate totals
                    total_tools += tools_info.get("tool_count", 0)
                    total_tokens += tools_info.get("token_count", 0)
                    
                except Exception as e:
                    logger.error(f"Failed to get tools info for {name}: {e}")
                    service_info.update({
                        "tool_count": 0,
                        "token_count": 0,
                        "tools_error": str(e)
                    })
            else:
                # Placeholder values for fast status checks
                service_info.update({
                    "tool_count": 0,     # Will be populated when tools info is requested
                    "token_count": 0     # Will be populated when tools info is requested
                })
            
            services[name] = service_info
        
        return {
            "services": services,
            "total_services": len(self.config),
            "running_services": 0,  # Per-request pattern - no persistent connections
            "oauth_enabled": len(self.oauth_providers) > 0,
            "total_tools": total_tools if include_tools_info else 0,
            "total_tokens": total_tokens if include_tools_info else 0,
            "mcp_version": MCP_PROTOCOL_VERSION
        }

# Global supervisor instance
supervisor = MCPSupervisorV2()

# OAuth callback handler for web server
async def oauth_callback_handler(request):
    """Handle OAuth callback from the authorization server"""
    global oauth_callback_data
    
    query = request.query
    
    if 'code' in query:
        # Store callback data for all active OAuth sessions
        # Since we can't know which service this callback is for, we store it for all pending sessions
        code = query['code']
        state = query.get('state')
        
        # Find all pending OAuth sessions and notify them
        for key in list(oauth_callback_data.keys()):
            if not oauth_callback_data[key].get("received"):
                oauth_callback_data[key].update({
                    'received': True,
                    'code': code,
                    'state': state,
                    'error': None
                })
        
        logger.info("OAuth authorization code received and broadcast to all pending sessions")
        
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
        
        # Broadcast error to all pending OAuth sessions
        for key in list(oauth_callback_data.keys()):
            if not oauth_callback_data[key].get("received"):
                oauth_callback_data[key].update({
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
                    body {{ font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }}
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
        # Broadcast invalid request to all pending OAuth sessions
        for key in list(oauth_callback_data.keys()):
            if not oauth_callback_data[key].get("received"):
                oauth_callback_data[key].update({
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
    """GET /status - Get services status with optional tools info"""
    try:
        # Check for include_tools_info query parameter
        include_tools = request.query.get('include_tools_info', '').lower() in ('true', '1', 'yes')
        status = await supervisor.status(include_tools_info=include_tools)
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
    """GET /tools/{name} - Get tools for a service with count and token information"""
    service_name = request.match_info['name']
    try:
        # Get both tools and tools info (includes token counting)
        tools = await supervisor.get_tools_for_service(service_name)
        tools_info = await supervisor.get_tools_info_for_service(service_name)
        
        response = {
            "tools": tools, 
            "count": len(tools),
            "tool_count": tools_info.get("tool_count", len(tools)),
            "token_count": tools_info.get("token_count", 0)
        }
        
        # Include any error information from tools_info
        if "tools_error" in tools_info:
            response["tools_error"] = tools_info["tools_error"]
            
        return web.json_response(response)
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
