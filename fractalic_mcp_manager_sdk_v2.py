#!/usr/bin/env python3
"""
MCP Manager using official Python SDK v1.12.4
Per-Request Session Pattern (No Persistent Connections)
"""

import asyncio
import json
import os
import logging
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path

# Always resolve repository root from this file location
ROOT_DIR = Path(__file__).resolve().parent
from pathlib import Path as _Path  # alias for later use in token storage
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
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> 'ServiceConfig':
        # Auto-detect transport if not specified
        transport = config.get('transport', 'stdio')
        if 'url' in config and transport == 'stdio':
            url_val = config['url']
            # Determine path segments to identify embedded 'sse'
            from urllib.parse import urlparse
            path = urlparse(url_val).path.rstrip('/').lower()
            # If path ends with /sse OR contains /mcp/ and ends with /sse style segment treat as SSE
            if path.endswith('/sse') or '/sse/' in path or path.endswith('/mcp/sse'):
                transport = 'sse'
            else:
                # Default to streamable HTTP for generic URLs
                transport = 'http'
        
        return cls(
            name=name,
            transport=transport,
            spec=config,
            has_oauth=config.get('has_oauth', False),
            enabled=config.get('enabled', True)
        )

class FileTokenStorage(TokenStorage):
    """File-based token storage for OAuth - service-specific"""
    
    def __init__(self, file_path: str, service_name: str = "default"):
        # Persist tokens at repository root regardless of where server started
        p = _Path(file_path)
        if not p.is_absolute():
            p = ROOT_DIR / p
        self.file_path = p
        self.service_name = service_name
    
    async def get_tokens(self) -> Optional[OAuthToken]:
        """Load tokens from file for specific service (no manual refresh; SDK provider handles refresh)."""
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
            import time as _time
            data[self.service_name] = {
                'access_token': tokens.access_token,
                'token_type': tokens.token_type,
                'expires_in': tokens.expires_in,
                'refresh_token': tokens.refresh_token,
                'scope': tokens.scope,
                'obtained_at': _time.time()
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

    async def delete_tokens(self) -> None:
        """Delete stored tokens for this service (force re-auth/refresh)."""
        try:
            if not self.file_path.exists():
                return
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            if self.service_name in data:
                del data[self.service_name]
                with open(self.file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Deleted tokens for {self.service_name} (forced invalidation)")
        except Exception as e:
            logger.error(f"Failed to delete tokens for {self.service_name}: {e}")


# Global callback data for OAuth
oauth_callback_data = {}

class MCPSupervisorV2:
    """MCP Service Manager using official Python SDK - Per-Request Sessions"""
    
    def __init__(self):
        # Load configuration & initialize state containers
        self.config: Dict[str, ServiceConfig] = self.load_config()
        self.oauth_providers: Dict[str, OAuthClientProvider] = {}
        self.token_storages: Dict[str, FileTokenStorage] = {}
        self.service_states: Dict[str, str] = {}
        self.tools_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.oauth_attempt_timestamps: Dict[str, float] = {}
        self.last_unauthorized: Dict[str, float] = {}

        # Initialize OAuth providers for services likely needing OAuth
        self._init_oauth_providers()

        # Set initial enabled/disabled states from config
        for service_name, svc in self.config.items():
            self.service_states[service_name] = "enabled" if getattr(svc, 'enabled', True) else "disabled"
        # tools_cache populated asynchronously after server starts
    
    def load_config(self) -> Dict[str, ServiceConfig]:
        """Load MCP servers configuration"""
        config_file = ROOT_DIR / "mcp_servers.json"
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
            # Create OAuth providers for services that likely need OAuth
            if self._is_oauth_service(service.spec.get('url', '')):
                self._create_oauth_provider(name, service)
                logger.info(f"OAuth provider created for detected OAuth service: {name}")
        
        # For other services, the SDK will automatically discover OAuth requirements
        # when they return 401 + WWW-Authenticate headers (per RFC 9728)
    
    async def _populate_initial_cache(self):
        """
        Populate tools cache on startup for all enabled services.
        This provides instant tool lookups instead of per-request connections.
        """
        logger.info("Starting initial tools cache population...")
        cache_start_time = time.time()
        
        # Load tools for all enabled services in parallel
        tasks = []
        for service_name in self.config:
            if self.service_states.get(service_name, "enabled") == "enabled":
                task = asyncio.create_task(self._cache_tools_for_service(service_name))
                tasks.append(task)
        
        # Wait for all services to complete (with individual timeouts handled in _cache_tools_for_service)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        cache_end_time = time.time()
        cached_services = len([name for name in self.tools_cache if self.tools_cache[name]])
        total_tools = sum(len(tools) for tools in self.tools_cache.values())
        
        logger.info(f"Initial tools cache populated in {cache_end_time - cache_start_time:.2f}s: "
                   f"{cached_services} services, {total_tools} total tools")
    
    async def _cache_tools_for_service(self, service_name: str):
        """
        Cache tools for a single service with error handling.
        This is used during startup and when services are enabled.
        """
        try:
            tools = await self._get_tools_for_service_direct(service_name)
            self.tools_cache[service_name] = tools
            logger.debug(f"Cached {len(tools)} tools for {service_name}")
        except Exception as e:
            logger.warning(f"Failed to cache tools for {service_name}: {e}")
            self.tools_cache[service_name] = []  # Empty cache on failure
    
    async def _get_tools_for_service_direct(self, service_name: str) -> List[Dict[str, Any]]:
        """
        Direct tool retrieval bypassing cache - used for cache population and refresh.
        This is the original get_tools_for_service logic before caching optimization.
        """
        if service_name not in self.config:
            return []
        
        # Check if service is enabled before attempting connection
        if self.service_states.get(service_name, "enabled") == "disabled":
            logger.debug(f"Service {service_name} is disabled, skipping direct tool retrieval")
            return []
        
        service = self.config[service_name]
        
        try:
            # Apply httpx low-and-slow attack protection using asyncio.wait_for
            # For OAuth services, use longer timeout to allow browser authentication
            timeout = 3.0  # Default fast timeout
            
            # Dynamic OAuth detection - check if service requires OAuth
            needs_oauth = (service_name in self.oauth_providers or 
                          self._is_oauth_service(service.spec.get('url', '')))
            
            if needs_oauth:
                token_storage = FileTokenStorage("oauth_tokens.json", service_name)
                tokens = await token_storage.get_tokens()
                if not tokens:
                    now = time.time()
                    last = self.oauth_attempt_timestamps.get(service_name)
                    if last and (now - last) < 90:  # 90s cool-down
                        logger.info(f"Skipping new OAuth attempt for {service_name}; last attempt {(now-last):.1f}s ago")
                        return []
                    self.oauth_attempt_timestamps[service_name] = now
                    timeout = 120.0  # Allow user to complete browser auth
                    logger.debug(f"Using extended timeout ({timeout}s) for initial OAuth flow for {service_name}")
                
            return await asyncio.wait_for(
                self._get_tools_for_service_impl(service_name, service), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            needs_oauth = (service_name in self.oauth_providers or 
                          self._is_oauth_service(service.spec.get('url', '')))
            if needs_oauth:
                logger.warning(f"OAuth timeout for {service_name} - user may need to complete browser authentication")
            else:
                logger.warning(f"Timeout (3s) getting tools for {service_name} - fast fail, non-blocking.")
            return []
        except Exception as e:
            logger.error(f"Failed to get tools directly for {service_name}: {e}")
            return []
    
    def _update_service_cache(self, service_name: str, enabled: bool):
        """
        Update cache when service is enabled/disabled.
        Adds tools to cache when enabled, removes when disabled.
        """
        if enabled:
            # Service enabled - cache tools asynchronously
            asyncio.create_task(self._cache_tools_for_service(service_name))
            logger.info(f"Service {service_name} enabled - updating cache")
        else:
            # Service disabled - remove from cache
            if service_name in self.tools_cache:
                tool_count = len(self.tools_cache[service_name])
                del self.tools_cache[service_name]
                logger.info(f"Service {service_name} disabled - removed {tool_count} tools from cache")

    # -------- Helper consolidation (auth, sessions, formatting) --------

    def _protocol_headers(self) -> Dict[str, str]:
        return {'MCP-Protocol-Version': MCP_PROTOCOL_VERSION} if MCP_PROTOCOL_VERSION else {}

    def _resolve_oauth_provider(self, service_name: str, url: str) -> Optional[OAuthClientProvider]:
        """Return existing OAuth provider (no heuristic creation here).
        Dynamic creation now performed only in unified 401 retry wrapper to stay protocol-agnostic."""
        return self.oauth_providers.get(service_name)

    def _base_server_url(self, full_url: str) -> str:
        """Derive base server URL (scheme://host) for OAuth provider creation."""
        from urllib.parse import urlparse
        try:
            p = urlparse(full_url)
            if p.scheme and p.netloc:
                return f"{p.scheme}://{p.netloc}"
        except Exception:
            pass
        return full_url  # fallback

    def ensure_oauth_provider(self, service_name: str) -> bool:
        """Proactively ensure an OAuth provider exists for a service.
        Returns True if provider exists or was created, False otherwise."""
        if service_name in self.oauth_providers:
            return True
        svc = self.config.get(service_name)
        if not svc:
            return False
        url = svc.spec.get('url')
        if not url:
            return False
        base = self._base_server_url(url)
        provider = self._create_dynamic_oauth_provider(service_name, base)
        return provider is not None

    async def reset_oauth_tokens(self, service_name: str) -> bool:
        """Delete stored tokens and clear last unauthorized timestamp."""
        storage = self.token_storages.get(service_name)
        if not storage:
            # If provider not yet created, create it so storage exists
            if not self.ensure_oauth_provider(service_name):
                return False
            storage = self.token_storages.get(service_name)
        if storage and hasattr(storage, 'delete_tokens'):
            await storage.delete_tokens()
            self.last_unauthorized.pop(service_name, None)
            return True
        return False

    async def _run_with_oauth_retry(self, service_name: str, purpose: str, transport: str, url: str, op_coro_factory):
        """Execute an async operation with at most one OAuth-driven retry on 401.
        Logic:
          1. Attempt operation with existing provider (if any).
          2. If 401 and no provider: create provider from base server URL, retry once.
          3. If 401 and provider exists: delete tokens (if any) to force re-auth, retry once.
          4. Record cooldown to avoid tight unauthorized loops (60s).
        """
        provider = self.oauth_providers.get(service_name)
        attempts = 0
        cooldown = 60.0
        while attempts < 2:
            try:
                return await op_coro_factory(provider)
            except Exception as e:
                msg = str(e)
                if '401' in msg:
                    now = time.time()
                    last = self.last_unauthorized.get(service_name, 0)
                    if attempts == 0 and (now - last) < cooldown:
                        # Suppress repeated immediate retries
                        self._log_event(service_name, purpose, transport, time.perf_counter(), status="unauthorized_suppressed", since=f"{now-last:.1f}s")
                        raise
                    self.last_unauthorized[service_name] = now
                    self._log_event(service_name, purpose, transport, time.perf_counter(), status="unauthorized", attempt=attempts+1)
                    # Prepare retry path
                    if provider is None:
                        # Create provider
                        base = self._base_server_url(url)
                        provider = self._create_dynamic_oauth_provider(service_name, base)
                        if provider is None:
                            self._log_event(service_name, purpose, transport, time.perf_counter(), status="oauth_provider_create_failed")
                            raise
                    else:
                        # Existing provider: nuke tokens to force new auth
                        storage = self.token_storages.get(service_name)
                        if storage and hasattr(storage, 'delete_tokens'):
                            try:
                                await storage.delete_tokens()
                            except Exception:
                                logger.debug("Token deletion failed during retry", exc_info=True)
                    attempts += 1
                    continue
                raise
        # If we exit loop without return, last attempt failed and raised. Here just raise generic.
        raise RuntimeError(f"{service_name} {purpose} failed after OAuth retry")

    def _compute_timeouts(self, service_name: str, transport: str, purpose: str, has_oauth: bool):
        """Unified timeout policy. Returns (connection_timeout, read_timeout_or_None).
        SSE expects float; HTTP expects timedelta objects."""
        from datetime import timedelta
        # Base defaults
        if transport == 'sse':
            base_conn = 3.0
            if has_oauth and purpose in ('tools','init'):
                base_conn = 10.0  # allow discovery / first auth
            if service_name.lower() == 'zapier':
                base_conn = max(base_conn, 5.0)
            return base_conn, None
        else:  # http streamable
            base_conn = 5
            read = 5
            if has_oauth and purpose in ('tools','init'):
                base_conn = 15
                read = 30
            if service_name.lower() == 'zapier':
                base_conn = max(base_conn, 8)
                read = max(read, 15)
            return timedelta(seconds=base_conn), timedelta(seconds=read)

    # Formatting helpers
    def _format_tools(self, tools_result) -> List[Dict[str, Any]]:
        tools = []
        for tool in getattr(tools_result, 'tools', []) or []:
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            if hasattr(tool, 'title') and tool.title:
                tool_dict["title"] = tool.title
            if hasattr(tool, 'annotations') and tool.annotations:
                tool_dict["annotations"] = tool.annotations
            tools.append(tool_dict)
        return tools

    def _format_tool_call_result(self, result) -> Dict[str, Any]:
        response = {"content": [], "isError": getattr(result, 'isError', False)}
        for content in getattr(result, 'content', []) or []:
            if hasattr(content, 'text'):
                response["content"].append({"type": "text", "text": content.text})
            elif hasattr(content, 'data'):
                response["content"].append({"type": "image", "data": content.data, "mimeType": getattr(content, 'mimeType', 'image/png')})
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

    def _format_prompt(self, prompt_result) -> Dict[str, Any]:
        response = {"description": getattr(prompt_result, 'description', ''), "messages": []}
        for message in getattr(prompt_result, 'messages', []) or []:
            msg_dict = {"role": message.role.value if hasattr(message.role, 'value') else str(message.role), "content": []}
            for content in message.content:
                if hasattr(content, 'text'):
                    msg_dict["content"].append({"type": "text", "text": content.text})
                elif hasattr(content, 'data'):
                    msg_dict["content"].append({"type": "image", "data": content.data, "mimeType": getattr(content, 'mimeType', 'image/png')})
                elif hasattr(content, 'resource'):
                    msg_dict["content"].append({
                        "type": "resource",
                        "resource": {
                            "uri": content.resource.uri,
                            "text": getattr(content.resource, 'text', None),
                            "blob": getattr(content.resource, 'blob', None)
                        }
                    })
            response["messages"].append(msg_dict)
        return response

    def _format_resource(self, resource_result, resource_uri: str) -> Dict[str, Any]:
        response = {"uri": resource_uri, "contents": []}
        for content in getattr(resource_result, 'contents', []) or []:
            if hasattr(content, 'text'):
                response["contents"].append({"type": "text", "text": content.text, "uri": getattr(content, 'uri', resource_uri)})
            elif hasattr(content, 'blob'):
                response["contents"].append({
                    "type": "blob",
                    "blob": content.blob,
                    "mimeType": getattr(content, 'mimeType', 'application/octet-stream'),
                    "uri": getattr(content, 'uri', resource_uri)
                })
        return response

    def _is_oauth_service(self, url: str) -> bool:
        """Minimal heuristic for proactive OAuth hinting.
        Remains intentionally path/host-pattern based only (no vendor/domain allow‑list) to stay protocol‑agnostic.
        Real canonical detection occurs via RFC 9728 401 + WWW-Authenticate challenge; when that happens we lazily
        create an OAuth provider (see 401 branches in tool/resource/prompt retrieval). This helper only provides an
        early hint for obvious /oauth|/authorize style URLs; absence never blocks dynamic creation after a 401.
        """
        if not url:
            return False
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        indicators = ('/oauth', '/auth', '/authorize', '/token', 'login.', 'accounts.', 'signin')
        return any(ind in path or ind in domain for ind in indicators)

    # Adaptive timeout metrics helpers (generic)
    def _record_timeout(self, service_name: str):
        m = getattr(self, 'service_metrics', None)
        if m is None:
            self.service_metrics = {}
            m = self.service_metrics
        entry = m.setdefault(service_name, {'timeouts': 0, 'last_success': None})
        entry['timeouts'] += 1

    def _record_success(self, service_name: str):
        m = getattr(self, 'service_metrics', None)
        if m is None:
            self.service_metrics = {}
            m = self.service_metrics
        entry = m.setdefault(service_name, {'timeouts': 0, 'last_success': None})
        entry['last_success'] = time.time()
        if entry['timeouts'] > 0:
            entry['timeouts'] -= 1

    def _adaptive_timeout(self, service_name: str, base: float, max_factor: float = 5.0) -> float:
        m = getattr(self, 'service_metrics', None)
        if not m:
            return base
        entry = m.get(service_name)
        if not entry:
            return base
        factor = 1 + min(entry.get('timeouts', 0), max_factor - 1)
        return min(base * factor, base * max_factor)

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

    # -------- Structured logging helpers --------
    def _log_event(self, service: str, action: str, transport: Optional[str], start_ts: float, status: str = "ok", **extra):
        """Centralized single-line structured event logging.
        Always emits: evt, service, action, transport, status, ms plus any extra key=value pairs.
        """
        try:
            duration_ms = (time.perf_counter() - start_ts) * 1000.0
            base = {
                "evt": "mcp",
                "service": service,
                "action": action,
                "transport": transport,
                "status": status,
                "ms": f"{duration_ms:.1f}"
            }
            # Filter out None values
            for k, v in extra.items():
                if v is not None:
                    base[k] = v
            logger.info(" ".join(f"{k}={v}" for k, v in base.items()))
        except Exception:  # Never let logging raise
            logger.debug("Structured logging failed", exc_info=True)
    
    async def get_tools_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """
        Get tools for a service using startup cache for instant response.
        
        This now uses the tools_cache populated during startup for extremely fast response times.
        Cache is updated when services are enabled/disabled via _update_service_cache.
        """
        start_ts = time.perf_counter()
        if service_name not in self.config:
            self._log_event(service_name, "get_tools", None, start_ts, status="skip", reason="not_found")
            return []
        
        # Check if service is enabled before returning cached tools
        if self.service_states.get(service_name, "enabled") == "disabled":
            logger.debug(f"Service {service_name} is disabled, returning empty tools")
            self._log_event(service_name, "get_tools", self.config[service_name].transport, start_ts, status="skip", reason="disabled")
            return []
        
        # Return cached tools for instant response (with lazy fallback fetch if empty)
        cached_tools = self.tools_cache.get(service_name, [])
        if cached_tools:
            logger.debug(f"Returning {len(cached_tools)} cached tools for {service_name}")
            self._log_event(service_name, "get_tools", self.config[service_name].transport, start_ts, tool_count=len(cached_tools), cache="hit")
            return cached_tools
        # If cache empty attempt a one-shot direct fetch (non-blocking fast timeout rules apply)
        try:
            logger.info(f"Cache empty for {service_name}; attempting direct fetch now")
            direct = await self._get_tools_for_service_direct(service_name)
            if direct:
                self.tools_cache[service_name] = direct
            self._log_event(service_name, "get_tools", self.config[service_name].transport, start_ts, tool_count=len(direct), cache="miss")
            return direct
        except Exception as e:
            logger.warning(f"Direct fetch failed for {service_name}: {e}")
            self._log_event(service_name, "get_tools", self.config[service_name].transport, start_ts, status="error", error=type(e).__name__)
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
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth_provider):
                    # compute timeouts each attempt (provider may appear after retry)
                    conn_timeout, _ = self._compute_timeouts(service_name, 'sse', 'tools', oauth_provider is not None)
                    target_url = url
                    try:
                        async with sse_client(target_url, auth=oauth_provider, headers=headers, timeout=conn_timeout) as (r,w):
                            async with ClientSession(r,w) as session:
                                await session.initialize()
                                tr = await session.list_tools()
                                return self._format_tools(tr)
                    except Exception as e:
                        if '404' in str(e) and target_url.endswith('/sse'):
                            alt = target_url.rstrip('/sse') + '/mcp'
                            logger.warning(f"{service_name}: /sse 404 -> retry alt endpoint {alt}")
                            async with sse_client(alt, auth=oauth_provider, headers=headers, timeout=conn_timeout) as (r2,w2):
                                async with ClientSession(r2,w2) as session2:
                                    await session2.initialize()
                                    tr2 = await session2.list_tools()
                                    return self._format_tools(tr2)
                        raise
                return await self._run_with_oauth_retry(service_name, 'tools', 'sse', url, op)
                        
            elif service.transport == "http":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth_provider):
                    connection_timeout, read_timeout = self._compute_timeouts(service_name, 'http', 'tools', oauth_provider is not None)
                    async with streamablehttp_client(
                        url,
                        auth=oauth_provider,
                        headers=headers,
                        timeout=connection_timeout,
                        sse_read_timeout=read_timeout
                    ) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            tr = await session.list_tools()
                            return self._format_tools(tr)
                return await self._run_with_oauth_retry(service_name, 'tools', 'http', url, op)
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
        
        except Exception as e:
            logger.error(f"Error in _get_tools_for_service_impl for {service_name}: {e}")
            # Check if this is a timeout error for OAuth services
            if "timeout" in str(e).lower() or "ReadTimeout" in str(e):
                if service_name in self.oauth_providers or self._is_oauth_service(service.spec.get('url', '')):
                    logger.warning(f"Timeout detected for OAuth service {service_name}. This may be normal for dynamic services that perform OAuth discovery + tool listing.")
                else:
                    logger.warning(f"Timeout detected for {service_name}. Service may be slow or unresponsive.")
            raise  # Re-raise to be caught by the outer timeout handler
    
    async def get_service_capabilities(self, service_name: str) -> Dict[str, bool]:
        """
        Get server capabilities by connecting and calling initialize().
        Returns dict with capability flags: {'prompts': bool, 'resources': bool, 'tools': bool}
        """
        start_ts = time.perf_counter()
        if service_name not in self.config:
            self._log_event(service_name, "capabilities", None, start_ts, status="skip", reason="not_found")
            return {"prompts": False, "resources": False, "tools": False}
        
        # Check if service is enabled before attempting connection
        if self.service_states.get(service_name, "enabled") == "disabled":
            logger.debug(f"Service {service_name} is disabled, capabilities unavailable")
            self._log_event(service_name, "capabilities", self.config[service_name].transport, start_ts, status="skip", reason="disabled")
            return {"prompts": False, "resources": False, "tools": False}
        
        service = self.config[service_name]
        
        try:
            # Longer timeout for OAuth services during first capability probe
            needs_oauth = (service_name in self.oauth_providers or 
                           self._is_oauth_service(service.spec.get('url', '')))
            cap_timeout = 3.0
            if needs_oauth:
                cap_timeout = 15.0  # allow discovery / auth redirect headers
            return await asyncio.wait_for(
                self._get_service_capabilities_impl(service_name, service), 
                timeout=cap_timeout
            )
        except asyncio.TimeoutError:
            if needs_oauth:
                logger.warning(f"Timeout ({cap_timeout}s) getting capabilities for OAuth service {service_name} - may proceed after auth.")
            else:
                logger.warning(f"Timeout ({cap_timeout}s) getting capabilities for {service_name} - fast fail, non-blocking.")
            return {"prompts": False, "resources": False, "tools": False}
        except Exception as e:
            logger.debug(f"Failed to get capabilities for {service_name}: {e}")
            self._log_event(service_name, "capabilities", self.config[service_name].transport, start_ts, status="error", error=type(e).__name__)
            return {"prompts": False, "resources": False, "tools": False}
        finally:
            if service_name in self.config:
                caps = await self._safe_caps(service_name)
                self._log_event(service_name, "capabilities", self.config[service_name].transport, start_ts, **caps)

    async def _safe_caps(self, service_name: str) -> Dict[str, Any]:
        try:
            # Do not recurse; read from cache/quick call (no network). Here simply returns placeholders.
            return {}
        except Exception:
            return {}
    
    async def _get_service_capabilities_impl(self, service_name: str, service: ServiceConfig) -> Dict[str, bool]:
        """Internal implementation of get_service_capabilities"""
        try:
            if service.transport == "stdio":
                # STDIO transport
                server_params = StdioServerParameters(
                    command=service.spec['command'],
                    args=service.spec.get('args', []),
                    env=service.spec.get('env', {})
                )
                
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        init_result = await session.initialize()
                        
                        # Extract capabilities from initialization result
                        capabilities = {"prompts": False, "resources": False, "tools": False}
                        
                        if hasattr(init_result, 'capabilities') and init_result.capabilities:
                            server_caps = init_result.capabilities
                            capabilities["prompts"] = getattr(server_caps, 'prompts', None) is not None
                            capabilities["resources"] = getattr(server_caps, 'resources', None) is not None  
                            capabilities["tools"] = getattr(server_caps, 'tools', None) is not None
                        
                        return capabilities
                        
            elif service.transport == "sse":
                # SSE transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = None
                if service_name in self.oauth_providers or self._is_oauth_service(service.spec.get('url', '')):
                    token_storage = FileTokenStorage("oauth_tokens.json", service_name)
                    tokens = await token_storage.get_tokens()
                    if not tokens:
                        oauth_provider = self.oauth_providers.get(service_name) or self._create_dynamic_oauth_provider(service_name, url)
                
                connection_timeout = 3.0
                
                async with sse_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout
                ) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        init_result = await session.initialize()
                        
                        # Extract capabilities from initialization result
                        capabilities = {"prompts": False, "resources": False, "tools": False}
                        
                        if hasattr(init_result, 'capabilities') and init_result.capabilities:
                            server_caps = init_result.capabilities
                            capabilities["prompts"] = getattr(server_caps, 'prompts', None) is not None
                            capabilities["resources"] = getattr(server_caps, 'resources', None) is not None  
                            capabilities["tools"] = getattr(server_caps, 'tools', None) is not None
                        
                        return capabilities
                        
            elif service.transport == "http":
                # HTTP transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = self.oauth_providers.get(service_name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    oauth_provider = self._create_dynamic_oauth_provider(service_name, url)
                
                from datetime import timedelta
                connection_timeout = timedelta(seconds=3)
                sse_read_timeout = timedelta(seconds=3)
                
                async with streamablehttp_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        init_result = await session.initialize()
                        
                        # Extract capabilities from initialization result
                        capabilities = {"prompts": False, "resources": False, "tools": False}
                        
                        if hasattr(init_result, 'capabilities') and init_result.capabilities:
                            server_caps = init_result.capabilities
                            capabilities["prompts"] = getattr(server_caps, 'prompts', None) is not None
                            capabilities["resources"] = getattr(server_caps, 'resources', None) is not None  
                            capabilities["tools"] = getattr(server_caps, 'tools', None) is not None
                        
                        return capabilities
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
        
        except Exception as e:
            logger.debug(f"Error in _get_service_capabilities_impl for {service_name}: {e}")
            return {"prompts": False, "resources": False, "tools": False}

    async def get_prompts_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get prompts for a service using per-request session with capability checking"""
        start_ts = time.perf_counter()
        if service_name not in self.config:
            self._log_event(service_name, "list_prompts", None, start_ts, status="skip", reason="not_found")
            return []
        
        # Check if service is enabled before attempting connection
        if self.service_states.get(service_name, "enabled") == "disabled":
            logger.info(f"Service {service_name} is disabled, skipping prompt retrieval")
            return []
        
        # Check if service supports prompts capability first
        try:
            capabilities = await self.get_service_capabilities(service_name)
            if not capabilities.get("prompts", False):
                logger.debug(f"Service {service_name} does not support prompts capability")
                return []
        except Exception as e:
            logger.debug(f"Could not check capabilities for {service_name}: {e}")
            return []
        
        service = self.config[service_name]
        
        try:
            needs_oauth = (service_name in self.oauth_providers or 
                           self._is_oauth_service(service.spec.get('url', '')))
            prompts_timeout = 3.0 if not needs_oauth else 20.0
            return await asyncio.wait_for(
                self._get_prompts_for_service_impl(service_name, service), 
                timeout=prompts_timeout
            )
        except asyncio.TimeoutError:
            if needs_oauth:
                logger.warning(f"Timeout ({prompts_timeout}s) getting prompts for OAuth service {service_name} - may resolve post-auth.")
            else:
                logger.warning(f"Timeout ({prompts_timeout}s) getting prompts for {service_name} - fast fail, non-blocking.")
            return []
        except Exception as e:
            logger.error(f"Failed to get prompts for {service_name}: {e}")
            self._log_event(service_name, "list_prompts", service.transport, start_ts, status="error", error=type(e).__name__)
            return []
        finally:
            if service_name in self.config:
                # Log event with count if available
                try:
                    prompts_cached = self.tools_cache.get(service_name, [])  # reuse structure if needed
                    self._log_event(service_name, "list_prompts", self.config[service_name].transport, start_ts, prompt_count=len(prompts_cached) if prompts_cached else None)
                except Exception:
                    pass
    
    async def _get_prompts_for_service_impl(self, service_name: str, service: ServiceConfig) -> List[Dict[str, Any]]:
        """Internal implementation of get_prompts_for_service"""
        try:
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
                        
                        prompts_result = await session.list_prompts()
                        prompts = []
                        
                        for prompt in prompts_result.prompts:
                            prompt_dict = {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": []
                            }
                            
                            # Include prompt arguments if available
                            if hasattr(prompt, 'arguments') and prompt.arguments:
                                for arg in prompt.arguments:
                                    arg_dict = {
                                        "name": arg.name,
                                        "description": getattr(arg, 'description', ''),
                                        "required": getattr(arg, 'required', False)
                                    }
                                    prompt_dict["arguments"].append(arg_dict)
                            
                            prompts.append(prompt_dict)
                        
                        return prompts
                        
            elif service.transport == "sse":
                # SSE transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = None
                if service_name in self.oauth_providers or self._is_oauth_service(service.spec.get('url', '')):
                    token_storage = FileTokenStorage("oauth_tokens.json", service_name)
                    tokens = await token_storage.get_tokens()
                    if not tokens:
                        oauth_provider = self.oauth_providers.get(service_name) or self._create_dynamic_oauth_provider(service_name, url)
                
                connection_timeout = 3.0
                
                async with sse_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout
                ) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        
                        prompts_result = await session.list_prompts()
                        prompts = []
                        
                        for prompt in prompts_result.prompts:
                            prompt_dict = {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": []
                            }
                            
                            if hasattr(prompt, 'arguments') and prompt.arguments:
                                for arg in prompt.arguments:
                                    arg_dict = {
                                        "name": arg.name,
                                        "description": getattr(arg, 'description', ''),
                                        "required": getattr(arg, 'required', False)
                                    }
                                    prompt_dict["arguments"].append(arg_dict)
                            
                            prompts.append(prompt_dict)
                        
                        return prompts
                        
            elif service.transport == "http":
                # HTTP transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = self.oauth_providers.get(service_name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    oauth_provider = self._create_dynamic_oauth_provider(service_name, url)
                
                from datetime import timedelta
                connection_timeout = timedelta(seconds=3)
                sse_read_timeout = timedelta(seconds=3)
                
                async with streamablehttp_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        
                        prompts_result = await session.list_prompts()
                        prompts = []
                        
                        for prompt in prompts_result.prompts:
                            prompt_dict = {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": []
                            }
                            
                            if hasattr(prompt, 'arguments') and prompt.arguments:
                                for arg in prompt.arguments:
                                    arg_dict = {
                                        "name": arg.name,
                                        "description": getattr(arg, 'description', ''),
                                        "required": getattr(arg, 'required', False)
                                    }
                                    prompt_dict["arguments"].append(arg_dict)
                            
                            prompts.append(prompt_dict)
                        
                        return prompts
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
        
        except Exception as e:
            logger.error(f"Error in _get_prompts_for_service_impl for {service_name}: {e}")
            raise
    
    async def get_resources_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get resources for a service using per-request session with capability checking"""
        start_ts = time.perf_counter()
        if service_name not in self.config:
            self._log_event(service_name, "list_resources", None, start_ts, status="skip", reason="not_found")
            return []
        
        # Check if service is enabled before attempting connection
        if self.service_states.get(service_name, "enabled") == "disabled":
            logger.info(f"Service {service_name} is disabled, skipping resource retrieval")
            return []
        
        # Check if service supports resources capability first
        try:
            capabilities = await self.get_service_capabilities(service_name)
            if not capabilities.get("resources", False):
                logger.debug(f"Service {service_name} does not support resources capability")
                return []
        except Exception as e:
            logger.debug(f"Could not check capabilities for {service_name}: {e}")
            return []
        
        service = self.config[service_name]
        
        try:
            needs_oauth = (service_name in self.oauth_providers or 
                           self._is_oauth_service(service.spec.get('url', '')))
            resources_timeout = 3.0 if not needs_oauth else 20.0
            return await asyncio.wait_for(
                self._get_resources_for_service_impl(service_name, service), 
                timeout=resources_timeout
            )
        except asyncio.TimeoutError:
            if needs_oauth:
                logger.warning(f"Timeout ({resources_timeout}s) getting resources for OAuth service {service_name} - may resolve post-auth.")
            else:
                logger.warning(f"Timeout ({resources_timeout}s) getting resources for {service_name} - fast fail, non-blocking.")
            return []
        except Exception as e:
            logger.error(f"Failed to get resources for {service_name}: {e}")
            self._log_event(service_name, "list_resources", service.transport, start_ts, status="error", error=type(e).__name__)
            return []
        finally:
            if service_name in self.config:
                try:
                    self._log_event(service_name, "list_resources", self.config[service_name].transport, start_ts)
                except Exception:
                    pass
    
    async def _get_resources_for_service_impl(self, service_name: str, service: ServiceConfig) -> List[Dict[str, Any]]:
        """Internal implementation of get_resources_for_service"""
        try:
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
                        
                        resources_result = await session.list_resources()
                        resources = []
                        
                        for resource in resources_result.resources:
                            resource_dict = {
                                "uri": str(resource.uri),
                                "name": resource.name,
                                "description": getattr(resource, 'description', ''),
                                "mimeType": getattr(resource, 'mimeType', None)
                            }
                            
                            # Include annotations if available
                            if hasattr(resource, 'annotations') and resource.annotations:
                                resource_dict["annotations"] = resource.annotations
                            
                            resources.append(resource_dict)
                        
                        return resources
                        
            elif service.transport == "sse":
                # SSE transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = None
                if service_name in self.oauth_providers or self._is_oauth_service(service.spec.get('url', '')):
                    token_storage = FileTokenStorage("oauth_tokens.json", service_name)
                    tokens = await token_storage.get_tokens()
                    if not tokens:
                        oauth_provider = self.oauth_providers.get(service_name) or self._create_dynamic_oauth_provider(service_name, url)
                
                connection_timeout = 3.0
                
                async with sse_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout
                ) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        
                        resources_result = await session.list_resources()
                        resources = []
                        
                        for resource in resources_result.resources:
                            resource_dict = {
                                "uri": str(resource.uri),
                                "name": resource.name,
                                "description": getattr(resource, 'description', ''),
                                "mimeType": getattr(resource, 'mimeType', None)
                            }
                            
                            if hasattr(resource, 'annotations') and resource.annotations:
                                resource_dict["annotations"] = resource.annotations
                            
                            resources.append(resource_dict)
                        
                        return resources
                        
            elif service.transport == "http":
                # HTTP transport
                url = service.spec['url']
                headers = {}
                if MCP_PROTOCOL_VERSION:
                    headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
                
                oauth_provider = self.oauth_providers.get(service_name)
                if not oauth_provider and not self._has_embedded_auth(url):
                    oauth_provider = self._create_dynamic_oauth_provider(service_name, url)
                
                from datetime import timedelta
                connection_timeout = timedelta(seconds=3)
                sse_read_timeout = timedelta(seconds=3)
                
                async with streamablehttp_client(
                    url, 
                    auth=oauth_provider, 
                    headers=headers,
                    timeout=connection_timeout,
                    sse_read_timeout=sse_read_timeout
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        
                        resources_result = await session.list_resources()
                        resources = []
                        
                        for resource in resources_result.resources:
                            resource_dict = {
                                "uri": str(resource.uri),
                                "name": resource.name,
                                "description": getattr(resource, 'description', ''),
                                "mimeType": getattr(resource, 'mimeType', None)
                            }
                            
                            if hasattr(resource, 'annotations') and resource.annotations:
                                resource_dict["annotations"] = resource.annotations
                            
                            resources.append(resource_dict)
                        
                        return resources
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
        
        except Exception as e:
            logger.error(f"Error in _get_resources_for_service_impl for {service_name}: {e}")
            raise
    
    async def get_tools_info_for_service(self, service_name: str, tools_override: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Get tool count and token count for a specific service.
        Returns count information aligned with legacy MCP manager.
        """
        start_ts = time.perf_counter()
        if service_name not in self.config:
            self._log_event(service_name, "tools_info", None, start_ts, status="skip", reason="not_found")
            return {"tool_count": 0, "token_count": 0, "tools_error": "Service not found"}
        
        try:
            # Allow caller to provide tools to avoid duplicate retrieval (and duplicate logging)
            tools = tools_override if tools_override is not None else await self.get_tools_for_service(service_name)
            
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
                        # Rough estimate if no tokenizer or LiteLLM available
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
            self._log_event(service_name, "tools_info", self.config[service_name].transport, start_ts, status="error", error=type(e).__name__)
            return {"tool_count": 0, "token_count": 0, "tools_error": str(e)}
        finally:
            if service_name in self.config:
                try:
                    self._log_event(service_name, "tools_info", self.config[service_name].transport, start_ts)
                except Exception:
                    pass
    
    async def call_tool_for_service(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a service using per-request session with proper context management"""
        start_ts = time.perf_counter()
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
                        result = await session.call_tool(tool_name, arguments)
                        return self._format_tool_call_result(result)
                        
            elif service.transport == "sse":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth_provider):
                    conn_timeout, _ = self._compute_timeouts(service_name, 'sse', 'call', oauth_provider is not None)
                    target_url = url
                    try:
                        async with sse_client(target_url, auth=oauth_provider, headers=headers, timeout=conn_timeout) as (r,w):
                            async with ClientSession(r,w) as session:
                                await session.initialize()
                                res = await session.call_tool(tool_name, arguments)
                                return self._format_tool_call_result(res)
                    except Exception as e:
                        if '404' in str(e) and target_url.endswith('/sse'):
                            alt = target_url.rstrip('/sse') + '/mcp'
                            async with sse_client(alt, auth=oauth_provider, headers=headers, timeout=conn_timeout) as (r2,w2):
                                async with ClientSession(r2,w2) as session2:
                                    await session2.initialize()
                                    res2 = await session2.call_tool(tool_name, arguments)
                                    return self._format_tool_call_result(res2)
                        raise
                return await self._run_with_oauth_retry(service_name, 'call_tool', 'sse', url, op)
                        
            elif service.transport == "http":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth_provider):
                    conn_timeout, read_timeout = self._compute_timeouts(service_name, 'http', 'call', oauth_provider is not None)
                    async with streamablehttp_client(url, auth=oauth_provider, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            result = await session.call_tool(tool_name, arguments)
                            return self._format_tool_call_result(result)
                return await self._run_with_oauth_retry(service_name, 'call_tool', 'http', url, op)
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
                
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} for {service_name}: {e}")
            self._log_event(service_name, "call_tool", service.transport, start_ts, status="error", tool=tool_name, error=type(e).__name__)
            raise
        else:
            self._log_event(service_name, "call_tool", service.transport, start_ts, tool=tool_name)
    
    async def get_prompt_for_service(self, service_name: str, prompt_name: str, arguments: Dict[str, str] = None) -> Dict[str, Any]:
        """Get prompt content for a service using per-request session"""
        start_ts = time.perf_counter()
        if service_name not in self.config:
            raise ValueError(f"Service {service_name} not found")
        service = self.config[service_name]
        try:
            if service.transport == "stdio":
                server_params = StdioServerParameters(
                    command=service.spec['command'],
                    args=service.spec.get('args', []),
                    env=service.spec.get('env', {})
                )
                async with stdio_client(server_params) as (r,w):
                    async with ClientSession(r,w) as session:
                        await session.initialize()
                        result = await session.get_prompt(prompt_name, arguments or {})
                        return self._format_prompt(result)
            elif service.transport == "sse":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, _ = self._compute_timeouts(service_name, 'sse', 'prompt', oauth is not None)
                    async with sse_client(url, auth=oauth, headers=headers, timeout=conn_timeout) as (r,w):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            result = await session.get_prompt(prompt_name, arguments or {})
                            return self._format_prompt(result)
                return await self._run_with_oauth_retry(service_name, 'get_prompt', 'sse', url, op)
            elif service.transport == "http":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, read_timeout = self._compute_timeouts(service_name, 'http', 'prompt', oauth is not None)
                    async with streamablehttp_client(url, auth=oauth, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            result = await session.get_prompt(prompt_name, arguments or {})
                            return self._format_prompt(result)
                return await self._run_with_oauth_retry(service_name, 'get_prompt', 'http', url, op)
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_name} for {service_name}: {e}")
            self._log_event(service_name, "get_prompt", service.transport, start_ts, status="error", prompt=prompt_name, error=type(e).__name__)
            raise
        else:
            self._log_event(service_name, "get_prompt", service.transport, start_ts, prompt=prompt_name)
    
    
    async def read_resource_for_service(self, service_name: str, resource_uri: str) -> Dict[str, Any]:
        """Read resource content for a service using per-request session"""
        start_ts = time.perf_counter()
        if service_name not in self.config:
            raise ValueError(f"Service {service_name} not found")
        service = self.config[service_name]
        try:
            uri = AnyUrl(resource_uri)
            if service.transport == "stdio":
                server_params = StdioServerParameters(
                    command=service.spec['command'],
                    args=service.spec.get('args', []),
                    env=service.spec.get('env', {})
                )
                async with stdio_client(server_params) as (r,w):
                    async with ClientSession(r,w) as session:
                        await session.initialize()
                        result = await session.read_resource(uri)
                        return self._format_resource(result, resource_uri)
            if service.transport == "sse":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, _ = self._compute_timeouts(service_name, 'sse', 'resource', oauth is not None)
                    async with sse_client(url, auth=oauth, headers=headers, timeout=conn_timeout) as (r,w):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            result = await session.read_resource(uri)
                            return self._format_resource(result, resource_uri)
                return await self._run_with_oauth_retry(service_name, 'read_resource', 'sse', url, op)
            if service.transport == "http":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, read_timeout = self._compute_timeouts(service_name, 'http', 'resource', oauth is not None)
                    async with streamablehttp_client(url, auth=oauth, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            result = await session.read_resource(uri)
                            return self._format_resource(result, resource_uri)
                return await self._run_with_oauth_retry(service_name, 'read_resource', 'http', url, op)
            raise ValueError(f"Unsupported transport: {service.transport}")
        except Exception as e:
            logger.error(f"Failed to read resource {resource_uri} for {service_name}: {e}")
            self._log_event(service_name, "read_resource", service.transport, start_ts, status="error", resource=resource_uri, error=type(e).__name__)
            raise
        finally:
            if 'e' not in locals():
                self._log_event(service_name, "read_resource", service.transport, start_ts, resource=resource_uri)
    
    async def test_service_connection(self, service_name: str) -> bool:
        """Test connection to a service with proper context management"""
        start_ts = time.perf_counter()
        if service_name not in self.config:
            self._log_event(service_name, "test_connection", None, start_ts, status="skip", reason="not_found")
            return False
        
        service = self.config[service_name]
        
        
        try:
            return await self._test_service_connection_impl(service_name, service)
        except Exception as e:
            logger.error(f"Failed to test connection to {service_name}: {e}")
            self.service_states[service_name] = "disabled"
            self._log_event(service_name, "test_connection", service.transport, start_ts, status="error", error=type(e).__name__)
            return False
        else:
            self._log_event(service_name, "test_connection", service.transport, start_ts)
    
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
                        
                        self.service_states[service_name] = "enabled"
                        logger.info(f"Service {service_name} connection tested successfully")
                        return True
                        
            elif service.transport == "sse":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, _ = self._compute_timeouts(service_name, 'sse', 'init', oauth is not None)
                    async with sse_client(url, auth=oauth, headers=headers, timeout=conn_timeout) as (r,w):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            await session.list_tools()
                            self.service_states[service_name] = "enabled"
                            return True
                return await self._run_with_oauth_retry(service_name, 'test_connection', 'sse', url, op)
                        
            elif service.transport == "http":
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, read_timeout = self._compute_timeouts(service_name, 'http', 'init', oauth is not None)
                    async with streamablehttp_client(url, auth=oauth, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            await session.list_tools()
                            self.service_states[service_name] = "enabled"
                            return True
                return await self._run_with_oauth_retry(service_name, 'test_connection', 'http', url, op)
                        
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
            # Base configuration info - per-request pattern with enabled/disabled states
            service_status = self.service_states.get(name, "enabled")
            service_info = {
                "status": service_status,
                "enabled": service_status == "enabled",  # Flag for LLM response schema building
                "connected": False,  # Per-request pattern - no persistent connections
                "has_oauth": name in self.oauth_providers or self._is_oauth_service(service.spec.get('url', '')),
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
            "enabled_services": len([s for s in services.values() if s.get("enabled", True)]),
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

async def toggle_service_handler(request):
    """POST /toggle/{name} - Toggle service enabled/disabled state"""
    service_name = request.match_info['name']
    
    if service_name not in supervisor.config:
        return web.json_response({"error": f"Service {service_name} not found"}, status=404)
    
    try:
        data = await request.json()
        enabled = data.get('enabled', None)
        
        if enabled is None:
            # Toggle current state
            current_state = supervisor.service_states.get(service_name, "enabled")
            new_state = "disabled" if current_state == "enabled" else "enabled"
        else:
            # Set explicit state
            new_state = "enabled" if enabled else "disabled"
        
        supervisor.service_states[service_name] = new_state
        
        # Update tools cache when service state changes
        supervisor._update_service_cache(service_name, new_state == "enabled")
        
        # If enabling, test connection to validate it works
        connection_test_success = True
        if new_state == "enabled":
            try:
                connection_test_success = await supervisor.test_service_connection(service_name)
                if not connection_test_success:
                    supervisor.service_states[service_name] = "disabled"
                    # Update cache again since we're disabling due to failed test
                    supervisor._update_service_cache(service_name, False)
                    return web.json_response({
                        "success": False,
                        "service": service_name,
                        "enabled": False,
                        "message": "Service connection test failed, keeping disabled"
                    })
            except Exception as e:
                supervisor.service_states[service_name] = "disabled"
                # Update cache again since we're disabling due to failed test
                supervisor._update_service_cache(service_name, False)
                return web.json_response({
                    "success": False,
                    "service": service_name, 
                    "enabled": False,
                    "error": f"Connection test failed: {str(e)}"
                })
        
        return web.json_response({
            "success": True,
            "service": service_name,
            "enabled": new_state == "enabled",
            "status": new_state,
            "previous_state": current_state if enabled is None else None
        })
    except Exception as e:
        logger.error(f"Toggle service error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def list_tools_handler(request):
    """GET /list_tools - Get all tools from all enabled services (Fractalic compatibility endpoint)
    Optimized to avoid duplicate tool retrieval + duplicate logging by passing tools into tools_info.
    """
    try:
        all_tools = []
        total_token_count = 0
        enabled_services = [name for name, state in supervisor.service_states.items() if state == 'enabled']

        for service_name in enabled_services:
            try:
                tools = await supervisor.get_tools_for_service(service_name)
                if not tools:
                    continue
                tools_info = await supervisor.get_tools_info_for_service(service_name, tools_override=tools)
                for tool in tools:
                    tool_with_service = {
                        **tool,
                        "name": f"{service_name}.{tool['name']}",
                        "service": service_name,
                        "original_name": tool['name']
                    }
                    all_tools.append(tool_with_service)
                total_token_count += tools_info.get('token_count', 0)
            except Exception as e:
                logger.warning(f"Failed to get tools for {service_name}: {e}")

        return web.json_response({
            'tools': all_tools,
            'count': len(all_tools),
            'total_token_count': total_token_count,
            'services_count': len(enabled_services)
        })
    except Exception as e:
        logger.error(f"List tools error: {e}")
        return web.json_response({'error': str(e)}, status=500)

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

async def get_capabilities_handler(request):
    """GET /capabilities/{name} - Get capabilities for a service"""
    service_name = request.match_info['name']
    try:
        capabilities = await supervisor.get_service_capabilities(service_name)
        return web.json_response({
            "service": service_name,
            "capabilities": capabilities
        })
    except Exception as e:
        logger.error(f"Get capabilities error for {service_name}: {e}")
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

async def list_prompts_handler(request):
    """GET /list_prompts - Get all prompts from all enabled services"""
    try:
        services_response = {}
        
        for service_name, service in supervisor.config.items():
            # Check if service is enabled
            if supervisor.service_states.get(service_name, "enabled") == "disabled":
                continue
                
            try:
                prompts = await supervisor.get_prompts_for_service(service_name)
                if prompts:
                    services_response[service_name] = {"prompts": prompts}
            except Exception as e:
                logger.warning(f"Failed to get prompts for {service_name}: {e}")
                services_response[service_name] = {"error": str(e), "prompts": []}
        
        return web.json_response(services_response)
    except Exception as e:
        logger.error(f"List prompts error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def get_prompt_handler(request):
    """POST /prompt/{service}/{prompt} - Get prompt content with arguments"""
    service_name = request.match_info['service']
    prompt_name = request.match_info['prompt']
    
    try:
        data = await request.json()
        arguments = data.get('arguments', {})
        
        result = await supervisor.get_prompt_for_service(service_name, prompt_name, arguments)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Get prompt error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def list_resources_handler(request):
    """GET /list_resources - Get all resources from all enabled services"""
    try:
        services_response = {}
        
        for service_name, service in supervisor.config.items():
            # Check if service is enabled
            if supervisor.service_states.get(service_name, "enabled") == "disabled":
                continue
                
            try:
                resources = await supervisor.get_resources_for_service(service_name)
                if resources:
                    services_response[service_name] = {"resources": resources}
            except Exception as e:
                logger.warning(f"Failed to get resources for {service_name}: {e}")
                services_response[service_name] = {"error": str(e), "resources": []}
        
        return web.json_response(services_response)
    except Exception as e:
        logger.error(f"List resources error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def read_resource_handler(request):
    """POST /resource/{service}/read - Read resource content by URI"""
    service_name = request.match_info['service']
    
    try:
        data = await request.json()
        resource_uri = data.get('uri')
        
        if not resource_uri:
            return web.json_response({"error": "Missing 'uri' parameter"}, status=400)
        
        result = await supervisor.read_resource_for_service(service_name, resource_uri)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Read resource error: {e}")
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

async def oauth_reset_handler(request):
    """POST /oauth/reset/{service} - Delete stored tokens and reset auth state"""
    service_name = request.match_info['service']
    if service_name not in supervisor.config:
        return web.json_response({"error": "Service not found"}, status=404)
    try:
        success = await supervisor.reset_oauth_tokens(service_name)
        return web.json_response({
            "service": service_name,
            "reset": success
        }, status=200 if success else 500)
    except Exception as e:
        logger.error(f"OAuth reset error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def oauth_authorize_handler(request):
    """POST /oauth/authorize/{service} - Proactively create provider and trigger authorization via test connection."""
    service_name = request.match_info['service']
    if service_name not in supervisor.config:
        return web.json_response({"error": "Service not found"}, status=404)
    try:
        ensured = supervisor.ensure_oauth_provider(service_name)
        if not ensured:
            return web.json_response({"error": "Could not create OAuth provider"}, status=500)
        # Trigger flow: test connection (will 401 -> browser redirect)
        started = await supervisor.test_service_connection(service_name)
        return web.json_response({
            "service": service_name,
            "authorization_triggered": True,
            "connection_success": started
        })
    except Exception as e:
        logger.error(f"OAuth authorize error: {e}")
        return web.json_response({"error": str(e)}, status=500)

def create_app():
    """Create aiohttp application"""
    app = web.Application()
    
    # API routes
    app.router.add_get('/status', status_handler)
    app.router.add_get('/list_tools', list_tools_handler)  # Fractalic compatibility endpoint
    app.router.add_post('/toggle/{name}', toggle_service_handler)
    app.router.add_get('/tools/{name}', get_tools_handler)
    app.router.add_get('/capabilities/{name}', get_capabilities_handler)
    app.router.add_post('/call/{service}/{tool}', call_tool_handler)
    
    # MCP Full Feature Set routes
    app.router.add_get('/list_prompts', list_prompts_handler)
    app.router.add_post('/prompt/{service}/{prompt}', get_prompt_handler)
    app.router.add_get('/list_resources', list_resources_handler)
    app.router.add_post('/resource/{service}/read', read_resource_handler)
    
    app.router.add_post('/oauth/start/{service}', oauth_start_handler)
    app.router.add_post('/oauth/reset/{service}', oauth_reset_handler)
    app.router.add_post('/oauth/authorize/{service}', oauth_authorize_handler)
    
    # OAuth callback route
    app.router.add_get('/oauth/callback', oauth_callback_handler)
    
    return app

async def serve(port: int = 5859, host: str = '0.0.0.0'):
    """Run the HTTP server on specified host/port (default 0.0.0.0:5859)"""
    import signal
    # Force working directory to repository root so relative paths (config, tokens) are consistent
    try:
        if os.getcwd() != str(ROOT_DIR):
            os.chdir(ROOT_DIR)
            logger.info(f"Changed working directory to repo root: {ROOT_DIR}")
    except Exception as _cwd_e:
        logger.warning(f"Could not change working directory to root: {_cwd_e}")
    
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    logger.info(f"MCP Manager V2 started on http://{host}:{port}")
    logger.info("Per-request session pattern - no persistent connections")
    
    # Populate initial tools cache asynchronously so server can accept requests immediately
    async def _background_cache():
        try:
            await supervisor._populate_initial_cache()
        except Exception as e:
            logger.error(f"Initial cache population failed: {e}")
    asyncio.create_task(_background_cache())
    logger.info("Initial cache population started in background")
    
    # Proper signal handling
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, shutting down...")
        raise KeyboardInterrupt()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        await runner.cleanup()
        logger.info("Server shutdown complete")

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
    serve_parser = subparsers.add_parser('serve', help='Start HTTP server')
    serve_parser.add_argument('--port', type=int, default=5859, help='Port to listen on (default: 5859)')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host/interface to bind (default: 0.0.0.0)')
    
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
        asyncio.run(serve(port=getattr(args, 'port', 5859), host=getattr(args, 'host', '0.0.0.0')))
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
