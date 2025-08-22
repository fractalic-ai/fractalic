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
import aiohttp_cors
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

# MCP Protocol Version - Let SDK handle negotiation automatically
MCP_PROTOCOL_VERSION = None

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
                transport = 'streamable-http'
        
        return cls(
            name=name,
            transport=transport,
            spec=config,
            has_oauth=config.get('has_oauth', False),
            enabled=config.get('enabled', True)
        )

class FileTokenStorage(TokenStorage):
    """File-based token storage for OAuth - service-specific

    NOTE: Previous revisions had a stray 'self.on_set_tokens' at class scope causing NameError.
    This implementation keeps all instance assignments inside __init__.
    """

    def __init__(self, file_path: str, service_name: str = "default"):
        p = _Path(file_path)
        if not p.is_absolute():
            p = ROOT_DIR / p
        self.file_path = p
        self.service_name = service_name
        # Hook populated by supervisor to schedule refresh after token save
        self.on_set_tokens = None  # type: ignore[attr-defined]
    
    async def get_tokens(self) -> Optional[OAuthToken]:
        """Load tokens from file for specific service (no manual refresh; SDK provider handles refresh)."""
        logger.info(f"Loading tokens for service: {self.service_name} from {self.file_path}")
        
        if not self.file_path.exists():
            logger.warning(f"Token file {self.file_path} not found")
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # Validate JSON structure
            if not isinstance(data, dict):
                logger.error(f"Token file has invalid structure: expected dict, got {type(data)}")
                return None
                
            logger.info(f"Token file loaded, available services: {list(data.keys())}")
            
            # Load service-specific token or fall back to default
            token_key = self.service_name if self.service_name in data else "default"
            if token_key in data:
                token_data = data[token_key]
                
                # Validate required fields
                if not isinstance(token_data, dict) or 'access_token' not in token_data:
                    logger.error(f"Invalid token structure for {token_key}: missing access_token")
                    return None
                
                access_token = token_data.get('access_token')
                if not access_token:
                    logger.error(f"Empty access_token for {token_key}")
                    return None
                
                logger.info(f"Found tokens for {token_key}, access_token starts with: {access_token[:20]}...")
                return OAuthToken(
                    access_token=access_token,
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
                try:
                    with open(self.file_path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    # Corrupted file: back it up and start fresh
                    try:
                        corrupt_backup = self.file_path.with_suffix('.invalid')
                        self.file_path.rename(corrupt_backup)
                        logger.warning(f"Corrupted token file backed up to {corrupt_backup}")
                    except Exception:
                        pass
                    data = {}
            
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
            
            self._atomic_write_json(data)
                
            logger.info(f"OAuth tokens saved successfully for {self.service_name}")
            if self.on_set_tokens:
                try:
                    self.on_set_tokens()
                except Exception:
                    logger.debug("on_set_tokens hook failed", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to save tokens for {self.service_name}: {e}")

    async def replace_access_token(self, access_token: str, expires_in: Optional[int], scope: Optional[str], refresh_token: Optional[str] = None):
        """Update access token fields during refresh, optionally updating refresh token if server provides new one."""
        try:
            if not self.file_path.exists():
                return
            try:
                with open(self.file_path,'r') as f:
                    data=json.load(f)
            except Exception:
                return
            svc=data.get(self.service_name)
            if not svc:
                return
            import time as _time
            svc['access_token']=access_token
            if expires_in is not None:
                svc['expires_in']=expires_in
            if scope is not None:
                svc['scope']=scope
            if refresh_token is not None:
                svc['refresh_token']=refresh_token
                logger.info(f"Server provided new refresh token for {self.service_name} (refresh token rotation)")
            else:
                logger.info(f"Server reused existing refresh token for {self.service_name} (refresh token reuse pattern)")
            svc['obtained_at']=_time.time()
            data[self.service_name]=svc
            self._atomic_write_json(data)
            logger.info(f"Refreshed access token stored for {self.service_name}")
        except Exception as e:
            logger.warning(f"Failed updating refreshed token for {self.service_name}: {e}")
    
    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        # Load persisted dynamic client registration (if any)
        try:
            if not self.file_path.exists():
                return None
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            svc = data.get(self.service_name)
            if not svc:
                return None
            ci = svc.get('client_info')
            if not ci:
                return None
            # Reconstruct object (fields may evolve; use kwargs subset)
            fields = {}
            for k in ('client_id','client_secret','client_id_issued_at','client_secret_expires_at','redirect_uris','grant_types','response_types','scope','token_endpoint_auth_method'):
                if k in ci:
                    fields[k] = ci[k]
            return OAuthClientInformationFull(**fields)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Failed loading client_info for {self.service_name}: {e}")
            return None
    
    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        # Persist dynamic client registration details alongside tokens
        try:
            data = {}
            if self.file_path.exists():
                try:
                    with open(self.file_path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            svc = data.get(self.service_name) or {}
            # Serialize pydantic model / dataclass generically
            ci_dict = {}
            for attr in dir(client_info):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(client_info, attr)
                except Exception:
                    continue
                if callable(val):
                    continue
                # Convert non-primitive / Pydantic types to str
                if isinstance(val, (str, int, float, bool)) or val is None:
                    ci_dict[attr] = val
                elif isinstance(val, (list, tuple, set)):
                    conv_list = []
                    for item in val:
                        if isinstance(item, (str, int, float, bool)) or item is None:
                            conv_list.append(item)
                        else:
                            conv_list.append(str(item))
                    ci_dict[attr] = conv_list
                else:
                    # Fallback to string representation
                    ci_dict[attr] = str(val)
            svc['client_info'] = ci_dict
            data[self.service_name] = svc
            self._atomic_write_json(data)
            logger.info(f"Persisted client_info for {self.service_name}")
        except Exception as e:
            logger.warning(f"Failed persisting client_info for {self.service_name}: {e}")

    async def delete_tokens(self) -> None:
        """Delete stored tokens for this service (force re-auth/refresh)."""
        try:
            if not self.file_path.exists():
                return
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            if self.service_name in data:
                del data[self.service_name]
                self._atomic_write_json(data)
                logger.info(f"Deleted tokens for {self.service_name} (forced invalidation)")
        except Exception as e:
            logger.error(f"Failed to delete tokens for {self.service_name}: {e}")

    # --------------- Internal helpers ---------------
    def _atomic_write_json(self, data: Dict[str, Any]):
        """Write JSON atomically to avoid truncated files if serialization fails.
        Writes to a temporary file then renames into place.
        """
        try:
            import tempfile, os
            tmp_fd, tmp_path = tempfile.mkstemp(prefix='oauth_tokens_', suffix='.tmp', dir=str(self.file_path.parent))
            try:
                with os.fdopen(tmp_fd, 'w') as tmp_f:
                    json.dump(data, tmp_f, indent=2)
                os.replace(tmp_path, self.file_path)
            except Exception:
                # Cleanup tmp file if something went wrong
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise
        except Exception as e:
            logger.error(f"Atomic write failed for {self.file_path}: {e}")


# Global callback data for OAuth
oauth_callback_data = {}

class OAuthPending(Exception):
    """Signal that OAuth is already in progress / awaiting user completion."""
    pass

class MCPSupervisorV2:
    """MCP Service Manager using official Python SDK - Per-Request Sessions"""
    def __init__(self):
        """Constructor sets up in-memory state. No network I/O here."""
        # Core config/state
        self.config: Dict[str, ServiceConfig] = self.load_config()
        self.oauth_providers: Dict[str, OAuthClientProvider] = {}
        self.token_storages: Dict[str, FileTokenStorage] = {}
        self.service_states: Dict[str, str] = {}
        self.tools_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.prompts_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.resources_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.capabilities_cache: Dict[str, Dict[str, bool]] = {}
        # Auth / throttling bookkeeping
        self.oauth_attempt_timestamps: Dict[str, float] = {}
        self.last_unauthorized: Dict[str, float] = {}
        self.oauth_redirect_times: Dict[str, float] = {}
        self.oauth_in_progress: Dict[str, bool] = {}
        self.oauth_last_pending_log: Dict[str, float] = {}
        # OAuth concurrency + refresh tracking
        self.oauth_futures: Dict[str, asyncio.Future] = {}
        self.oauth_refresh_tasks: Dict[str, asyncio.Task] = {}
        self.oauth_refresh_run_at: Dict[str, float] = {}
        self.oauth_refresh_status: Dict[str, str] = {}
        self.oauth_refresh_error: Dict[str, str] = {}

        # Initialize provider objects where heuristically obvious
        self._init_oauth_providers()

        # Load persisted redirect timestamps (cooldown across restarts)
        self._load_redirect_state()

        # Initialize service enabled states
        for service_name, svc in self.config.items():
            self.service_states[service_name] = (
                "enabled" if getattr(svc, 'enabled', True) else "disabled"
            )
        # tools_cache gets filled asynchronously after server start
    
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
            # Create OAuth providers for services that likely need OAuth (heuristic)
            url = service.spec.get('url', '')
            if self._is_oauth_service(url):
                # Skip OAuth provider creation for services with embedded auth
                if self._has_embedded_auth(url):
                    logger.info(f"Skipping OAuth provider creation for {name} - has embedded auth")
                    continue
                self._create_oauth_provider(name, service)
                logger.info(f"OAuth provider created for detected OAuth service: {name}")

        # If tokens already exist on disk for a service, proactively create provider so they are used on first attempt.
        token_file = ROOT_DIR / 'oauth_tokens.json'
        if token_file.exists():
            try:
                with open(token_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for name, service in self.config.items():
                        if name in self.oauth_providers:
                            continue
                        if name in data and 'access_token' in data[name]:
                            url = service.spec.get('url', '')
                            base = self._base_server_url(url)
                            provider = self._create_dynamic_oauth_provider(name, base)
                            # Attempt to restore persisted client_info if present (ensures consistent client identity across restarts)
                            try:
                                svc_entry = data.get(name, {})
                                ci = svc_entry.get('client_info') if isinstance(svc_entry, dict) else None
                                if ci and provider is not None:
                                    # provider.storage is FileTokenStorage; we call set_client_info only if not already set internally
                                    # Reconstruct minimal model accepted by OAuthClientInformationFull; ignore unknown keys
                                    fields = {}
                                    for k in ('client_id','client_secret','client_id_issued_at','client_secret_expires_at','redirect_uris','grant_types','response_types','scope','token_endpoint_auth_method'):
                                        if k in ci:
                                            fields[k] = ci[k]
                                    try:
                                        ci_obj = OAuthClientInformationFull(**fields)  # type: ignore[arg-type]
                                        # Persist back through storage for uniform access later
                                        # (storage.set_client_info overwrites existing file entry with normalized form)
                                        asyncio.create_task(provider.storage.set_client_info(ci_obj))  # type: ignore[attr-defined]
                                        logger.info(f"Restored client_info for {name} (client_id={ci.get('client_id')})")
                                    except Exception:
                                        logger.debug(f"Failed reconstructing client_info object for {name}", exc_info=True)
                            except Exception:
                                logger.debug(f"Client info restoration failed for {name}", exc_info=True)
                            logger.info(f"OAuth provider created from existing tokens for service: {name}")
            except Exception:
                logger.debug("Failed proactive provider creation from existing tokens", exc_info=True)

        # Remaining services rely on automatic discovery via 401 challenges.
    # -------- Persistent OAuth redirect cooldown state --------
    def _redirect_state_path(self) -> Path:
        return ROOT_DIR / "oauth_redirect_state.json"

    def _load_redirect_state(self):
        path = self._redirect_state_path()
        if not path.exists():
            return
        try:
            with open(path,'r') as f:
                data=json.load(f)
            if isinstance(data, dict):
                for k,v in data.items():
                    try:
                        self.oauth_redirect_times[k]=float(v)
                    except Exception:
                        continue
                if self.oauth_redirect_times:
                    logger.info(f"Loaded OAuth redirect cooldown state for {len(self.oauth_redirect_times)} services")
        except Exception as e:
            logger.warning(f"Failed loading redirect state: {e}")

    def _persist_redirect_state(self):
        path=self._redirect_state_path()
        try:
            tmp=path.with_suffix('.tmp')
            with open(tmp,'w') as f:
                json.dump(self.oauth_redirect_times,f,indent=2)
            os.replace(tmp,path)
        except Exception as e:
            logger.debug(f"Failed persisting redirect state: {e}")
    
    async def _populate_initial_cache(self, comprehensive: bool = False):
        """
        Populate cache on startup for all enabled services.
        
        Args:
            comprehensive: If True, caches tools, prompts, and resources. 
                          If False, only caches tools for faster startup.
        """
        cache_type = "comprehensive" if comprehensive else "tools"
        logger.info(f"Starting initial {cache_type} cache population...")
        cache_start_time = time.time()
        
        # Load data for all enabled services in parallel
        tasks = []
        for service_name in self.config:
            if self.service_states.get(service_name, "enabled") == "enabled":
                if comprehensive:
                    task = asyncio.create_task(self._cache_all_data_for_service(service_name))
                else:
                    task = asyncio.create_task(self._cache_tools_for_service(service_name))
                tasks.append(task)
        
        # Wait for all services to complete (with individual timeouts handled in cache methods)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        cache_end_time = time.time()
        cached_services = len([name for name in self.tools_cache if self.tools_cache[name]])
        total_tools = sum(len(tools) for tools in self.tools_cache.values())
        
        if comprehensive:
            cached_prompts = len([name for name in self.prompts_cache if self.prompts_cache[name]])
            total_prompts = sum(len(prompts) for prompts in self.prompts_cache.values())
            cached_resources = len([name for name in self.resources_cache if self.resources_cache[name]])
            total_resources = sum(len(resources) for resources in self.resources_cache.values())
            
            logger.info(f"Initial comprehensive cache populated in {cache_end_time - cache_start_time:.2f}s: "
                       f"{cached_services} services, {total_tools} tools, {total_prompts} prompts, {total_resources} resources")
        else:
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
    
    async def _cache_all_data_for_service(self, service_name: str):
        """
        Cache tools, prompts, resources, and capabilities for a single service with error handling.
        This comprehensive caching reduces subsequent API call overhead.
        """
        try:
            # Cache tools
            tools = await self._get_tools_for_service_direct(service_name)
            self.tools_cache[service_name] = tools
            logger.debug(f"Cached {len(tools)} tools for {service_name}")
        except Exception as e:
            logger.warning(f"Failed to cache tools for {service_name}: {e}")
            self.tools_cache[service_name] = []
            
        try:
            # Cache prompts
            prompts = await self.get_prompts_for_service(service_name)
            self.prompts_cache[service_name] = prompts
            logger.debug(f"Cached {len(prompts)} prompts for {service_name}")
        except Exception as e:
            logger.warning(f"Failed to cache prompts for {service_name}: {e}")
            self.prompts_cache[service_name] = []
            
        try:
            # Cache resources
            resources = await self.get_resources_for_service(service_name)
            self.resources_cache[service_name] = resources
            logger.debug(f"Cached {len(resources)} resources for {service_name}")
        except Exception as e:
            logger.warning(f"Failed to cache resources for {service_name}: {e}")
            self.resources_cache[service_name] = []
            
        try:
            # Cache capabilities
            capabilities = await self._get_service_capabilities_impl(service_name, self.config[service_name])
            self.capabilities_cache[service_name] = capabilities
            logger.debug(f"Cached capabilities for {service_name}: {capabilities}")
        except Exception as e:
            logger.warning(f"Failed to cache capabilities for {service_name}: {e}")
            self.capabilities_cache[service_name] = {"prompts": False, "resources": False, "tools": False}
    
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
        
        # In-flight dedupe
        if not hasattr(self, '_tools_inflight'):
            self._tools_inflight = {}
        existing = self._tools_inflight.get(service_name)
        if existing and not existing.done():
            try:
                return await asyncio.wait_for(existing, timeout=30)
            except Exception:
                pass

        async def _fetch():
            stage = {'entered': False, 'initialized': False}
            try:
                timeout = 3.0
                needs_oauth = (service_name in self.oauth_providers or 
                              self._is_oauth_service(service.spec.get('url', '')))
                if needs_oauth:
                    token_storage = FileTokenStorage("oauth_tokens.json", service_name)
                    tokens = await token_storage.get_tokens()
                    if not tokens:
                        now = time.time()
                        last = self.oauth_attempt_timestamps.get(service_name)
                        if last and (now - last) < 90:
                            logger.info(f"Skipping new OAuth attempt for {service_name}; last attempt {(now-last):.1f}s ago")
                            return []
                        self.oauth_attempt_timestamps[service_name] = now
                        timeout = 120.0
                        logger.debug(f"Using extended timeout ({timeout}s) for initial OAuth flow for {service_name}")
                    else:
                        if service.transport == 'sse':
                            timeout = 60.0  # Increased for replicate debugging
                        else:
                            timeout = 12.0
                        logger.debug(f"Using extended post-auth timeout ({timeout}s) for {service_name} transport={service.transport} with tokens")
                return await asyncio.wait_for(self._get_tools_for_service_impl(service_name, service, stage), timeout=timeout)
            except asyncio.TimeoutError:
                classification = 'timeout'
                if stage['entered'] and not stage['initialized']:
                    classification = 'init_timeout'
                elif stage['initialized']:
                    classification = 'list_timeout'
                logger.warning(f"Tools {classification} for {service_name} (entered={stage['entered']} initialized={stage['initialized']})")
                return []
            except Exception as e:
                logger.error(f"Failed to get tools directly for {service_name}: {e}")
                return []
            finally:
                fut2 = self._tools_inflight.get(service_name)
                if fut2 and fut2.done():
                    self._tools_inflight.pop(service_name, None)

        loop = asyncio.get_running_loop()
        fut = loop.create_task(_fetch())
        self._tools_inflight[service_name] = fut
        return await fut
    
    def _update_service_cache(self, service_name: str, enabled: bool):
        """
        Update cache when service is enabled/disabled.
        Adds tools/prompts/resources to cache when enabled, removes when disabled.
        """
        if enabled:
            # Service enabled - cache all data asynchronously
            asyncio.create_task(self._cache_all_data_for_service(service_name))
            logger.info(f"Service {service_name} enabled - updating cache")
        else:
            # Service disabled - remove from all caches
            removed_items = 0
            if service_name in self.tools_cache:
                removed_items += len(self.tools_cache[service_name])
                del self.tools_cache[service_name]
            if service_name in self.prompts_cache:
                removed_items += len(self.prompts_cache[service_name])
                del self.prompts_cache[service_name]
            if service_name in self.resources_cache:
                removed_items += len(self.resources_cache[service_name])
                del self.resources_cache[service_name]
            if service_name in self.capabilities_cache:
                del self.capabilities_cache[service_name]
            logger.info(f"Service {service_name} disabled - removed {removed_items} items from cache")

    # -------- Helper consolidation (auth, sessions, formatting) --------
    
    def _get_server_supported_scope(self, server_url: str) -> str:
        """Get supported scopes from OAuth authorization server discovery endpoint."""
        try:
            # Try to get OAuth discovery info
            discovery_url = f"{server_url}/.well-known/oauth-authorization-server"
            import httpx
            response = httpx.get(discovery_url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                scopes = data.get('scopes_supported', [])
                if scopes:
                    return ' '.join(scopes)
                # If no scopes declared, use empty scope for Notion
                elif 'notion.com' in server_url:
                    return ''
        except Exception:
            pass
        
        # Default fallback for services like Notion that use standard scopes
        return 'read write'

    def _protocol_headers(self) -> Dict[str, str]:
        headers = {}
        if MCP_PROTOCOL_VERSION:
            headers['MCP-Protocol-Version'] = MCP_PROTOCOL_VERSION
        # Set Accept header for streamable HTTP like Inspector does
        headers['Accept'] = 'text/event-stream, application/json'
        return headers

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

    def debug_oauth_state(self, service_name: str) -> Dict[str, Any]:
        """Return internal OAuth state snapshot for diagnostics (no secrets)."""
        state: Dict[str, Any] = {}
        try:
            provider = self.oauth_providers.get(service_name)
            storage = self.token_storages.get(service_name)
            tokens_present = False
            token_meta: Dict[str, Any] = {}
            if storage and storage.file_path.exists():
                try:
                    with open(storage.file_path, 'r') as f:
                        data = json.load(f)
                    svc = data.get(service_name) or {}
                    if 'access_token' in svc:
                        tokens_present = True
                        at = svc.get('access_token','')
                        # Mask access token
                        token_meta['access_token_preview'] = f"{at[:6]}...{at[-4:]}" if len(at) > 10 else 'short'
                    for k in ('expires_in','obtained_at','scope'):
                        if k in svc:
                            token_meta[k] = svc[k]
                    if 'client_info' in svc:
                        ci = svc['client_info']
                        token_meta['client_id'] = ci.get('client_id')
                except Exception as e:
                    token_meta['read_error'] = type(e).__name__
            now = time.time()
            state.update({
                'service': service_name,
                'provider_exists': provider is not None,
                'oauth_in_progress': self.oauth_in_progress.get(service_name, False),
                'tokens_present': tokens_present,
                'token_meta': token_meta,
                'last_unauthorized_delta_s': (round(now - self.last_unauthorized[service_name],1) if service_name in self.last_unauthorized else None),
                'redirect_recent_delta_s': (round(now - self.oauth_redirect_times[service_name],1) if service_name in self.oauth_redirect_times else None),
                'refresh_scheduled_in_s': (round(self.oauth_refresh_run_at[service_name]-now,1) if service_name in self.oauth_refresh_run_at else None),
                'state': self.service_states.get(service_name),
            })
        except Exception as e:
            state['error'] = f"debug_oauth_state_failed:{type(e).__name__}"
        return state

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
            # Clear redirect cooldown to allow immediate new browser open
            try:
                if service_name in self.oauth_redirect_times:
                    self.oauth_redirect_times.pop(service_name, None)
                    self._persist_redirect_state()
            except Exception:
                logger.debug("Failed clearing redirect cooldown on reset", exc_info=True)
            return True
        return False

    def clear_redirect_cooldown(self, service_name: str) -> bool:
        """Clear redirect cooldown for a service (force next auth window)."""
        try:
            if service_name in self.oauth_redirect_times:
                self.oauth_redirect_times.pop(service_name, None)
                self._persist_redirect_state()
            return True
        except Exception:
            return False

    async def _run_with_oauth_retry(self, service_name: str, purpose: str, transport: str, url: str, op_coro_factory):
        """Execute an async operation with at most one OAuth-driven retry on 401.
        Logic:
          1. Attempt operation with existing provider (if any).
          2. If 401 and no provider: create provider from base server URL, retry once.
          3. If 401 and provider exists: delete tokens (if any) to force re-auth, retry once.
          4. Record cooldown to avoid tight unauthorized loops (60s).
        """
        # Clear stale in-progress flag if tokens already present (e.g., after restart)
        if self.oauth_in_progress.get(service_name):
            try:
                storage_chk = self.token_storages.get(service_name)
                if storage_chk:
                    existing = await storage_chk.get_tokens()
                    if existing:
                        self._log_event(service_name, purpose, transport, time.perf_counter(), status='oauth_in_progress_cleared_stale')
                        self.oauth_in_progress[service_name] = False
            except Exception:
                pass
        
        # Check for stuck OAuth flow (> 5 minutes)
        if self.oauth_in_progress.get(service_name):
            start_time = self.oauth_attempt_timestamps.get(service_name, time.time())
            elapsed = time.time() - start_time
            if elapsed > 300:  # 5 minutes timeout
                logger.warning(f"OAuth flow stuck for {service_name} after {elapsed:.1f}s, clearing flag")
                self.oauth_in_progress[service_name] = False
                self.oauth_attempt_timestamps.pop(service_name, None)
                self._log_event(service_name, purpose, transport, time.perf_counter(), status='oauth_timeout_cleared', elapsed=elapsed)
        
        # Concurrency guard: if still in progress, surface pending quickly
        if self.oauth_in_progress.get(service_name):
            monotonic_now = time.perf_counter()
            last_log = self.oauth_last_pending_log.get(service_name, 0)
            # Store last log time in wall clock dict but compare using perf counter delta where possible
            if (time.time() - last_log) > 15:  # keep original semantics for cooldown using wall time
                self.oauth_last_pending_log[service_name] = time.time()
                self._log_event(service_name, purpose, transport, monotonic_now, status='oauth_pending_in_progress')
            raise OAuthPending(f'OAuth in progress for {service_name}')

        # Pre-flight proactive refresh before expiry (avoid deleting tokens; refresh instead)
        storage_prefetch = self.token_storages.get(service_name)
        if storage_prefetch:
            try:
                _tokens = await storage_prefetch.get_tokens()
                if _tokens and _tokens.expires_in:
                    fp = storage_prefetch.file_path
                    if fp.exists():
                        import json as _json, time as _time
                        with open(fp,'r') as f: _data=_json.load(f)
                        rec = _data.get(service_name)
                        if rec and 'obtained_at' in rec:
                            age = _time.time() - rec['obtained_at']
                            remaining = _tokens.expires_in - age
                            if remaining < 120:  # refresh when <2m remaining
                                await self._maybe_refresh_tokens(service_name)
                                self._log_event(service_name, purpose, transport, time.perf_counter(), status='token_prefetch_refresh', remaining=round(remaining,1))
            except Exception:
                logger.debug("Pre-flight refresh check failed", exc_info=True)

        provider = self.oauth_providers.get(service_name)
        attempts = 0
        cooldown = 60.0
        while attempts < 2:
            attempt_start = time.perf_counter()
            try:
                self._log_event(service_name, purpose, transport, attempt_start, status='attempt', attempt=attempts+1, oauth_provider=bool(provider), in_progress=self.oauth_in_progress.get(service_name, False))
                result = await op_coro_factory(provider)
                status_final = "ok" if attempts == 0 else "ok_after_retry"
                self._log_event(service_name, purpose, transport, attempt_start, status=status_final, attempt=attempts+1, oauth_provider=bool(provider))
                if self.oauth_in_progress.get(service_name):
                    self.oauth_in_progress[service_name] = False
                return result
            except Exception as e:
                status_code, www_auth = self._extract_http_error_info(e)
                msg = str(e)
                is_unauthorized = (status_code == 401) or ('401' in msg)
                if is_unauthorized:
                    now = time.time()
                    last = self.last_unauthorized.get(service_name, 0)
                    if attempts == 0 and (now - last) < cooldown:
                        self._log_event(service_name, purpose, transport, attempt_start, status="unauthorized_suppressed", since=f"{now-last:.1f}s", www_auth=www_auth, attempt=attempts+1, oauth_provider=bool(provider))
                        raise
                    self.last_unauthorized[service_name] = now
                    self._log_event(service_name, purpose, transport, attempt_start, status="unauthorized", attempt=attempts+1, oauth_provider=bool(provider), www_auth=www_auth, code=status_code)
                    if provider is None:
                        mark_in_progress = True
                        try:
                            storage_tokens = self.token_storages.get(service_name)
                            if storage_tokens:
                                tk = await storage_tokens.get_tokens()
                                if tk:
                                    mark_in_progress = False
                        except Exception:
                            pass
                        if mark_in_progress:
                            self.oauth_in_progress[service_name] = True
                            self.oauth_attempt_timestamps[service_name] = time.time()
                        base = self._base_server_url(url)
                        self._log_event(service_name, purpose, transport, time.perf_counter(), status='oauth_begin_flow', base=base)
                        fut = getattr(self, 'oauth_futures', {}).get(service_name)
                        if fut and not fut.done():
                            self._log_event(service_name, purpose, transport, time.perf_counter(), status='oauth_future_join')
                            await fut
                            provider = self.oauth_providers.get(service_name)
                        else:
                            if not hasattr(self, 'oauth_futures'):
                                self.oauth_futures = {}
                            loop = asyncio.get_running_loop()
                            fut = loop.create_future()
                            self.oauth_futures[service_name] = fut
                            try:
                                provider = self._create_dynamic_oauth_provider(service_name, self._base_server_url(url))
                                if provider is None:
                                    self._log_event(service_name, purpose, transport, time.perf_counter(), status="oauth_provider_create_failed")
                                    fut.set_result(False)
                                    raise
                                else:
                                    self._log_event(service_name, purpose, transport, time.perf_counter(), status="oauth_provider_created")
                                    fut.set_result(True)
                                    asyncio.create_task(self._maybe_schedule_refresh(service_name))
                            finally:
                                pass
                    else:
                        self.oauth_in_progress[service_name] = True
                        self.oauth_attempt_timestamps[service_name] = time.time()
                        self._log_event(service_name, purpose, transport, time.perf_counter(), status='oauth_reauth_forced')
                        storage = self.token_storages.get(service_name)
                        if storage and hasattr(storage, 'delete_tokens'):
                            try:
                                await storage.delete_tokens()
                                self._log_event(service_name, purpose, transport, time.perf_counter(), status="tokens_deleted", attempt=attempts+1)
                            except Exception:
                                logger.debug("Token deletion failed during retry", exc_info=True)
                    attempts += 1
                    continue
                self._log_event(service_name, purpose, transport, attempt_start, status="error", attempt=attempts+1, error=type(e).__name__, msg=str(e)[:160])
                raise
        if self.oauth_in_progress.get(service_name):
            self.oauth_in_progress[service_name] = False
        raise RuntimeError(f"{service_name} {purpose} failed after OAuth retry")

    async def _maybe_schedule_refresh(self, service_name: str):
        """Schedule token refresh ahead of expiry; reschedules itself after run."""
        storage = self.token_storages.get(service_name)
        if not storage:
            return
        try:
            tokens = await storage.get_tokens()
            if not tokens or not tokens.expires_in:
                return
            fp = storage.file_path
            if not fp.exists():
                return
            import json as _json, time as _time
            with open(fp,'r') as f: data=_json.load(f)
            rec = data.get(service_name)
            if not rec or 'obtained_at' not in rec:
                return
            obtained = rec['obtained_at']
            age = _time.time() - obtained
            expires_in = tokens.expires_in
            remaining = expires_in - age
            if remaining <= 0:
                return
            # choose delay
            lead = 60
            delay = max(5, remaining - lead)
            if remaining < 120:
                delay = max(5, int(remaining * 0.5))
            # Cancel existing
            t_old = getattr(self, 'oauth_refresh_tasks', {}).pop(service_name, None) if hasattr(self, 'oauth_refresh_tasks') else None
            if t_old:
                t_old.cancel()
            if not hasattr(self, 'oauth_refresh_tasks'):
                self.oauth_refresh_tasks = {}
            async def _task():
                await asyncio.sleep(delay)
                try:
                    self._log_event(service_name, 'oauth_refresh', None, time.perf_counter(), status='attempt', in_seconds=delay)
                    # Attempt proactive refresh if within window and refresh token present
                    await self._maybe_refresh_tokens(service_name)
                    await self.test_service_connection(service_name)
                    self._log_event(service_name, 'oauth_refresh', None, time.perf_counter(), status='validated')
                except Exception as e:
                    self._log_event(service_name, 'oauth_refresh', None, time.perf_counter(), status='error', error=type(e).__name__)
                finally:
                    asyncio.create_task(self._maybe_schedule_refresh(service_name))
            t = asyncio.create_task(_task())
            self.oauth_refresh_tasks[service_name] = t
            self._log_event(service_name, 'oauth_refresh_schedule', None, time.perf_counter(), status='scheduled', delay=delay)
        except Exception:
            pass

    async def _perform_manual_token_exchange(self, service_name: str, auth_code: str, client_info, state: str = None) -> bool:
        """Manually perform OAuth token exchange when the MCP SDK fails to do it automatically"""
        try:
            import httpx
            # Get the base server URL from the service config
            service_config = self.config.get(service_name)
            if not service_config:
                logger.error(f"No service config found for {service_name}")
                return False
                
            server_url = service_config.spec.get('url', '').replace('/sse', '')  # Remove /sse suffix
            if not server_url:
                logger.error(f"No server URL found for {service_name}")
                return False
            
            # Discover token endpoint
            discovery_url = server_url.rstrip('/') + '/.well-known/oauth-authorization-server'
            token_endpoint = None
            
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    resp = await client.get(discovery_url)
                    if resp.status_code == 200:
                        discovery_data = resp.json()
                        token_endpoint = discovery_data.get('token_endpoint')
                        logger.info(f"Discovered token endpoint for {service_name}: {token_endpoint}")
                except Exception as e:
                    logger.warning(f"Failed to discover token endpoint for {service_name}: {e}")
            
            if not token_endpoint:
                # Fallback to standard endpoint
                token_endpoint = server_url.rstrip('/') + '/token'
                logger.info(f"Using fallback token endpoint for {service_name}: {token_endpoint}")
            
            # Prepare token exchange payload
            payload = {
                'grant_type': 'authorization_code',
                'code': auth_code,
                'redirect_uri': 'http://localhost:5859/oauth/callback',
                'client_id': client_info.client_id,
            }
            
            # Add client_secret if available (required for confidential clients)
            if hasattr(client_info, 'client_secret') and client_info.client_secret:
                payload['client_secret'] = client_info.client_secret
            
            logger.info(f"Performing manual token exchange for {service_name} with code={auth_code[:20]}...")
            
            # Perform token exchange
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(token_endpoint, data=payload)
                
                if resp.status_code != 200:
                    logger.error(f"Token exchange failed for {service_name}: HTTP {resp.status_code}")
                    try:
                        error_data = resp.json()
                        logger.error(f"Token exchange error: {error_data}")
                    except:
                        logger.error(f"Token exchange error text: {resp.text}")
                    return False
                
                token_data = resp.json()
                access_token = token_data.get('access_token')
                
                if not access_token:
                    logger.error(f"No access_token in response for {service_name}")
                    return False
                
                # Save tokens using our storage system
                storage = self.token_storages.get(service_name)
                if storage:
                    oauth_token = OAuthToken(
                        access_token=access_token,
                        token_type=token_data.get('token_type', 'Bearer'),
                        expires_in=token_data.get('expires_in'),
                        refresh_token=token_data.get('refresh_token'),
                        scope=token_data.get('scope')
                    )
                    
                    await storage.set_tokens(oauth_token)
                    
                    # Preserve client_info for future refresh operations
                    await self._preserve_client_info(service_name, client_info)
                    
                    logger.info(f"Successfully saved tokens for {service_name} via manual exchange")
                    return True
                else:
                    logger.error(f"No token storage found for {service_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Manual token exchange failed for {service_name}: {e}", exc_info=True)
            return False
    
    async def _preserve_client_info_from_provider(self, service_name: str):
        """Preserve client_info from OAuth provider after tokens are saved"""
        try:
            provider = self.oauth_providers.get(service_name)
            if not provider:
                return
                
            # Get client_info from provider's context
            if hasattr(provider, 'context') and provider.context:
                client_info = getattr(provider.context, 'client_info', None)
                if client_info:
                    await self._preserve_client_info(service_name, client_info)
                    logger.info(f"Preserved client_info for {service_name} from OAuth provider")
        except Exception as e:
            logger.warning(f"Failed to preserve client_info from provider for {service_name}: {e}")
    
    async def _preserve_client_info(self, service_name: str, client_info):
        """Preserve client_info in token storage for future refresh operations"""
        try:
            storage = self.token_storages.get(service_name)
            if not storage or not client_info:
                return
                
            fp = storage.file_path
            if not fp.exists():
                return
                
            # Use a lock to prevent concurrent file access
            if not hasattr(self, '_file_locks'):
                self._file_locks = {}
            
            if str(fp) not in self._file_locks:
                import asyncio
                self._file_locks[str(fp)] = asyncio.Lock()
            
            async with self._file_locks[str(fp)]:
                import json as _json
                
                # Read current tokens
                with open(fp, 'r') as f:
                    data = _json.load(f)
                
                # Skip if client_info already exists
                if service_name in data and 'client_info' in data[service_name]:
                    logger.debug(f"Client info already exists for {service_name}, skipping")
                    return
                
                # Add client_info to service record
                if service_name in data:
                    # Convert client_info to dict if it's a Pydantic model
                    if hasattr(client_info, 'model_dump'):
                        # Use mode='json' to handle AnyUrl and other special types
                        client_info_dict = client_info.model_dump(mode='json')
                    elif hasattr(client_info, '__dict__'):
                        client_info_dict = client_info.__dict__
                    else:
                        client_info_dict = client_info
                        
                    data[service_name]['client_info'] = client_info_dict
                    
                    # Write back to file atomically
                    temp_fp = fp.with_suffix('.tmp')
                    with open(temp_fp, 'w') as f:
                        _json.dump(data, f, indent=2)
                    temp_fp.replace(fp)  # Atomic move
                        
                    logger.info(f"Client info preserved for {service_name}")
        except Exception as e:
            logger.warning(f"Failed to preserve client_info for {service_name}: {e}")

    async def _maybe_refresh_tokens(self, service_name: str, force_refresh: bool = False):
        """If tokens near expiry and refresh_token present, perform refresh grant.
        
        Args:
            service_name: Name of the service to refresh tokens for
            force_refresh: If True, refresh even if not near expiry (for 401 errors)
        """
        storage = self.token_storages.get(service_name)
        provider = self.oauth_providers.get(service_name)
        if not storage or not provider:
            return
        try:
            tokens = await storage.get_tokens()
            if not tokens or not tokens.refresh_token:
                logger.info(f"No refresh token available for {service_name}, skipping refresh")
                return
            if not tokens.expires_in:
                logger.info(f"No expires_in available for {service_name}, skipping refresh")
                return
            fp = storage.file_path
            if not fp.exists():
                logger.info(f"Token file doesn't exist for {service_name}, skipping refresh")
                return
            import json as _json, time as _time, httpx
            with open(fp,'r') as f: data=_json.load(f)
            rec = data.get(service_name)
            if not rec or 'obtained_at' not in rec:
                logger.info(f"No token record with obtained_at for {service_name}, skipping refresh")
                return
            age = _time.time() - rec['obtained_at']
            remaining = tokens.expires_in - age
            if not force_refresh and remaining > 120:  # only refresh when close (<2 min) unless forced
                logger.info(f"Token for {service_name} not near expiry ({remaining:.1f}s remaining), skipping refresh")
                return
            
            logger.info(f"Starting token refresh for {service_name} (force={force_refresh}, remaining={remaining:.1f}s)")
            
            # Get server URL from provider context
            server_url = None
            if hasattr(provider, 'context') and provider.context:
                server_url = provider.context.server_url
            
            if not server_url:
                logger.warning(f"No server URL found for {service_name} refresh")
                return
                
            # Discover token endpoint via OAuth discovery
            discovery_url = server_url.rstrip('/') + '/.well-known/oauth-authorization-server'
            token_endpoint = None
            
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    resp = await client.get(discovery_url)
                    if resp.status_code == 200:
                        discovery_data = resp.json()
                        token_endpoint = discovery_data.get('token_endpoint')
                except:
                    pass
            
            if not token_endpoint:
                # Fallback to standard endpoint
                token_endpoint = server_url.rstrip('/') + '/token'
            
            # Get stored client credentials (from original OAuth flow)
            client_info = rec.get('client_info', {})
            client_id = client_info.get('client_id')
            client_secret = client_info.get('client_secret')
            
            if not client_id:
                logger.warning(f"No client_id found in stored tokens for {service_name}, cannot refresh")
                logger.debug(f"Available client_info keys for {service_name}: {list(client_info.keys())}")
                return
            
            payload = {
                'grant_type': 'refresh_token',
                'refresh_token': tokens.refresh_token,
                'client_id': client_id,
            }
            
            # Add client_secret if available (required for confidential clients)
            if client_secret:
                payload['client_secret'] = client_secret
            
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(token_endpoint, data=payload)
                if resp.status_code != 200:
                    self._log_event(service_name, 'token_refresh', None, time.perf_counter(), status='http_error', code=resp.status_code)
                    logger.warning(f"Token refresh failed for {service_name}: {resp.status_code}")
                    try:
                        error_data = resp.json()
                        error_type = error_data.get('error', '')
                        error_desc = error_data.get('error_description', '')
                        logger.warning(f"Refresh error details: {error_data}")
                        
                        # Handle client ID mismatch by clearing incompatible tokens
                        if error_type == 'invalid_grant' and 'mismatch' in error_desc.lower():
                            logger.warning(f"Client ID mismatch detected for {service_name}, clearing incompatible tokens")
                            # Remove the incompatible token data
                            await storage.delete_tokens()
                            self._log_event(service_name, 'token_refresh', None, time.perf_counter(), status='cleared_mismatched_tokens')
                            
                    except Exception as parse_error:
                        logger.warning(f"Refresh error text: {resp.text}")
                    return
                jd = resp.json()
                new_access = jd.get('access_token')
                new_refresh = jd.get('refresh_token')  # Check for new refresh token
                new_expires = jd.get('expires_in')
                new_scope = jd.get('scope')
                
                # Log what we received from the server
                logger.info(f"Refresh response for {service_name}: access_token={'✓' if new_access else '✗'}, refresh_token={'✓' if new_refresh else '✗'}, expires_in={new_expires}")
                
                if new_access:
                    await storage.replace_access_token(new_access, new_expires, new_scope, new_refresh)
                    self._log_event(service_name, 'token_refresh', None, time.perf_counter(), status='refreshed', remaining=int(remaining))
                    logger.info(f"Successfully refreshed tokens for {service_name}")
        except Exception as e:
            self._log_event(service_name, 'token_refresh', None, time.perf_counter(), status='error', error=type(e).__name__)
            logger.warning(f"Token refresh exception for {service_name}: {e}")
    
    async def _get_oauth_token_info(self, service_name: str) -> Dict[str, Any]:
        """Get comprehensive OAuth token information for a service"""
        oauth_info = {
            "has_access_token": False,
            "has_refresh_token": False,
            "access_token_obtained_at": None,
            "access_token_expires_in": None,
            "access_token_remaining_seconds": None,
            "refresh_token_obtained_at": None,
            "last_refresh_at": None,
            "refresh_needed": False,
            "client_configured": False
        }
        
        try:
            storage = self.token_storages.get(service_name)
            if not storage:
                return oauth_info
            
            # Get token information
            tokens = await storage.get_tokens()
            if tokens:
                oauth_info["has_access_token"] = bool(tokens.access_token)
                oauth_info["has_refresh_token"] = bool(tokens.refresh_token)
                
                # Get token file data for timestamps
                if storage.file_path.exists():
                    import json as _json, time as _time
                    with open(storage.file_path, 'r') as f:
                        data = _json.load(f)
                    
                    rec = data.get(service_name, {})
                    if rec:
                        # Access token info
                        obtained_at = rec.get('obtained_at')
                        if obtained_at:
                            oauth_info["access_token_obtained_at"] = obtained_at
                            age = _time.time() - obtained_at
                            
                            if tokens.expires_in:
                                oauth_info["access_token_expires_in"] = tokens.expires_in
                                remaining = tokens.expires_in - age
                                oauth_info["access_token_remaining_seconds"] = round(remaining, 1)
                                oauth_info["refresh_needed"] = remaining < 300  # Refresh needed if < 5 minutes
                        
                        # Check if we have refresh token metadata (for tracking when it was obtained)
                        # Since refresh tokens may be rotated, the obtained_at actually tracks the last refresh
                        if tokens.refresh_token:
                            oauth_info["refresh_token_obtained_at"] = obtained_at  # Same as access token for now
                            oauth_info["last_refresh_at"] = obtained_at  # This is effectively the last refresh time
                        
                        # Client info
                        client_info = rec.get('client_info', {})
                        oauth_info["client_configured"] = bool(client_info.get('client_id'))
            
            # Check provider status
            provider = self.oauth_providers.get(service_name)
            if provider:
                oauth_info["provider_configured"] = True
            else:
                oauth_info["provider_configured"] = False
                
        except Exception as e:
            logger.debug(f"Error getting OAuth token info for {service_name}: {e}")
        
        return oauth_info

    def _extract_http_error_info(self, exc: Exception):
        """Attempt to pull status code & WWW-Authenticate header from common HTTP client exceptions."""
        status_code = None
        www_auth = None
        try:
            # httpx style: exc.response
            resp = getattr(exc, 'response', None)
            if resp is not None:
                status_code = getattr(resp, 'status_code', None)
                headers = getattr(resp, 'headers', {}) or {}
                if isinstance(headers, dict):
                    www_auth = headers.get('www-authenticate') or headers.get('WWW-Authenticate')
            # aiohttp style: exc.status, exc.headers
            if status_code is None:
                status_code = getattr(exc, 'status', None)
            if www_auth is None:
                hdrs = getattr(exc, 'headers', None)
                if hdrs:
                    try:
                        www_auth = hdrs.get('www-authenticate') or hdrs.get('WWW-Authenticate')
                    except Exception:
                        pass
        except Exception:
            pass
        # If not directly detectable AND this looks like an aggregated TaskGroup/ExceptionGroup, dive deeper
        try:
            if (status_code is None or (status_code != 401 and '401' not in str(exc))) and isinstance(exc, BaseExceptionGroup):
                for leaf in self._flatten_exceptions(exc):
                    sc, wa = self._extract_http_error_info_single(leaf)
                    if sc == 401 or '401' in str(leaf):
                        return sc or 401, wa
        except Exception:
            pass
        return status_code, www_auth

    def _extract_http_error_info_single(self, exc: Exception):
        """Single (non-group) extraction helper used during deep scan."""
        status_code = None
        www_auth = None
        try:
            resp = getattr(exc, 'response', None)
            if resp is not None:
                status_code = getattr(resp, 'status_code', None)
                headers = getattr(resp, 'headers', {}) or {}
                if isinstance(headers, dict):
                    www_auth = headers.get('www-authenticate') or headers.get('WWW-Authenticate')
            if status_code is None:
                status_code = getattr(exc, 'status', None)
            if www_auth is None:
                hdrs = getattr(exc, 'headers', None)
                if hdrs:
                    try:
                        www_auth = hdrs.get('www-authenticate') or hdrs.get('WWW-Authenticate')
                    except Exception:
                        pass
        except Exception:
            pass
        return status_code, www_auth

    def _flatten_exceptions(self, exc: Exception):
        """Flatten nested BaseExceptionGroup / ExceptionGroup trees to leaf exceptions."""
        leaves = []
        stack = [exc]
        while stack:
            cur = stack.pop()
            if isinstance(cur, BaseExceptionGroup):
                try:
                    # Python 3.11 BaseExceptionGroup has .exceptions
                    for sub in getattr(cur, 'exceptions', []) or []:
                        stack.append(sub)
                except Exception:
                    pass
            else:
                leaves.append(cur)
        return leaves

    async def _get_bearer_auth_header(self, service_name: str) -> Optional[Dict[str, str]]:
        """Get bearer auth header if valid tokens exist, None otherwise."""
        storage = self.token_storages.get(service_name)
        if not storage:
            return None
            
        try:
            tokens = await storage.get_tokens()
            if tokens and tokens.access_token:
                return {'Authorization': f'Bearer {tokens.access_token}'}
        except Exception as e:
            logger.debug(f"Failed to get bearer token for {service_name}: {e}")
        
        return None

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
        else:  # http or streamable-http
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
                # Convert ToolAnnotations to dict for JSON serialization
                try:
                    if hasattr(tool.annotations, 'model_dump'):
                        tool_dict["annotations"] = tool.annotations.model_dump()
                    elif hasattr(tool.annotations, 'dict'):
                        tool_dict["annotations"] = tool.annotations.dict()
                    else:
                        # Manual serialization for complex objects
                        annotations_dict = {}
                        for attr in dir(tool.annotations):
                            if not attr.startswith('_') and hasattr(tool.annotations, attr):
                                value = getattr(tool.annotations, attr)
                                if not callable(value):
                                    annotations_dict[attr] = value
                        tool_dict["annotations"] = annotations_dict
                except Exception as e:
                    logger.warning(f"Failed to serialize annotations for tool {tool.name}: {e}")
                    # Skip annotations if serialization fails
                    pass
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

    def _has_embedded_auth(self, url: str) -> bool:
        """Detect if URL already contains embedded authentication material.

        This prevents us from attempting to create an OAuth provider for endpoints that
        already embed credentials (user:pass@host) or API key query parameters. We keep
        this intentionally lightweight; absence never blocks dynamic OAuth creation.
        """
        if not url:
            return False
        from urllib.parse import urlparse, parse_qs
        try:
            p = urlparse(url)
            if '@' in p.netloc:
                return True  # user:pass@host style
            
            # Check for path-based token embedding (like Zapier /s/token/mcp)
            path_parts = p.path.strip('/').split('/')
            # Look for 's' followed by a token in the path: /api/mcp/s/[token]/mcp
            for i in range(len(path_parts) - 1):
                if path_parts[i] == 's' and len(path_parts[i + 1]) > 10:  # Token likely to be >10 chars
                    return True
            
            qs = parse_qs(p.query)
            # Common api key param names (extendable)
            key_markers = {'api_key', 'apikey', 'token', 'auth', 'key'}
            return any(k.lower() in key_markers for k in qs.keys())
        except Exception:
            return False

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
                # Reuse existing provider
                if name in self.oauth_providers:
                    return self.oauth_providers[name]
                # Create service-specific token storage
                token_storage = FileTokenStorage("oauth_tokens.json", name)
                self.token_storages[name] = token_storage
                
                # Set up hook to preserve client_info when tokens are saved
                def preserve_client_on_save():
                    asyncio.create_task(self._preserve_client_info_from_provider(name))
                token_storage.on_set_tokens = preserve_client_on_save
                
                redirect_handler = lambda url: self._handle_oauth_redirect_service(name, url)
                provider = OAuthClientProvider(
                    server_url=server_url,
                    client_metadata=OAuthClientMetadata(
                        client_name=f"Fractalic MCP Client - {name}",
                        redirect_uris=[AnyUrl("http://localhost:5859/oauth/callback")],
                        grant_types=["authorization_code", "refresh_token"],
                        response_types=["code"],
                        scope=self._get_server_supported_scope(server_url),
                    ),
                    storage=token_storage,
                    redirect_handler=redirect_handler,
                    # No custom callback_handler - let MCP SDK handle it natively
                )
                self.oauth_providers[name] = provider
                logger.info(f"OAuth provider initialized for {name}")
        except Exception as e:
            logger.error(f"Failed to initialize OAuth for {name}: {e}")
    
    def _create_dynamic_oauth_provider(self, name: str, server_url: str) -> OAuthClientProvider:
        """Create OAuth provider dynamically for automatic OAuth discovery"""
        try:
            # Skip OAuth provider creation for services with embedded auth
            if self._has_embedded_auth(server_url):
                logger.info(f"Skipping OAuth provider creation for {name} - has embedded auth")
                return None
            
            # Reuse existing provider if present
            if name in self.oauth_providers:
                logger.info(f"Reusing existing OAuth provider for {name}")
                return self.oauth_providers[name]
            
            logger.info(f"Creating new OAuth provider for {name} with server_url={server_url}")
            
            # Create service-specific token storage
            token_storage = FileTokenStorage("oauth_tokens.json", name)
            self.token_storages[name] = token_storage
            
            # Set up hook to preserve client_info when tokens are saved
            def preserve_client_on_save():
                asyncio.create_task(self._preserve_client_info_from_provider(name))
            token_storage.on_set_tokens = preserve_client_on_save
            
            redirect_handler = lambda url: self._handle_oauth_redirect_service(name, url)
            
            # Use the proper callback handler that waits for the OAuth callback
            callback_handler = self._create_callback_handler_for_service(name)
            
            # Enable debug logging for OAuth provider
            import logging
            oauth_logger = logging.getLogger('mcp.client.auth')
            oauth_logger.setLevel(logging.DEBUG)
            
            # Log the redirect URI being used
            redirect_uri = "http://localhost:5859/oauth/callback"
            logger.info(f"OAuth provider for {name} using redirect_uri: {redirect_uri}")
            
            provider = OAuthClientProvider(
                server_url=server_url,
                client_metadata=OAuthClientMetadata(
                    client_name=f"Fractalic MCP Client - {name}",
                    redirect_uris=[AnyUrl(redirect_uri)],
                    grant_types=["authorization_code", "refresh_token"],
                    response_types=["code"],
                    scope=self._get_server_supported_scope(server_url),
                ),
                storage=token_storage,
                redirect_handler=redirect_handler,
                callback_handler=callback_handler,
            )
            # Store it for future use
            self.oauth_providers[name] = provider
            logger.info(f"Dynamic OAuth provider created successfully for {name} base_url={server_url}")
            return provider
        except Exception as e:
            logger.error(f"Failed to create dynamic OAuth provider for {name}: {e}", exc_info=True)
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

    async def _handle_oauth_redirect_service(self, service: str, auth_url: str) -> None:
        """Service-aware redirect handler with cooldown to prevent multiple windows."""
        COOLDOWN = 90  # seconds
        now = time.time()
        last = self.oauth_redirect_times.get(service, 0)
        if now - last < COOLDOWN:
            logger.info(f"OAuth redirect skipped for {service} - cooldown active (remaining: {COOLDOWN - (now - last):.1f}s)")
            self._log_event(service, 'oauth_redirect', None, now, status='skipped_cooldown', remaining=round(COOLDOWN - (now - last),1))
            return
        
        logger.info(f"Opening OAuth authorization URL for {service}: {auth_url}")
        self.oauth_redirect_times[service] = now
        self._persist_redirect_state()
        await self._handle_oauth_redirect(auth_url)
        self._log_event(service, 'oauth_redirect', None, now, status='opened')
    
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
                # Poll with periodic trace every 5s
                for i in range(timeout * 10):  # Check every 100ms
                    if oauth_callback_data[callback_key].get("received"):
                        break
                    if i % 50 == 0 and i > 0:  # every 5 seconds
                        logger.info(f"OAuth callback still pending for {service_name} elapsed={i/10:.1f}s in_progress={self.oauth_in_progress.get(service_name)}")
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
                
                logger.info(f"OAuth authorization code received successfully for {service_name} state_match={bool(callback_data.get('state'))}")
                
                return (callback_data["code"], callback_data.get("state"))
                
            finally:
                # Clean up the callback data
                oauth_callback_data.pop(callback_key, None)
        
        return callback_handler
    
    def _create_enhanced_callback_handler_for_service(self, service_name: str):
        """Create an enhanced callback handler that ensures token exchange completion"""
        async def enhanced_callback_handler() -> tuple[str, str | None]:
            """Enhanced OAuth callback handler with forced token exchange"""
            # Create a unique callback key for this service's OAuth session
            callback_key = f"{service_name}_{id(asyncio.current_task())}"
            
            # Register this callback session
            oauth_callback_data[callback_key] = {"received": False}
            
            # Wait for the callback with timeout
            timeout = 300  # 5 minutes
            logger.info(f"Enhanced OAuth callback waiting for service: {service_name}")
            
            try:
                # Poll with periodic trace every 5s
                for i in range(timeout * 10):  # Check every 100ms
                    if oauth_callback_data[callback_key].get("received"):
                        break
                    if i % 50 == 0 and i > 0:  # every 5 seconds
                        logger.info(f"OAuth callback still pending for {service_name} elapsed={i/10:.1f}s in_progress={self.oauth_in_progress.get(service_name)}")
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
                
                # Enhanced: Verify token exchange actually happened
                code = callback_data["code"]
                state = callback_data.get("state")
                
                logger.info(f"Enhanced OAuth: authorization code received for {service_name}, forcing token validation")
                
                # Give OAuth provider a moment to complete internal token exchange
                await asyncio.sleep(1.0)
                
                # Verify tokens were actually saved
                token_storage = self.token_storages.get(service_name)
                if token_storage:
                    tokens = await token_storage.get_tokens()
                    if tokens and tokens.access_token:
                        logger.info(f"Enhanced OAuth: tokens confirmed saved for {service_name}")
                    else:
                        logger.warning(f"Enhanced OAuth: authorization code received but no tokens saved for {service_name} - OAuth provider token exchange may have failed")
                        # Still return the code as the OAuth provider should handle this internally
                else:
                    logger.warning(f"Enhanced OAuth: no token storage found for {service_name}")
                
                return (code, state)
                
            finally:
                # Clean up the callback data
                oauth_callback_data.pop(callback_key, None)
        
        return enhanced_callback_handler

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
    
    async def _get_tools_for_service_impl(self, service_name: str, service: ServiceConfig, stage: Optional[Dict[str,bool]] = None) -> List[Dict[str, Any]]:
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
                
                # Try bearer auth first if tokens are available (more reliable than OAuth provider)
                bearer_auth = await self._get_bearer_auth_header(service_name)
                if bearer_auth:
                    # Pre-flight token refresh check (same as MCP manager startup)
                    try:
                        storage_prefetch = self.token_storages.get(service_name)
                        if storage_prefetch:
                            _tokens = await storage_prefetch.get_tokens()
                            if _tokens and _tokens.expires_in:
                                fp = storage_prefetch.file_path
                                if fp.exists():
                                    import json as _json, time as _time
                                    with open(fp,'r') as f: _data=_json.load(f)
                                    rec = _data.get(service_name)
                                    if rec and 'obtained_at' in rec:
                                        age = _time.time() - rec['obtained_at']
                                        remaining = _tokens.expires_in - age
                                        if remaining < 120:  # refresh when <2m remaining
                                            logger.info(f"Pre-flight token refresh for {service_name} (remaining={remaining:.1f}s)")
                                            await self._maybe_refresh_tokens(service_name)
                                            self._log_event(service_name, 'tools_sse_token_preflight_refresh', 'sse', time.perf_counter(), status='token_prefetch_refresh', remaining=round(remaining,1))
                                            # Get updated bearer auth after refresh
                                            bearer_auth = await self._get_bearer_auth_header(service_name)
                    except Exception:
                        logger.debug("Pre-flight token refresh check failed", exc_info=True)
                    
                    headers.update(bearer_auth)
                    logger.info(f"Using bearer token auth for {service_name} SSE connection")
                    
                    async def op_bearer():
                        conn_timeout, _ = self._compute_timeouts(service_name, 'sse', 'tools', True)  # has auth
                        read_timeout = 30.0
                        target_url = url
                        # Create local headers with bearer auth
                        bearer_headers = self._protocol_headers()
                        bearer_auth_local = await self._get_bearer_auth_header(service_name)
                        if bearer_auth_local:
                            bearer_headers.update(bearer_auth_local)
                        open_ts = time.perf_counter()
                        self._log_event(service_name, 'tools_sse_open', 'sse', open_ts, status='begin', conn_timeout=conn_timeout, read_timeout=read_timeout, oauth=False, bearer=True)
                        try:
                            async with sse_client(target_url, auth=None, headers=bearer_headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w):
                                entered_ts = time.perf_counter()
                                self._log_event(service_name, 'tools_sse_open', 'sse', entered_ts, status='opened')
                                if stage is not None:
                                    stage['entered'] = True
                                    self._log_event(service_name, 'tools_stage', 'sse', time.perf_counter(), status='entered')
                                async with ClientSession(r,w) as session:
                                    if stage is not None:
                                        stage['initialized'] = False
                                    init_ts = time.perf_counter()
                                    self._log_event(service_name, 'tools_init', 'sse', init_ts, status='begin')
                                    await session.initialize()
                                    if stage is not None:
                                        stage['initialized'] = True
                                        self._log_event(service_name, 'tools_stage', 'sse', time.perf_counter(), status='initialized')
                                    self._log_event(service_name, 'tools_init', 'sse', init_ts, status='ok')
                                    lt_ts = time.perf_counter()
                                    self._log_event(service_name, 'tools_list', 'sse', lt_ts, status='begin')
                                    tr = await session.list_tools()
                                    self._log_event(service_name, 'tools_list', 'sse', lt_ts, status='ok', count=len(getattr(tr,'tools',[]) or []))
                                    return self._format_tools(tr)
                        except Exception as e:
                            # Check if this is a 401 error indicating expired token
                            if "401" in str(e) or "Unauthorized" in str(e):
                                logger.info(f"Bearer token appears expired for {service_name}, attempting refresh...")
                                # Try to refresh tokens (force refresh on 401 error)
                                await self._maybe_refresh_tokens(service_name, force_refresh=True)
                                
                                # Get updated bearer auth header
                                updated_bearer_auth = await self._get_bearer_auth_header(service_name)
                                if updated_bearer_auth:
                                    # Reset headers with new token
                                    headers = self._protocol_headers()
                                    headers.update(updated_bearer_auth)
                                    logger.info(f"Retrying with refreshed bearer token for {service_name}")
                                    
                                    # Retry the connection with refreshed token
                                    async with sse_client(target_url, auth=None, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w):
                                        entered_ts = time.perf_counter()
                                        self._log_event(service_name, 'tools_sse_open', 'sse', entered_ts, status='retry_refreshed')
                                        if stage is not None:
                                            stage['entered'] = True
                                        async with ClientSession(r,w) as session:
                                            if stage is not None:
                                                stage['initialized'] = False
                                            await session.initialize()
                                            if stage is not None:
                                                stage['initialized'] = True
                                            lt_ts = time.perf_counter()
                                            tr = await session.list_tools()
                                            self._log_event(service_name, 'tools_list', 'sse', lt_ts, status='ok_after_refresh', count=len(getattr(tr,'tools',[]) or []))
                                            return self._format_tools(tr)
                                else:
                                    logger.error(f"Token refresh failed for {service_name}, no updated bearer auth available")
                                    raise
                            else:
                                # Non-auth related error, re-raise
                                logger.error(f"Bearer auth SSE connection failed for {service_name}: {e}")
                                raise
                    
                    try:
                        return await op_bearer()
                    except Exception as e:
                        # Check if we have refresh tokens - if so, don't fall back to OAuth provider
                        storage = self.token_storages.get(service_name)
                        has_refresh_token = False
                        if storage:
                            try:
                                tokens = await storage.get_tokens()
                                has_refresh_token = tokens and tokens.refresh_token
                            except Exception:
                                pass
                        
                        if has_refresh_token:
                            logger.error(f"Bearer auth failed for {service_name} despite having refresh token, not falling back to OAuth provider: {e}")
                            raise e  # Don't fall back if we have refresh tokens - something else is wrong
                        else:
                            logger.warning(f"Bearer auth failed for {service_name} (no refresh token), falling back to OAuth provider: {e}")
                
                # Fallback to OAuth provider only if bearer auth not available or no refresh tokens
                async def op(oauth_provider):
                    # compute timeouts each attempt (provider may appear after retry)
                    conn_timeout, _ = self._compute_timeouts(service_name, 'sse', 'tools', oauth_provider is not None)
                    read_timeout = 30.0  # explicit SSE read timeout separate from connection
                    target_url = url
                    open_ts = time.perf_counter()
                    # Reset headers for OAuth within the function scope
                    oauth_headers = self._protocol_headers()
                    self._log_event(service_name, 'tools_sse_open', 'sse', open_ts, status='begin', conn_timeout=conn_timeout, read_timeout=read_timeout, oauth=bool(oauth_provider))
                    try:
                        async with sse_client(target_url, auth=oauth_provider, headers=oauth_headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w):
                            entered_ts = time.perf_counter()
                            self._log_event(service_name, 'tools_sse_open', 'sse', entered_ts, status='opened')
                            if stage is not None:
                                stage['entered'] = True
                                self._log_event(service_name, 'tools_stage', 'sse', time.perf_counter(), status='entered')
                            async with ClientSession(r,w) as session:
                                if stage is not None:
                                    stage['initialized'] = False
                                init_ts = time.perf_counter()
                                self._log_event(service_name, 'tools_init', 'sse', init_ts, status='begin')
                                await session.initialize()
                                if stage is not None:
                                    stage['initialized'] = True
                                    self._log_event(service_name, 'tools_stage', 'sse', time.perf_counter(), status='initialized')
                                self._log_event(service_name, 'tools_init', 'sse', init_ts, status='ok')
                                lt_ts = time.perf_counter()
                                self._log_event(service_name, 'tools_list', 'sse', lt_ts, status='begin')
                                tr = await session.list_tools()
                                self._log_event(service_name, 'tools_list', 'sse', lt_ts, status='ok', count=len(getattr(tr,'tools',[]) or []))
                                return self._format_tools(tr)
                    except Exception as e:
                        # Generic deep 401 detection across all services (aggregated TaskGroup / ExceptionGroup)
                        unauthorized_detected = False
                        had_www_auth = None
                        try:
                            for leaf in self._flatten_exceptions(e):
                                sc, wa = self._extract_http_error_info_single(leaf)
                                if sc == 401 or '401' in str(leaf):
                                    unauthorized_detected = True
                                    had_www_auth = wa is not None
                                    break
                        except Exception:
                            pass

                        if (("401" in str(e)) or unauthorized_detected) and oauth_provider is None:
                            base = self._base_server_url(target_url)
                            if not self._has_embedded_auth(target_url):
                                self._create_dynamic_oauth_provider(service_name, base)
                            self._log_event(service_name, 'tools', 'sse', time.perf_counter(), status='forced_oauth_provider', base=base, agg=bool(unauthorized_detected), had_www_auth=had_www_auth)
                            raise e  # Let retry wrapper perform the retry with new provider

                        # TaskGroup bug mitigation (generic)
                        # Only disable if it's NOT an auth-related TaskGroup error
                        if 'TaskGroup' in str(e) and '401' not in str(e) and not unauthorized_detected:
                            prev_state = self.service_states.get(service_name, 'enabled')
                            self.service_states[service_name] = 'disabled'
                            self._log_event(service_name, 'tools', 'sse', time.perf_counter(), status='disabled_taskgroup_bug', previous_state=prev_state)
                            raise
                        elif 'TaskGroup' in str(e):
                            # It's a 401-related TaskGroup error, let OAuth retry handle it
                            self._log_event(service_name, 'tools', 'sse', time.perf_counter(), status='taskgroup_401_retry', unauthorized=unauthorized_detected)
                            raise
                        if '404' in str(e) and target_url.endswith('/sse'):
                            alt = target_url.rstrip('/sse') + '/mcp'
                            logger.warning(f"{service_name}: /sse 404 -> retry alt endpoint {alt}")
                            async with sse_client(alt, auth=oauth_provider, headers=headers, timeout=conn_timeout) as (r2,w2):
                                async with ClientSession(r2,w2) as session2:
                                    await session2.initialize()
                                    tr2 = await session2.list_tools()
                                    return self._format_tools(tr2)
                        raise
                # Run initial attempt
                stage_local = {'entered': False, 'initialized': False}
                # Attach stage_local so inner op can mark it; wrap op to pass through
                async def op_with_stage(oauth_provider):
                    nonlocal stage_local
                    return await op(oauth_provider)
                tools = await self._run_with_oauth_retry(service_name, 'tools', 'sse', url, op_with_stage)
                if tools:
                    return tools
                # Stall detection: no tools AND never entered session
                if not stage_local['entered']:
                    self._log_event(service_name, 'tools', 'sse', time.perf_counter(), status='stall_detected', detail='no_entered')
                    # Recreate provider once (maybe stale underlying session / registration)
                    try:
                        prov = self.oauth_providers.get(service_name)
                        if prov:
                            base = self._base_server_url(url)
                            # Drop existing provider
                            self.oauth_providers.pop(service_name, None)
                            self._log_event(service_name, 'tools', 'sse', time.perf_counter(), status='provider_dropped_for_stall')
                            if not self._has_embedded_auth(target_url):
                                self._create_dynamic_oauth_provider(service_name, base)
                            self._log_event(service_name, 'tools', 'sse', time.perf_counter(), status='provider_recreated')
                    except Exception:
                        logger.debug("Provider recreation failed during stall recovery", exc_info=True)
                    # Second attempt with extended timeouts
                    async def op_retry(oauth_provider):
                        conn_timeout_retry = 15.0
                        read_timeout_retry = 45.0
                        self._log_event(service_name, 'tools_sse_open', 'sse', time.perf_counter(), status='begin_retry', conn_timeout=conn_timeout_retry, read_timeout=read_timeout_retry, oauth=bool(oauth_provider))
                        async with sse_client(url, auth=oauth_provider, headers=headers, timeout=conn_timeout_retry, sse_read_timeout=read_timeout_retry) as (r,w):
                            self._log_event(service_name, 'tools_sse_open', 'sse', time.perf_counter(), status='opened_retry')
                            async with ClientSession(r,w) as session:
                                self._log_event(service_name, 'tools_init', 'sse', time.perf_counter(), status='begin_retry')
                                await session.initialize()
                                self._log_event(service_name, 'tools_init', 'sse', time.perf_counter(), status='ok_retry')
                                self._log_event(service_name, 'tools_list', 'sse', time.perf_counter(), status='begin_retry')
                                tr = await session.list_tools()
                                self._log_event(service_name, 'tools_list', 'sse', time.perf_counter(), status='ok_retry', count=len(getattr(tr,'tools',[]) or []))
                                return self._format_tools(tr)
                    try:
                        tools_retry = await self._run_with_oauth_retry(service_name, 'tools', 'sse', url, op_retry)
                        if tools_retry:
                            return tools_retry
                        self._log_event(service_name, 'tools', 'sse', time.perf_counter(), status='stall_unrecovered')
                    except Exception as e:
                        self._log_event(service_name, 'tools', 'sse', time.perf_counter(), status='stall_retry_error', error=type(e).__name__)
                return tools
                        
            elif service.transport in ("http", "streamable-http"):
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth_provider):
                    connection_timeout, read_timeout = self._compute_timeouts(service_name, service.transport, 'tools', oauth_provider is not None)
                    async with streamablehttp_client(
                        url,
                        auth=oauth_provider,
                        headers=headers,
                        timeout=connection_timeout,
                        sse_read_timeout=read_timeout
                    ) as (r,w,_sid):
                        logger.info(f"[{service_name}] Streamable HTTP client connected successfully")
                        async with ClientSession(r,w) as session:
                            logger.info(f"[{service_name}] Starting session initialization...")
                            await session.initialize()
                            logger.info(f"[{service_name}] Session initialized, fetching tools...")
                            tr = await session.list_tools()
                            logger.info(f"[{service_name}] Tools fetched: {len(tr.tools) if hasattr(tr, 'tools') else 'unknown'}")
                            return self._format_tools(tr)
                return await self._run_with_oauth_retry(service_name, 'tools', service.transport, url, op)
                        
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
        Get server capabilities using cache-first approach.
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

        # Return cached capabilities if available
        cached_capabilities = self.capabilities_cache.get(service_name)
        if cached_capabilities:
            logger.debug(f"Returning cached capabilities for {service_name}: {cached_capabilities}")
            self._log_event(service_name, "capabilities", self.config[service_name].transport, start_ts, status="ok", cache="hit", **cached_capabilities)
            return cached_capabilities

        # Cache miss - fetch and cache capabilities
        service = self.config[service_name]
        
        try:
            # Longer timeout for OAuth services during first capability probe
            needs_oauth = (service_name in self.oauth_providers or 
                           self._is_oauth_service(service.spec.get('url', '')))
            cap_timeout = 3.0
            if needs_oauth:
                cap_timeout = 15.0  # allow discovery / auth redirect headers
            
            capabilities = await asyncio.wait_for(
                self._get_service_capabilities_impl(service_name, service), 
                timeout=cap_timeout
            )
            
            # Cache the result
            self.capabilities_cache[service_name] = capabilities
            logger.debug(f"Cached capabilities for {service_name}: {capabilities}")
            self._log_event(service_name, "capabilities", service.transport, start_ts, status="ok", cache="miss", **capabilities)
            return capabilities
            
        except asyncio.TimeoutError:
            default_caps = {"prompts": False, "resources": False, "tools": False}
            if needs_oauth:
                logger.warning(f"Timeout ({cap_timeout}s) getting capabilities for OAuth service {service_name} - may proceed after auth.")
            else:
                logger.warning(f"Timeout ({cap_timeout}s) getting capabilities for {service_name} - fast fail, non-blocking.")
            self.capabilities_cache[service_name] = default_caps
            self._log_event(service_name, "capabilities", service.transport, start_ts, status="timeout", cache="miss", **default_caps)
            return default_caps
        except Exception as e:
            default_caps = {"prompts": False, "resources": False, "tools": False}
            logger.debug(f"Failed to get capabilities for {service_name}: {e}")
            self.capabilities_cache[service_name] = default_caps
            self._log_event(service_name, "capabilities", service.transport, start_ts, status="error", error=type(e).__name__, cache="miss", **default_caps)
            return default_caps

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
                        if not self._has_embedded_auth(url):
                            oauth_provider = self.oauth_providers.get(service_name) or self._create_dynamic_oauth_provider(service_name, url)
                        else:
                            oauth_provider = None
                
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
                        
            elif service.transport in ("http", "streamable-http"):
                # HTTP/Streamable HTTP transport
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
        """Get prompts for a service using cache-first approach with per-request session fallback"""
        start_ts = time.perf_counter()
        if service_name not in self.config:
            self._log_event(service_name, "list_prompts", None, start_ts, status="skip", reason="not_found")
            return []
        
        # Check if service is enabled before attempting connection
        if self.service_states.get(service_name, "enabled") == "disabled":
            logger.info(f"Service {service_name} is disabled, skipping prompt retrieval")
            return []
        
        # Return cached prompts for instant response (with lazy fallback fetch if empty)
        cached_prompts = self.prompts_cache.get(service_name, [])
        if cached_prompts:
            logger.debug(f"Returning {len(cached_prompts)} cached prompts for {service_name}")
            self._log_event(service_name, "list_prompts", self.config[service_name].transport, start_ts, prompt_count=len(cached_prompts), cache="hit")
            return cached_prompts
            
        # Check if service supports prompts capability first
        try:
            capabilities = await self.get_service_capabilities(service_name)
            if not capabilities.get("prompts", False):
                logger.debug(f"Service {service_name} does not support prompts capability")
                self.prompts_cache[service_name] = []  # Cache empty result
                return []
        except Exception as e:
            logger.debug(f"Could not check capabilities for {service_name}: {e}")
            return []
        
        service = self.config[service_name]
        
        # If cache empty attempt a one-shot direct fetch
        try:
            logger.info(f"Cache empty for {service_name}; attempting direct prompt fetch now")
            needs_oauth = (service_name in self.oauth_providers or 
                           self._is_oauth_service(service.spec.get('url', '')))
            prompts_timeout = 3.0 if not needs_oauth else 20.0
            direct = await asyncio.wait_for(
                self._get_prompts_for_service_impl(service_name, service), 
                timeout=prompts_timeout
            )
            if direct:
                self.prompts_cache[service_name] = direct
            self._log_event(service_name, "list_prompts", self.config[service_name].transport, start_ts, prompt_count=len(direct), cache="miss")
            return direct
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
                        if not self._has_embedded_auth(url):
                            oauth_provider = self.oauth_providers.get(service_name) or self._create_dynamic_oauth_provider(service_name, url)
                        else:
                            oauth_provider = None
                
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
                        
            elif service.transport in ("http", "streamable-http"):
                # HTTP/Streamable HTTP transport
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
        """Get resources for a service using cache-first approach with per-request session fallback"""
        start_ts = time.perf_counter()
        if service_name not in self.config:
            self._log_event(service_name, "list_resources", None, start_ts, status="skip", reason="not_found")
            return []
        
        # Check if service is enabled before attempting connection
        if self.service_states.get(service_name, "enabled") == "disabled":
            logger.info(f"Service {service_name} is disabled, skipping resource retrieval")
            return []
        
        # Return cached resources for instant response (with lazy fallback fetch if empty)
        cached_resources = self.resources_cache.get(service_name, [])
        if cached_resources:
            logger.debug(f"Returning {len(cached_resources)} cached resources for {service_name}")
            self._log_event(service_name, "list_resources", self.config[service_name].transport, start_ts, resource_count=len(cached_resources), cache="hit")
            return cached_resources
            
        # Check if service supports resources capability first
        try:
            capabilities = await self.get_service_capabilities(service_name)
            if not capabilities.get("resources", False):
                logger.debug(f"Service {service_name} does not support resources capability")
                self.resources_cache[service_name] = []  # Cache empty result
                return []
        except Exception as e:
            logger.debug(f"Could not check capabilities for {service_name}: {e}")
            return []
        
        service = self.config[service_name]
        
        # If cache empty attempt a one-shot direct fetch
        try:
            logger.info(f"Cache empty for {service_name}; attempting direct resource fetch now")
            needs_oauth = (service_name in self.oauth_providers or 
                           self._is_oauth_service(service.spec.get('url', '')))
            resources_timeout = 3.0 if not needs_oauth else 20.0
            direct = await asyncio.wait_for(
                self._get_resources_for_service_impl(service_name, service), 
                timeout=resources_timeout
            )
            if direct:
                self.resources_cache[service_name] = direct
            self._log_event(service_name, "list_resources", self.config[service_name].transport, start_ts, resource_count=len(direct), cache="miss")
            return direct
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
                        if not self._has_embedded_auth(url):
                            oauth_provider = self.oauth_providers.get(service_name) or self._create_dynamic_oauth_provider(service_name, url)
                        else:
                            oauth_provider = None
                
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
                        
            elif service.transport in ("http", "streamable-http"):
                # HTTP/Streamable HTTP transport
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
                    
                    # Use LiteLLM to count tokens - use gpt-4o to match actual usage
                    token_count = litellm.token_counter(model="gpt-4o", messages=[], tools=openai_tools)
                    
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
                        
            elif service.transport in ("http", "streamable-http"):
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth_provider):
                    conn_timeout, read_timeout = self._compute_timeouts(service_name, service.transport, 'call', oauth_provider is not None)
                    async with streamablehttp_client(url, auth=oauth_provider, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            result = await session.call_tool(tool_name, arguments)
                            return self._format_tool_call_result(result)
                return await self._run_with_oauth_retry(service_name, 'call_tool', service.transport, url, op)
                        
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
            elif service.transport in ("http", "streamable-http"):
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, read_timeout = self._compute_timeouts(service_name, service.transport, 'prompt', oauth is not None)
                    async with streamablehttp_client(url, auth=oauth, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            result = await session.get_prompt(prompt_name, arguments or {})
                            return self._format_prompt(result)
                return await self._run_with_oauth_retry(service_name, 'get_prompt', service.transport, url, op)
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
            if service.transport in ("http", "streamable-http"):
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, read_timeout = self._compute_timeouts(service_name, service.transport, 'resource', oauth is not None)
                    async with streamablehttp_client(url, auth=oauth, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            result = await session.read_resource(uri)
                            return self._format_resource(result, resource_uri)
                return await self._run_with_oauth_retry(service_name, 'read_resource', service.transport, url, op)
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
            # Treat OAuthPending as transient (keep enabled) so background OAuth can finish
            if isinstance(e, OAuthPending):
                logger.info(f"OAuth pending during test for {service_name}; leaving service enabled")
                self._log_event(service_name, "test_connection", service.transport, start_ts, status="pending_oauth")
                return False
            
            # Check if it's an auth error that might be resolved by token refresh
            error_str = str(e)
            is_auth_error = ('401' in error_str or 'Unauthorized' in error_str or 
                           'TaskGroup' in error_str and '401' in error_str)
            
            if is_auth_error:
                logger.warning(f"Auth error during test for {service_name}, keeping enabled for retry: {e}")
                self._log_event(service_name, "test_connection", service.transport, start_ts, status="auth_error", error=type(e).__name__)
                return False
            
            # Only disable for non-recoverable errors
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
                        self._update_service_cache(service_name, True)  # Refresh cache after enabling
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
                            self._update_service_cache(service_name, True)  # Refresh cache after enabling
                            return True
                return await self._run_with_oauth_retry(service_name, 'test_connection', 'sse', url, op)
                        
            elif service.transport in ("http", "streamable-http"):
                url = service.spec['url']
                headers = self._protocol_headers()
                async def op(oauth):
                    conn_timeout, read_timeout = self._compute_timeouts(service_name, service.transport, 'init', oauth is not None)
                    async with streamablehttp_client(url, auth=oauth, headers=headers, timeout=conn_timeout, sse_read_timeout=read_timeout) as (r,w,_sid):
                        async with ClientSession(r,w) as session:
                            await session.initialize()
                            await session.list_tools()
                            self.service_states[service_name] = "enabled"
                            self._update_service_cache(service_name, True)  # Refresh cache after enabling
                            return True
                return await self._run_with_oauth_retry(service_name, 'test_connection', service.transport, url, op)
                        
            else:
                raise ValueError(f"Unsupported transport: {service.transport}")
        
        except Exception as e:
            logger.error(f"Error in _test_service_connection_impl for {service_name}: {e}")
            if "timeout" in str(e).lower() or "ReadTimeout" in str(e):
                logger.warning(f"Timeout detected during connection test for {service_name}")
            raise  # Re-raise to be caught by the outer timeout handler
    
    async def status(self, include_tools_info: bool = False, include_complete_data: bool = False) -> Dict[str, Any]:
        """
        Get status of all services with optional complete data inclusion.
        
        Args:
            include_tools_info: Include tool counts and token counts
            include_complete_data: Include full tools, prompts, resources, and capabilities data
        
        Returns complete MCP server state to reduce UI-to-manager traffic.
        """
        services = {}
        total_tools = 0
        total_tokens = 0
        total_prompts = 0
        total_resources = 0
        
        for name, service in self.config.items():
            # Base configuration info - per-request pattern with enabled/disabled states
            service_status = self.service_states.get(name, "enabled")
            service_info = {
                "status": service_status,
                "enabled": service_status == "enabled",  # Flag for LLM response schema building
                "connected": False,  # Per-request pattern - no persistent connections
                "has_oauth": name in self.oauth_providers,  # Only report OAuth if we actually have a provider (RFC 9728 compliant)
                "transport": service.transport,
                "url": service.spec.get('url', None),
                "command": service.spec.get('command', None)
            }
            
            # Add comprehensive OAuth token information if OAuth is configured
            if name in self.oauth_providers:
                oauth_info = await self._get_oauth_token_info(name)
                service_info["oauth"] = oauth_info
            
            # Include tools information if requested (aligned with legacy manager)
            if include_tools_info or include_complete_data:
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
            
            # Include complete data if requested (full MCP feature set)
            if include_complete_data and service_status == "enabled":
                # Get tools data (use cache if available)
                try:
                    cached_tools = self.tools_cache.get(name, [])
                    if cached_tools:
                        tools = cached_tools
                        logger.debug(f"Using cached tools for {name}: {len(tools)} tools")
                    else:
                        tools = await self.get_tools_for_service(name)
                        if tools:
                            self.tools_cache[name] = tools
                    service_info["tools"] = tools
                except Exception as e:
                    logger.warning(f"Failed to get tools for {name}: {e}")
                    service_info["tools"] = []
                    service_info["tools_error"] = str(e)
                
                # Get prompts data (use cache if available)
                try:
                    cached_prompts = self.prompts_cache.get(name, [])
                    if cached_prompts:
                        prompts = cached_prompts
                        logger.debug(f"Using cached prompts for {name}: {len(prompts)} prompts")
                    else:
                        prompts = await self.get_prompts_for_service(name)
                        if prompts:
                            self.prompts_cache[name] = prompts
                    service_info["prompts"] = prompts
                    service_info["prompt_count"] = len(prompts)
                    total_prompts += len(prompts)
                except Exception as e:
                    logger.warning(f"Failed to get prompts for {name}: {e}")
                    service_info["prompts"] = []
                    service_info["prompt_count"] = 0
                    service_info["prompts_error"] = str(e)
                
                # Get resources data (use cache if available)
                try:
                    cached_resources = self.resources_cache.get(name, [])
                    if cached_resources:
                        resources = cached_resources
                        logger.debug(f"Using cached resources for {name}: {len(resources)} resources")
                    else:
                        resources = await self.get_resources_for_service(name)
                        if resources:
                            self.resources_cache[name] = resources
                    service_info["resources"] = resources
                    service_info["resource_count"] = len(resources)
                    total_resources += len(resources)
                except Exception as e:
                    logger.warning(f"Failed to get resources for {name}: {e}")
                    service_info["resources"] = []
                    service_info["resource_count"] = 0
                    service_info["resources_error"] = str(e)
                
                # Get capabilities data
                try:
                    capabilities = await self.get_service_capabilities(name)
                    service_info["capabilities"] = capabilities
                except Exception as e:
                    logger.warning(f"Failed to get capabilities for {name}: {e}")
                    service_info["capabilities"] = {}
                    service_info["capabilities_error"] = str(e)
            
            services[name] = service_info
        
        return {
            "services": services,
            "total_services": len(self.config),
            "enabled_services": len([s for s in services.values() if s.get("enabled", True)]),
            "oauth_enabled": len(self.oauth_providers) > 0,
            "total_tools": total_tools if (include_tools_info or include_complete_data) else 0,
            "total_tokens": total_tokens if (include_tools_info or include_complete_data) else 0,
            "total_prompts": total_prompts if include_complete_data else 0,
            "total_resources": total_resources if include_complete_data else 0,
            "mcp_version": MCP_PROTOCOL_VERSION,
            "complete_data_included": include_complete_data
        }

# OAuth callback handler for web server
async def oauth_callback_handler(request):
    """Handle OAuth callback from the authorization server"""
    global oauth_callback_data
    supervisor = request.app['supervisor']
    
    query = request.query

    try:
        pending_keys = [k for k,v in oauth_callback_data.items() if not v.get('received')]
        logger.info(f"/oauth/callback invoked params={list(query.keys())} pending_sessions={len(pending_keys)}")
    except Exception:
        pass
    
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
        # Release in-progress flags so subsequent requests don't get short-circuited as pending
        try:
            released = []
            for svc, in_prog in list(getattr(supervisor, 'oauth_in_progress', {}).items()):
                if in_prog:
                    supervisor.oauth_in_progress[svc] = False
                    supervisor.oauth_attempt_timestamps.pop(svc, None)
                    released.append(svc)
            if released:
                logger.info(f"OAuth in_progress flags released for services={released}")
                # Proactively trigger background connection tests to force token exchange flows
                for svc in released:
                    async def _bg(s=svc):
                        try:
                            # Wait a moment for the OAuth provider to complete token exchange
                            await asyncio.sleep(1.0)
                            
                            # Attempt manual token exchange before connection test if tokens still absent.
                            try:
                                prov = supervisor.oauth_providers.get(s)
                                storage = supervisor.token_storages.get(s)
                                tokens_present = False
                                if storage:
                                    existing = await storage.get_tokens()
                                    tokens_present = existing is not None
                                    if tokens_present:
                                        logger.info(f"Tokens successfully stored for {s} after OAuth callback")
                                if prov and not tokens_present:
                                    logger.info(f"Manual token exchange attempt for {s} (no tokens present post-callback)")
                                    
                                    # Try multiple approaches to force token exchange
                                    success = False
                                    
                                    # Method 1: Check if provider has exchange_code_for_tokens method
                                    exch = getattr(prov, 'exchange_code_for_tokens', None)
                                    if callable(exch) and not success:
                                        try:
                                            await exch(code)  # use broadcast code
                                            # Verify tokens were saved
                                            if storage:
                                                tokens_after = await storage.get_tokens()
                                                if tokens_after and tokens_after.access_token:
                                                    logger.info(f"Manual token exchange successful for {s}")
                                                    success = True
                                        except Exception as ex:
                                            logger.warning(f"exchange_code_for_tokens failed for {s}: {type(ex).__name__} {ex}")
                                    
                                    # Method 2: Try auth flow completion
                                    if not success and hasattr(prov, 'complete_auth_flow'):
                                        try:
                                            await prov.complete_auth_flow(code, state)
                                            # Verify tokens were saved
                                            if storage:
                                                tokens_after = await storage.get_tokens()
                                                if tokens_after and tokens_after.access_token:
                                                    logger.info(f"Complete auth flow successful for {s}")
                                                    success = True
                                        except Exception as ex:
                                            logger.warning(f"complete_auth_flow failed for {s}: {type(ex).__name__} {ex}")
                                    
                                    # Method 3: Force the OAuth provider to complete its auth flow
                                    if not success and hasattr(prov, '_complete_token_exchange'):
                                        try:
                                            await prov._complete_token_exchange(code, state)
                                            # Verify tokens were saved
                                            if storage:
                                                tokens_after = await storage.get_tokens()
                                                if tokens_after and tokens_after.access_token:
                                                    logger.info(f"Force token exchange successful for {s}")
                                                    success = True
                                        except Exception as ex:
                                            logger.warning(f"_complete_token_exchange failed for {s}: {type(ex).__name__} {ex}")
                                    
                                    # Method 4: Manual HTTP token exchange as fallback
                                    if not success:
                                        try:
                                            # Get the OAuth client info for token exchange
                                            client_info = await storage.get_client_info() if storage else None
                                            if client_info:
                                                logger.info(f"Attempting manual HTTP token exchange for {s} using client_id {client_info.client_id}")
                                                
                                                # Perform manual token exchange
                                                success = await supervisor._perform_manual_token_exchange(
                                                    s, code, client_info, state=state
                                                )
                                                
                                                if success:
                                                    logger.info(f"Manual HTTP token exchange successful for {s}")
                                                else:
                                                    logger.warning(f"Manual HTTP token exchange failed for {s}")
                                        except Exception as ex:
                                            logger.warning(f"Manual HTTP token exchange attempt failed for {s}: {type(ex).__name__} {ex}")
                                    
                                    if not success:
                                        logger.warning(f"All manual token exchange methods failed for {s} - OAuth provider should handle this automatically")
                                else:
                                    if tokens_present:
                                        logger.info(f"Tokens already present for {s}, skipping manual exchange")
                                    else:
                                        logger.info(f"No provider found for {s}, cannot attempt manual token exchange")
                            except Exception as inner:
                                logger.debug(f"Manual exchange block error for {s}: {inner}")
                            await supervisor.test_service_connection(s)
                        except Exception as _e:
                            logger.debug(f"Post-callback test_service_connection failed for {s}: {_e}")
                    asyncio.create_task(_bg())
        except Exception:
            logger.debug("Failed releasing oauth_in_progress flags", exc_info=True)
        try:
            logger.info(f"Broadcast completed sessions={len(oauth_callback_data)} keys={[k for k in oauth_callback_data.keys()]}")
        except Exception:
            pass
        
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
        error_description = query.get('error_description', '')
        
        # Broadcast error to all pending OAuth sessions
        for key in list(oauth_callback_data.keys()):
            if not oauth_callback_data[key].get("received"):
                oauth_callback_data[key].update({
                    'received': True,
                    'code': None,
                    'state': None,
                    'error': f"{error}: {error_description}" if error_description else error
                })
        
        # Clear in-progress flags on error
        try:
            for svc in list(getattr(supervisor, 'oauth_in_progress', {}).keys()):
                if supervisor.oauth_in_progress.get(svc):
                    supervisor.oauth_in_progress[svc] = False
                    supervisor.oauth_attempt_timestamps.pop(svc, None)
                    logger.info(f"OAuth in_progress flag cleared for {svc} due to error")
        except Exception:
            pass
        
        logger.error(f"OAuth authorization error: {error}" + (f": {error_description}" if error_description else ""))
        
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
    """GET /status - Get services status with optional complete data inclusion
    
    Query parameters:
    - include_tools_info: Include tool counts (default: true for legacy compatibility)  
    - include_complete_data: Include full tools, prompts, resources, and capabilities
    """
    try:
        # Check for include_tools_info query parameter - default to True for legacy compatibility
        include_tools_param = request.query.get('include_tools_info', 'true').lower()
        include_tools = include_tools_param in ('true', '1', 'yes')
        
        # Check for include_complete_data query parameter - default to False
        include_complete_param = request.query.get('include_complete_data', 'false').lower()
        include_complete = include_complete_param in ('true', '1', 'yes')
        
        supervisor = request.app['supervisor']
        status = await supervisor.status(include_tools_info=include_tools, include_complete_data=include_complete)
        return web.json_response(status)
    except Exception as e:
        logger.error(f"Status error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def complete_status_handler(request):
    """GET /status/complete - Get complete MCP server state (tools, prompts, resources, capabilities)
    
    This single endpoint provides all MCP data to minimize UI-to-manager traffic.
    Returns the complete state including:
    - All services with their enabled/disabled status
    - Full tools list for each service
    - Full prompts list for each service  
    - Full resources list for each service
    - Service capabilities
    - OAuth status
    - Comprehensive totals and metadata
    """
    try:
        supervisor = request.app['supervisor']
        status = await supervisor.status(include_tools_info=True, include_complete_data=True)
        return web.json_response(status)
    except Exception as e:
        logger.error(f"Complete status error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def toggle_service_handler(request):
    """POST /toggle/{name} - Toggle service enabled/disabled state"""
    service_name = request.match_info['name']
    supervisor = request.app['supervisor']
    
    if service_name not in supervisor.config:
        return web.json_response({"error": f"Service {service_name} not found"}, status=404)
    
    try:
        # Try to get parameters from JSON body first, then fall back to query params
        enabled = None
        force = False
        
        # Check for JSON body - only parse if content-type is JSON and body is not empty
        content_type = request.content_type
        if content_type and 'application/json' in content_type:
            try:
                body = await request.text()
                if body and body.strip():  # Only parse if body is not empty
                    import json
                    data = json.loads(body)
                    enabled = data.get('enabled', None)
                    force = data.get('force', False)
            except Exception as e:
                # Log but don't fail - will use query params
                logger.debug(f"Failed to parse JSON body: {e}")
        
        # Fall back to query parameters if not found in body
        if enabled is None and 'enabled' in request.query:
            enabled_str = request.query.get('enabled', '').lower()
            if enabled_str in ('true', '1', 'yes'):
                enabled = True
            elif enabled_str in ('false', '0', 'no'):
                enabled = False
        
        if 'force' in request.query:
            force = request.query.get('force', '').lower() in ('true', '1', 'yes')
        
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
        if new_state == "enabled" and not force:
            try:
                connection_test_success = await supervisor.test_service_connection(service_name)
                if not connection_test_success:
                    # If test failed due to OAuth pending, keep enabled; else disable
                    if supervisor.service_states.get(service_name) != 'disabled':
                        pass
                    else:
                        supervisor._update_service_cache(service_name, False)
                        return web.json_response({
                            "success": False,
                            "service": service_name,
                            "enabled": False,
                            "message": "Service connection test failed, keeping disabled"
                        })
            except Exception as e:
                if isinstance(e, OAuthPending):
                    # Keep enabled, allow OAuth flow to proceed
                    connection_test_success = False
                else:
                    supervisor.service_states[service_name] = "disabled"
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
            "previous_state": current_state if enabled is None else None,
            "force": force,
            "connection_test_success": connection_test_success
        })
    except Exception as e:
        logger.error(f"Toggle service error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def list_tools_handler(request):
    """GET /list_tools - Get all tools from all enabled services (Fractalic compatibility endpoint)
    Optimized to avoid duplicate tool retrieval + duplicate logging by passing tools into tools_info.
    """
    logger.info("list_tools_handler: Request received")
    try:
        # Get supervisor from app context instead of global variable
        supervisor = request.app['supervisor']
        logger.info(f"list_tools_handler: supervisor retrieved from app context, type = {type(supervisor)}")
        
        try:
            service_states = supervisor.service_states
            logger.info(f"list_tools_handler: service_states accessed, type = {type(service_states)}")
        except Exception as e:
            logger.error(f"list_tools_handler: Error accessing service_states: {e}")
            return web.json_response({'error': f'supervisor.service_states error: {str(e)}'}, status=500)
        
        # Now get actual tools since basic functionality is working
        all_tools = []
        total_token_count = 0
        enabled_services = [name for name, state in supervisor.service_states.items() if state == 'enabled']
        logger.info(f"list_tools_handler: Processing {len(enabled_services)} enabled services")

        for service_name in enabled_services:
            try:
                logger.debug(f"list_tools_handler: Getting tools for {service_name}")
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

        logger.info(f"list_tools_handler: Collected {len(all_tools)} tools total")
        
        # Test JSON serialization before returning to catch errors early
        response_data = {
            'tools': all_tools,
            'count': len(all_tools),
            'total_token_count': total_token_count,
            'services_count': len(enabled_services)
        }
        
        logger.info("list_tools_handler: Testing JSON serialization...")
        try:
            # Test serialization to catch errors before web.json_response
            import json
            json.dumps(response_data)
            logger.info("list_tools_handler: JSON serialization successful")
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error in list_tools: {e}")
            logger.error(f"Problematic data types: {[(k, type(v)) for k, v in response_data.items() if k != 'tools']}")
            if all_tools:
                logger.error(f"First tool data types: {[(k, type(v)) for k, v in all_tools[0].items()]}")
            return web.json_response({'error': f'JSON serialization error: {str(e)}'}, status=500)
        
        logger.info("list_tools_handler: Returning JSON response")
        return web.json_response(response_data)
        
    except Exception as e:
        logger.error(f"List tools error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def get_all_tools_handler(request):
    """GET /tools - Get all tools from all enabled services"""
    try:
        supervisor = request.app['supervisor']
        all_tools = {}
        total_tools = 0
        total_tokens = 0
        
        for service_name in supervisor.config.keys():
            service_status = supervisor.service_states.get(service_name, "enabled")
            if service_status != "enabled":
                continue
                
            try:
                tools = await supervisor.get_tools_for_service(service_name)
                tools_info = await supervisor.get_tools_info_for_service(service_name)
            except OAuthPending:
                tools = []
                tools_info = {"tool_count": 0, "token_count": 0, "tools_error": "auth_pending"}
            except Exception as e:
                tools = []
                tools_info = {"tool_count": 0, "token_count": 0, "tools_error": str(e)}
            
            all_tools[service_name] = {
                "tools": tools,
                "count": len(tools),
                "tool_count": tools_info.get("tool_count", len(tools)),
                "token_count": tools_info.get("token_count", 0)
            }
            
            if "tools_error" in tools_info:
                all_tools[service_name]["tools_error"] = tools_info["tools_error"]
            
            total_tools += tools_info.get("tool_count", len(tools))
            total_tokens += tools_info.get("token_count", 0)
        
        return web.json_response({
            "services": all_tools,
            "total_tools": total_tools,
            "total_tokens": total_tokens,
            "enabled_services": len([s for s in supervisor.service_states.values() if s == "enabled"])
        })
        
    except Exception as e:
        logger.error(f"Get all tools error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def get_tools_handler(request):
    """GET /tools/{name} - Get tools for a service with count and token information"""
    service_name = request.match_info['name']
    supervisor = request.app['supervisor']
    try:
        # Get both tools and tools info (includes token counting)
        try:
            tools = await supervisor.get_tools_for_service(service_name)
            tools_info = await supervisor.get_tools_info_for_service(service_name)
        except OAuthPending:
            tools = []
            tools_info = {"tool_count": 0, "token_count": 0, "tools_error": "auth_pending"}
        
        response = {
            "tools": tools, 
            "count": len(tools),
            "tool_count": tools_info.get("tool_count", len(tools)),
            "token_count": tools_info.get("token_count", 0)
        }

        # Generic OAuth pending heuristic (service-agnostic)
        prov = supervisor.oauth_providers.get(service_name)
        storage = supervisor.token_storages.get(service_name)
        if not tools and 'tools_error' not in response:
            tokens_present = False
            try:
                if storage:
                    tk = await storage.get_tokens()
                    tokens_present = tk is not None
            except Exception:
                pass
            if not tokens_present and prov:
                response['tools_error'] = 'auth_pending'
        
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
    supervisor = request.app['supervisor']
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
    supervisor = request.app['supervisor']
    
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
    supervisor = request.app['supervisor']
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
    supervisor = request.app['supervisor']
    
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
    supervisor = request.app['supervisor']
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
    supervisor = request.app['supervisor']
    
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
    supervisor = request.app['supervisor']
    
    if service_name not in supervisor.oauth_providers:
        return web.json_response({"error": "Service does not support OAuth"}, status=400)
    
    try:
        success = await supervisor.test_service_connection(service_name)
        # Structured log
        supervisor._log_event(service_name, "oauth_start", supervisor.config.get(service_name).transport if service_name in supervisor.config else None, time.perf_counter(), status="ok" if success else "pending")
        return web.json_response({"oauth_started": True, "service": service_name, "success": success})
    except Exception as e:
        logger.error(f"OAuth start error: {e}")
        supervisor._log_event(service_name, "oauth_start", supervisor.config.get(service_name).transport if service_name in supervisor.config else None, time.perf_counter(), status="error", error=type(e).__name__)
        return web.json_response({"error": str(e)}, status=500)

async def oauth_reset_handler(request):
    """POST /oauth/reset/{service} - Delete stored tokens and reset auth state"""
    service_name = request.match_info['service']
    supervisor = request.app['supervisor']
    if service_name not in supervisor.config:
        return web.json_response({"error": "Service not found"}, status=404)
    try:
        success = await supervisor.reset_oauth_tokens(service_name)
        supervisor._log_event(service_name, "oauth_reset", supervisor.config.get(service_name).transport if service_name in supervisor.config else None, time.perf_counter(), status="ok" if success else "failed")
        return web.json_response({
            "service": service_name,
            "reset": success
        }, status=200 if success else 500)
    except Exception as e:
        logger.error(f"OAuth reset error: {e}")
        supervisor._log_event(service_name, "oauth_reset", supervisor.config.get(service_name).transport if service_name in supervisor.config else None, time.perf_counter(), status="error", error=type(e).__name__)
        return web.json_response({"error": str(e)}, status=500)

async def oauth_authorize_handler(request):
    """POST /oauth/authorize/{service} - Proactively ensure provider + trigger auth.

    Note: Definition moved above create_app usage to avoid NameError during module import.
    """
    service_name = request.match_info['service']
    supervisor = request.app['supervisor']
    if service_name not in supervisor.config:
        return web.json_response({"error": "Service not found"}, status=404)
    try:
        ensured = supervisor.ensure_oauth_provider(service_name)
        if not ensured:
            supervisor._log_event(service_name, "oauth_authorize", supervisor.config.get(service_name).transport if service_name in supervisor.config else None, time.perf_counter(), status="error", error="provider_create_failed")
            return web.json_response({"error": "Could not create OAuth provider"}, status=500)
        # Trigger a lightweight connection test which will in turn start OAuth if needed
        try:
            started = await supervisor.test_service_connection(service_name)
        except OAuthPending:
            started = False
        supervisor._log_event(service_name, "oauth_authorize", supervisor.config.get(service_name).transport if service_name in supervisor.config else None, time.perf_counter(), status="ok" if started else "pending")
        return web.json_response({
            "service": service_name,
            "authorization_triggered": True,
            "connection_success": started,
            "pending": not started
        })
    except Exception as e:
        logger.error(f"OAuth authorize error: {e}")
        supervisor._log_event(service_name, "oauth_authorize", supervisor.config.get(service_name).transport if service_name in supervisor.config else None, time.perf_counter(), status="error", error=type(e).__name__)
        return web.json_response({"error": str(e)}, status=500)

async def oauth_status_all_handler(request):
    """GET /oauth/status - Summary of OAuth state for all services."""
    supervisor = request.app['supervisor']
    try:
        out = {}
        now = time.time()
        for name in supervisor.config.keys():
            storage = supervisor.token_storages.get(name)
            tokens_present = False
            obtained_age = None
            if storage:
                try:
                    tokens = await storage.get_tokens()
                    if tokens:
                        tokens_present = True
                        # Attempt to read obtained_at from file (non-critical)
                        fp = storage.file_path
                        if fp.exists():
                            import json as _json
                            with open(fp,'r') as f:
                                data=_json.load(f)
                            svc=data.get(name)
                            if svc and 'obtained_at' in svc:
                                obtained_age = round(now - svc['obtained_at'],1)
                except Exception:
                    pass
            # Expiry ETA (best-effort) from tokens file
            expiry_eta = None
            if storage and tokens_present:
                try:
                    fp = storage.file_path
                    if fp.exists():
                        import json as _json
                        with open(fp,'r') as f: data=_json.load(f)
                        rec = data.get(name)
                        if rec and rec.get('expires_in') and rec.get('obtained_at'):
                            ttl = rec['expires_in'] - (now - rec['obtained_at'])
                            expiry_eta = round(ttl,1)
                except Exception:
                    pass
            next_refresh = None
            if name in supervisor.oauth_refresh_run_at:
                try:
                    next_refresh = round(supervisor.oauth_refresh_run_at[name] - now,1)
                except Exception:
                    next_refresh = None
            out[name] = {
                'provider': name in supervisor.oauth_providers,
                'tokens': tokens_present,
                'obtained_age_s': obtained_age,
                'expires_in_s': expiry_eta,
                'next_refresh_in_s': next_refresh,
                'last_refresh_status': supervisor.oauth_refresh_status.get(name),
                'last_refresh_error': supervisor.oauth_refresh_error.get(name),
                'last_unauthorized_delta_s': (round(now - supervisor.last_unauthorized[name],1) if name in supervisor.last_unauthorized else None),
                'redirect_recent': (round(now - supervisor.oauth_redirect_times[name],1) if name in supervisor.oauth_redirect_times else None),
                'in_progress': supervisor.oauth_in_progress.get(name, False),
                'state': supervisor.service_states.get(name,'unknown')
            }
        return web.json_response(out)
    except Exception as e:
        logger.error(f"OAuth status error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def oauth_status_service_handler(request):
    """GET /oauth/status/{service} - Detailed OAuth state for one service."""
    service = request.match_info['service']
    supervisor = request.app['supervisor']
    if service not in supervisor.config:
        return web.json_response({'error':'service not found'}, status=404)
    try:
        res = await oauth_status_all_handler(request)
        data = await res.json()
        return web.json_response({service: data.get(service)})
    except Exception as e:
        logger.error(f"OAuth status service error: {e}")
        return web.json_response({'error': str(e)}, status=500)

# Legacy compatibility endpoints
async def add_server_handler(request):
    """POST /add_server - Add server configuration (legacy compatibility)"""
    try:
        body = await request.json()
        supervisor = request.app['supervisor']
        
        # Handle different JSON formats from frontend (legacy compatibility)
        if "jsonConfig" in body and isinstance(body["jsonConfig"], str):
            try:
                body = json.loads(body["jsonConfig"])
            except json.JSONDecodeError as e:
                return web.json_response(
                    {"success": False, "error": "Invalid JSON in jsonConfig field"}, 
                    status=400
                )
        
        # Extract server configuration
        name = None
        server_config = {}
        
        if "mcpServers" in body and isinstance(body["mcpServers"], dict):
            # Format: {"mcpServers": {"server-name": {...}}}
            servers = body["mcpServers"]
            if len(servers) != 1:
                return web.json_response(
                    {"success": False, "error": "Exactly one server must be provided"}, 
                    status=400
                )
            name, server_config = next(iter(servers.items()))
        else:
            # Standard format
            name = body.get("name")
            url = body.get("url")
            config = body.get("config", {})
            
            if not name:
                return web.json_response(
                    {"success": False, "error": "Server name is required"}, 
                    status=400
                )
            
            # Build server config based on URL or command
            if url:
                server_config = {"url": url, **config}
            elif "command" in config:
                server_config = config
            else:
                return web.json_response(
                    {"success": False, "error": "Server URL or command is required"}, 
                    status=400
                )
        
        # Add to mcp_servers.json configuration
        config_path = ROOT_DIR / "mcp_servers.json"
        try:
            if config_path.exists():
                current_config = json.loads(config_path.read_text())
            else:
                current_config = {"mcpServers": {}}
            
            # Add the new server
            current_config["mcpServers"][name] = server_config
            
            # Write back to config
            config_path.write_text(json.dumps(current_config, indent=2))
            
            # Add to supervisor's runtime config
            supervisor.config[name] = ServiceConfig.from_dict(name, server_config)
            
            return web.json_response({"success": True, "message": f"Server '{name}' added successfully"})
            
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to add server: {str(e)}"}, 
                status=500
            )
            
    except Exception as e:
        logger.error(f"Add server error: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def delete_server_handler(request):
    """POST /delete_server - Remove server (legacy compatibility)"""
    try:
        body = await request.json()
        supervisor = request.app['supervisor']
        name = body.get("name")
        
        if not name:
            return web.json_response(
                {"success": False, "error": "Server name is required"}, 
                status=400
            )
        
        # Remove from mcp_servers.json configuration
        config_path = ROOT_DIR / "mcp_servers.json"
        try:
            if config_path.exists():
                current_config = json.loads(config_path.read_text())
            else:
                return web.json_response(
                    {"success": False, "error": f"Server '{name}' not found"}, 
                    status=404
                )
            
            # Check if server exists
            if name not in current_config.get("mcpServers", {}):
                return web.json_response(
                    {"success": False, "error": f"Server '{name}' not found"}, 
                    status=404
                )
            
            # Remove from config
            del current_config["mcpServers"][name]
            
            # Write back to config
            config_path.write_text(json.dumps(current_config, indent=2))
            
            # Remove from supervisor's runtime config
            if name in supervisor.config:
                del supervisor.config[name]
            
            return web.json_response({"success": True, "message": f"Server '{name}' deleted successfully"})
            
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to delete server: {str(e)}"}, 
                status=500
            )
            
    except Exception as e:
        logger.error(f"Delete server error: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def kill_handler(request):
    """POST /kill - Kill manager process (legacy compatibility)"""
    try:
        # Respond immediately, then trigger shutdown
        async def delayed_shutdown():
            await asyncio.sleep(0.1)  # Allow response to be sent
            # Set global shutdown event if available
            if hasattr(request.app, '_shutdown_event'):
                request.app._shutdown_event.set()
            else:
                # Fallback: use system exit after cleanup
                import os
                os._exit(0)
        
        # Start shutdown in background
        asyncio.create_task(delayed_shutdown())
        
        return web.json_response({"status": "shutting down"})
        
    except Exception as e:
        logger.error(f"Kill handler error: {e}")
        return web.json_response({"error": str(e)}, status=500)

def create_app(supervisor_instance):
    """Create aiohttp application"""
    app = web.Application()

    # Setup CORS for localhost development
    cors = aiohttp_cors.setup(app, defaults={
        "http://localhost:3000": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        ),
        "http://127.0.0.1:3000": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        ),
        "http://localhost:8000": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        ),
        "http://127.0.0.1:8000": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        ),
        "http://localhost:8080": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        ),
        # Allow all origins in development (can be restricted for production)
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=False,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })

    # Request logging middleware (debug aid) - logs method, path, status, and duration
    @web.middleware
    async def _req_logger(request, handler):
        start = time.perf_counter()
        try:
            resp = await handler(request)
            try:
                logger.info(f"HTTP {request.method} {request.path} status={resp.status} ms={(time.perf_counter()-start)*1000:.1f}")
            except Exception:
                pass
            return resp
        except Exception as e:
            try:
                logger.info(f"HTTP {request.method} {request.path} status=ERR error={type(e).__name__} ms={(time.perf_counter()-start)*1000:.1f}")
            except Exception:
                pass
            raise
    app.middlewares.append(_req_logger)
    
    # Store supervisor reference in app context
    app['supervisor'] = supervisor_instance
    
    # API routes with CORS
    cors.add(app.router.add_get('/status', status_handler))
    cors.add(app.router.add_get('/status/complete', complete_status_handler))  # Complete MCP state in single call
    cors.add(app.router.add_get('/list_tools', list_tools_handler))  # Fractalic compatibility endpoint
    cors.add(app.router.add_post('/toggle/{name}', toggle_service_handler))
    cors.add(app.router.add_get('/tools', get_all_tools_handler))  # Get all tools from all services
    cors.add(app.router.add_get('/tools/{name}', get_tools_handler))
    cors.add(app.router.add_get('/capabilities/{name}', get_capabilities_handler))
    cors.add(app.router.add_post('/call/{service}/{tool}', call_tool_handler))
    
    # MCP Full Feature Set routes
    cors.add(app.router.add_get('/list_prompts', list_prompts_handler))
    cors.add(app.router.add_post('/prompt/{service}/{prompt}', get_prompt_handler))
    cors.add(app.router.add_get('/list_resources', list_resources_handler))
    cors.add(app.router.add_post('/resource/{service}/read', read_resource_handler))
    
    cors.add(app.router.add_post('/oauth/start/{service}', oauth_start_handler))
    cors.add(app.router.add_post('/oauth/reset/{service}', oauth_reset_handler))
    cors.add(app.router.add_post('/oauth/authorize/{service}', oauth_authorize_handler))
    cors.add(app.router.add_get('/oauth/status', oauth_status_all_handler))
    cors.add(app.router.add_get('/oauth/status/{service}', oauth_status_service_handler))
    # Force authorize (clear cooldown + trigger test) - debugging only
    async def oauth_force_authorize_handler(request):
        service = request.match_info['service']
        supervisor = request.app['supervisor']
        if service not in supervisor.config:
            return web.json_response({'error':'service not found'}, status=404)
        supervisor.clear_redirect_cooldown(service)
        try:
            supervisor.ensure_oauth_provider(service)
            # Force a connection test which will open browser if 401
            try:
                await supervisor.test_service_connection(service)
            except Exception as e:
                if not isinstance(e, Exception):  # placeholder
                    pass
            return web.json_response({'service': service, 'forced': True})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    cors.add(app.router.add_post('/oauth/force_authorize/{service}', oauth_force_authorize_handler))
    # Debug route (non-production sensitive) - returns masked internal state
    async def oauth_debug_handler(request):
        service = request.match_info['service']
        supervisor = request.app['supervisor']
        if service not in supervisor.config:
            return web.json_response({'error':'service not found'}, status=404)
        try:
            snapshot = supervisor.debug_oauth_state(service)
            supervisor._log_event(service, 'oauth_debug', supervisor.config.get(service).transport if service in supervisor.config else None, time.perf_counter(), status='ok')
            return web.json_response(snapshot)
        except Exception as e:
            supervisor._log_event(service, 'oauth_debug', supervisor.config.get(service).transport if service in supervisor.config else None, time.perf_counter(), status='error', error=type(e).__name__)
            return web.json_response({'error': str(e)}, status=500)
    cors.add(app.router.add_get('/oauth/debug/{service}', oauth_debug_handler))
    
    # OAuth callback route
    cors.add(app.router.add_get('/oauth/callback', oauth_callback_handler))
    
    # Legacy compatibility endpoints
    cors.add(app.router.add_post('/add_server', add_server_handler))
    cors.add(app.router.add_post('/delete_server', delete_server_handler))
    cors.add(app.router.add_post('/kill', kill_handler))
    
    return app

async def serve(port: int = 5859, host: str = '0.0.0.0', disable_signals: bool = False):
    """Run the HTTP server on specified host/port (default 0.0.0.0:5859).

    Revised signal handling: instead of raising KeyboardInterrupt from inside a signal
    handler (which can surface as 'weird' SIGINT traces), we set an asyncio.Event and
    allow the main coroutine to unwind gracefully. This avoids spurious stack traces
    and makes shutdown idempotent. Pass disable_signals=True to skip registration
    (useful under certain orchestrators / test harnesses).
    """
    import signal
    # Force working directory to repository root so relative paths (config, tokens) are consistent
    try:
        if os.getcwd() != str(ROOT_DIR):
            os.chdir(ROOT_DIR)
            logger.info(f"Changed working directory to repo root: {ROOT_DIR}")
    except Exception as _cwd_e:
        logger.warning(f"Could not change working directory to root: {_cwd_e}")

    # Create supervisor instance for this server (use the enhanced version in this file)
    supervisor = MCPSupervisorV2()
    
    app = create_app(supervisor)
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"MCP Manager V2 started on http://{host}:{port}")
    logger.info(f"OAuth callback endpoint available at http://localhost:{port}/oauth/callback")
    logger.info("Per-request session pattern - no persistent connections")

    # Populate initial cache asynchronously so server can accept requests immediately
    async def _background_cache():
        try:
            # Use comprehensive caching to reduce subsequent API calls
            await supervisor._populate_initial_cache(comprehensive=True)
        except Exception as e:
            logger.error(f"Initial cache population failed: {e}")
    asyncio.create_task(_background_cache())
    logger.info("Initial comprehensive cache population started in background")

    shutdown_event = asyncio.Event()

    def _graceful(sig: int, frame=None):  # frame kept for signal API compatibility
        try:
            if not shutdown_event.is_set():
                logger.info(f"Signal {signal.Signals(sig).name} received - initiating graceful shutdown")
                shutdown_event.set()
        except Exception:
            # Never let signal handler explode
            pass

    if not disable_signals:
        loop = asyncio.get_running_loop()
        # Use loop.add_signal_handler when available (Unix); fallback to signal.signal
        try:
            loop.add_signal_handler(signal.SIGINT, _graceful, signal.SIGINT)
            loop.add_signal_handler(signal.SIGTERM, _graceful, signal.SIGTERM)
        except NotImplementedError:  # e.g. on Windows event loop policy
            signal.signal(signal.SIGINT, _graceful)
            signal.signal(signal.SIGTERM, _graceful)

    # Wait until shutdown requested
    await shutdown_event.wait()

    logger.info("Shutdown event set - cleaning up server")
    try:
        await runner.cleanup()
    finally:
        logger.info("Server shutdown complete")

# CLI interface
async def cli_status(port: int = 5859):
    """CLI command to get status via HTTP"""
    import aiohttp
    
    url = f"http://localhost:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(json.dumps(data, indent=2))
                else:
                    print(f"❌ Status request failed: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to server at port {port}: {e}")
    except Exception as e:
        print(f"❌ Error getting status: {e}")

async def cli_test_service(service_name: str, port: int = 5859):
    """CLI command to test a service via HTTP"""
    import aiohttp
    
    url = f"http://localhost:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{url}/test/{service_name}") as response:
                if response.status == 200:
                    data = await response.json()
                    success = data.get("success", False)
                    print(f"Service {service_name}: {'✅ Connected' if success else '❌ Failed'}")
                else:
                    print(f"❌ Test request failed: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to server at port {port}: {e}")
    except Exception as e:
        print(f"❌ Error testing service: {e}")

async def cli_get_tools(service_name: str, port: int = 5859):
    """CLI command to get tools via HTTP"""
    import aiohttp
    
    url = f"http://localhost:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/tools/{service_name}") as response:
                if response.status == 200:
                    data = await response.json()
                    tools = data.get("tools", [])
                    print(f"Tools for {service_name}:")
                    for tool in tools:
                        print(f"  - {tool['name']}: {tool['description']}")
                else:
                    print(f"❌ Tools request failed: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to server at port {port}: {e}")
    except Exception as e:
        print(f"❌ Error getting tools: {e}")

async def cli_kill(port: int = 5859):
    """CLI command to kill running server via HTTP"""
    import aiohttp
    
    url = f"http://localhost:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{url}/kill") as response:
                if response.status == 200:
                    print("✅ Kill command sent successfully")
                else:
                    print(f"❌ Kill command failed: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to server at port {port}: {e}")
    except Exception as e:
        print(f"❌ Error sending kill command: {e}")

def main():
    """Main entry point - full legacy CLI compatibility"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='MCP Manager V2 - Per-Request Sessions')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Serve command - loads services and starts server
    serve_parser = subparsers.add_parser('serve', help='Start HTTP server')
    serve_parser.add_argument('--port', type=int, default=5859, help='Port to listen on (default: 5859)')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host/interface to bind (default: 0.0.0.0)')
    serve_parser.add_argument('--disable-signals', action='store_true', help='Do not register SIGINT/SIGTERM handlers (diagnostics / embedding)')
    
    # Status command - HTTP client
    status_parser = subparsers.add_parser('status', help='Get services status')
    status_parser.add_argument('--port', type=int, default=5859, help='Port of server to query (default: 5859)')
    
    # Test command - HTTP client
    test_parser = subparsers.add_parser('test', help='Test service connection')
    test_parser.add_argument('service', help='Service name to test')
    test_parser.add_argument('--port', type=int, default=5859, help='Port of server to query (default: 5859)')
    
    # Tools command - HTTP client
    tools_parser = subparsers.add_parser('tools', help='Get tools for service')
    tools_parser.add_argument('service', help='Service name')
    tools_parser.add_argument('--port', type=int, default=5859, help='Port of server to query (default: 5859)')
    
    # Kill command - HTTP client
    kill_parser = subparsers.add_parser('kill', help='Kill running server')
    kill_parser.add_argument('--port', type=int, default=5859, help='Port of server to kill (default: 5859)')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        # Only serve command loads supervisor and services
        asyncio.run(serve(port=args.port, host=args.host, disable_signals=args.disable_signals))
    elif args.command in ['status', 'test', 'tools', 'kill']:
        # HTTP client commands - prevent supervisor initialization
        os.environ['FRACTALIC_NO_INIT'] = '1'
        if args.command == 'status':
            asyncio.run(cli_status(args.port))
        elif args.command == 'test':
            asyncio.run(cli_test_service(args.service, args.port))
        elif args.command == 'tools':
            asyncio.run(cli_get_tools(args.service, args.port))
        elif args.command == 'kill':
            asyncio.run(cli_kill(args.port))
    else:
        print("DEBUG: No valid command, showing help")
        parser.print_help()

if __name__ == '__main__':
    main()
