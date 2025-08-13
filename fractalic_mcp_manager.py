#!/usr/bin/env python3
# fractalic_mcp_manager.py – Fractalic MCP supervisor (auto-transport, health, back-off)
#
# CLI -----------------------------------------------------
#   python fractalic_mcp_manager.py serve        [--port 5859]
#   python fractalic_mcp_manager.py status       [--port 5859]
#   python fractalic_mcp_manager.py tools        [--port 5859]
#   python fractalic_mcp_manager.py start NAME   [--port 5859]
#   python fractalic_mcp_manager.py stop  NAME   [--port 5859]
#   python fractalic_mcp_manager.py restart NAME [--port 5859]
# ---------------------------------------------------------
from __future__ import annotations

import argparse, asyncio, contextlib, dataclasses, datetime, json, os, shlex, signal, subprocess, sys, time, gc
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TextIO

import aiohttp
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions, CorsViewMixin

from mcp.client.session          import ClientSession
from mcp.client.stdio            import stdio_client, StdioServerParameters
from mcp.client.streamable_http  import streamablehttp_client
from mcp.client.sse              import sse_client
from mcp.client.auth             import OAuthClientProvider
from mcp.shared.auth             import OAuthClientMetadata, OAuthToken

import errno
import tiktoken

# OAuth token storage directory
OAUTH_STORAGE_DIR = os.environ.get("OAUTH_STORAGE_PATH", "/tmp/fractalic_oauth")

class FileTokenStorage:
    """File-based OAuth token storage for persistence across restarts"""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.storage_dir = Path(OAUTH_STORAGE_DIR)
        self.token_file = self.storage_dir / f"{server_name}_tokens.json"
        self.client_file = self.storage_dir / f"{server_name}_client.json"
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    async def get_tokens(self) -> Optional[OAuthToken]:
        """Retrieve stored OAuth tokens"""
        try:
            if self.token_file.exists():
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                return OAuthToken(**token_data)
        except Exception as e:
            log(f"FileTokenStorage({self.server_name}): Failed to load tokens: {e}")
        return None
        
    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store OAuth tokens persistently"""
        try:
            with open(self.token_file, 'w') as f:
                json.dump(tokens.model_dump(), f, indent=2)
            log(f"FileTokenStorage({self.server_name}): Tokens saved to {self.token_file}")
        except Exception as e:
            log(f"FileTokenStorage({self.server_name}): Failed to save tokens: {e}")
            
    async def get_client_info(self):
        """Retrieve stored OAuth client information"""
        try:
            if self.client_file.exists():
                with open(self.client_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            log(f"FileTokenStorage({self.server_name}): Failed to load client info: {e}")
        return None
        
    async def set_client_info(self, client_info) -> None:
        """Store OAuth client information persistently"""
        try:
            with open(self.client_file, 'w') as f:
                json.dump(client_info, f, indent=2)
            log(f"FileTokenStorage({self.server_name}): Client info saved to {self.client_file}")
        except Exception as e:
            log(f"FileTokenStorage({self.server_name}): Failed to save client info: {e}")

def migrate_oauth_tokens(source_dir: str, target_dir: str) -> bool:
    """
    Migrate OAuth tokens from source to target directory for deployment.
    
    Args:
        source_dir: Source directory (e.g., development environment)
        target_dir: Target directory (e.g., container volume)
        
    Returns:
        bool: True if migration successful, False otherwise
    """
    try:
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        if not source_path.exists():
            log(f"OAuth migration: Source directory does not exist: {source_path}")
            return False
            
        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Find all OAuth token files
        token_files = list(source_path.glob("*_tokens.json"))
        client_files = list(source_path.glob("*_client.json"))
        
        if not token_files:
            log(f"OAuth migration: No token files found in {source_path}")
            return True  # Not an error if no tokens exist
            
        # Copy token files
        copied_count = 0
        for token_file in token_files:
            target_file = target_path / token_file.name
            with open(token_file, 'r') as src, open(target_file, 'w') as dst:
                dst.write(src.read())
            log(f"OAuth migration: Copied {token_file.name}")
            copied_count += 1
            
        # Copy client files
        for client_file in client_files:
            target_file = target_path / client_file.name
            with open(client_file, 'r') as src, open(target_file, 'w') as dst:
                dst.write(src.read())
            log(f"OAuth migration: Copied {client_file.name}")
            
        log(f"OAuth migration: Successfully migrated {copied_count} token files")
        return True
        
    except Exception as e:
        log(f"OAuth migration failed: {e}")
        return False

def list_oauth_tokens(storage_dir: str = None) -> Dict[str, Dict]:
    """
    List all available OAuth tokens and their status.
    
    Args:
        storage_dir: OAuth storage directory (uses default if None)
        
    Returns:
        Dict mapping server names to token information
    """
    if storage_dir is None:
        storage_dir = OAUTH_STORAGE_DIR
        
    result = {}
    storage_path = Path(storage_dir)
    
    if not storage_path.exists():
        return result
        
    try:
        for token_file in storage_path.glob("*_tokens.json"):
            server_name = token_file.stem.replace("_tokens", "")
            
            try:
                with open(token_file, 'r') as f:
                    token_data = json.load(f)
                    
                # Check token expiry
                expires_at = token_data.get("expires_at")
                is_expired = False
                if expires_at:
                    is_expired = time.time() > expires_at
                    
                result[server_name] = {
                    "token_file": str(token_file),
                    "has_access_token": bool(token_data.get("access_token")),
                    "has_refresh_token": bool(token_data.get("refresh_token")),
                    "expires_at": expires_at,
                    "is_expired": is_expired,
                    "last_modified": datetime.datetime.fromtimestamp(
                        token_file.stat().st_mtime
                    ).isoformat()
                }
                
            except Exception as e:
                result[server_name] = {
                    "error": f"Failed to read token file: {e}"
                }
                
    except Exception as e:
        log(f"Failed to list OAuth tokens: {e}")
        
    return result

# Import litellm for optional token counting alignment with Fractalic execution
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("Warning: litellm not available, falling back to tiktoken for token counting")

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# -------------------------------------------------------------------- Service Classification
class ServiceProfile:
    """Auto-detected service profile for adaptive timeout/retry settings"""
    def __init__(self, name: str, spec: dict, transport: Transport):
        self.name = name
        self.spec = spec
        self.transport = transport
        
        # Classify service characteristics
        self.is_external = self._detect_external_service()
        self.is_third_party_api = self._detect_third_party_api()
        self.is_high_activity = self._detect_high_activity_service()
        self.complexity_level = self._assess_complexity()
        
        # Apply adaptive settings
        self.init_timeout = self._calculate_init_timeout()
        self.retry_count = self._calculate_retry_count()
        self.health_failure_limit = self._calculate_health_failure_limit()
        self.max_retries = self._calculate_max_retries()
        self.tool_request_cooldown = self._calculate_tool_request_cooldown()
    
    def _detect_external_service(self) -> bool:
        """Detect if this is an external service that might be unreliable"""
        if self.transport != "http":
            return False
            
        url = self.spec.get("url", "").lower()
        
        # Common external service indicators
        external_domains = [
            "zapier.com", "api.zapier.com", "mcp.zapier.com",
            "api.github.com", "github.com",
            "api.openai.com", "openai.com",
            "googleapis.com", "google.com",
            "api.slack.com", "slack.com",
            "api.notion.com", "notion.so",
            "api.trello.com", "trello.com",
            "api.airtable.com", "airtable.com"
        ]
        
        # Check if URL contains any external domain
        return any(domain in url for domain in external_domains)
    
    def _detect_third_party_api(self) -> bool:
        """Detect if this is a third-party API (not localhost/internal)"""
        if self.transport != "http":
            return False
            
        url = self.spec.get("url", "").lower()
        
        # Internal/localhost indicators
        internal_indicators = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
        
        return not any(indicator in url for indicator in internal_indicators)
    
    def _detect_high_activity_service(self) -> bool:
        """Detect services known to make frequent/excessive requests"""
        # Known high-activity services that generate frequent tool requests
        high_activity_services = [
            "desktop-commander",  # Generates tools list repeatedly
            "playwright-mcp",     # May generate many browser-related tools
            "automation-server"   # General automation services tend to be chatty
        ]
        
        # Check if service name matches any known high-activity services
        return any(service_name in self.name.lower() for service_name in high_activity_services)

    def _assess_complexity(self) -> str:
        """Assess service complexity: 'simple', 'medium', 'complex'"""
        # Check for complexity indicators
        complexity_indicators = 0
        
        # External services are inherently more complex
        if self.is_external:
            complexity_indicators += 2
        
        # Third-party APIs add complexity
        if self.is_third_party_api:
            complexity_indicators += 1
            
        # HTTP services are generally more complex than stdio
        if self.transport == "http":
            complexity_indicators += 1
            
        # Check for environment complexity (many env vars = more complex setup)
        env_vars = self.spec.get("env", {})
        if len(env_vars) > 3:
            complexity_indicators += 1
        
        # High-activity services add complexity due to rate limiting needs
        if self.is_high_activity:
            complexity_indicators += 1
            
        # Classify based on indicators
        if complexity_indicators >= 4:
            return "complex"
        elif complexity_indicators >= 2:
            return "medium"
        else:
            return "simple"
    
    def _calculate_init_timeout(self) -> int:
        """Calculate appropriate initialization timeout"""
        base_timeout = 30  # Default for stdio services
        
        if self.transport == "http":
            base_timeout = 45  # HTTP services need more time
            
        if self.is_external:
            base_timeout += 15  # External services need extra time
            
        if self.complexity_level == "complex":
            base_timeout += 10
        elif self.complexity_level == "medium":
            base_timeout += 5
            
        return base_timeout
    
    def _calculate_retry_count(self) -> int:
        """Calculate appropriate retry count for startup"""
        base_retries = 3  # Default
        
        if self.is_external:
            base_retries += 2  # External services can be flaky
            
        if self.transport == "http":
            base_retries += 1  # HTTP services might need more retries
            
        return base_retries
    
    def _calculate_health_failure_limit(self) -> int:
        """Calculate how many health failures to tolerate"""
        base_limit = 5  # Default
        
        if self.is_external:
            base_limit += 5  # External services can have temporary outages
            
        if self.complexity_level == "complex":
            base_limit += 2
            
        return base_limit
    
    def _calculate_max_retries(self) -> int:
        """Calculate maximum retries before marking as errored"""
        base_retries = MAX_RETRY  # Default 5
        
        if self.is_external:
            base_retries += 3  # External services need more chances
            
        if self.transport == "http":
            base_retries += 1  # HTTP services might need more retries
            
        return base_retries
    
    def _calculate_tool_request_cooldown(self) -> float:
        """Calculate cooldown period between tool requests to prevent spam"""
        if self.is_high_activity:
            return 2.0  # 2 second cooldown for high-activity services
        elif self.complexity_level == "complex":
            return 1.0  # 1 second cooldown for complex services
        else:
            return 0.5  # 0.5 second cooldown for normal services

# -------------------------------------------------------------------- constants
CONF_PATH    = Path(__file__).parent / "mcp_servers.json"
DEFAULT_PORT = 5859

State     = Literal["starting", "running", "retrying", "stopped", "errored"]
Transport = Literal["stdio", "http"]

TIMEOUT_INITIAL = 120      # s – Increased timeout for slow operations like external services
HEALTH_INT    = 45         # s – between health probes (increased to avoid heartbeat clashes)
SESSION_TTL   = 3600       # s – refresh session after this period
MAX_RETRY     = 5
BACKOFF_BASE  = 2          # exponential back-off

# Configuration flags
ENABLE_SCHEMA_SANITIZATION = True  # Set to True to enable Vertex AI schema sanitization
USE_LITELLM_TOKEN_COUNT = True     # Set to True to use LiteLLM token counting (aligned with Fractalic execution), False for tiktoken direct counting

# -------------------------------------------------------------------- Vertex AI Schema Sanitization
def sanitize_tool_schema(tool_obj: dict, max_depth: int = 6) -> dict:
    """
    Sanitize MCP tool schema for Vertex AI/Gemini compatibility.
    
    Vertex AI has limitations:
    - Array types like ["object", "null"] cause "Proto field is not repeating" errors
    - Deep nesting beyond ~6-9 levels causes validation failures
    - Some JSON Schema constructs like anyOf, oneOf are not supported
    
    Args:
        tool_obj: Tool object with potential inputSchema
        max_depth: Maximum nesting depth allowed (default 6)
    
    Returns:
        Sanitized tool object safe for Vertex AI
    """
    if not isinstance(tool_obj, dict):
        return tool_obj
    
    # Create a copy to avoid mutating the original
    sanitized = tool_obj.copy()
    
    # Apply sanitization to inputSchema if present
    if "inputSchema" in sanitized:
        sanitized["inputSchema"] = _sanitize_schema_recursive(sanitized["inputSchema"], max_depth, 0)
    
    return sanitized

def _sanitize_schema_recursive(schema: any, max_depth: int, current_depth: int) -> any:
    """Recursively sanitize a JSON schema for Vertex AI compatibility."""
    if current_depth >= max_depth:
        # At max depth, return a simple fallback
        return {"type": "string", "description": "Complex nested data (simplified for compatibility)"}
    
    if not isinstance(schema, dict):
        return schema
    
    sanitized = {}
    
    for key, value in schema.items():
        if key == "type" and isinstance(value, list):
            # Convert array types to single type - use first non-null type
            sanitized[key] = _get_first_valid_type(value)
        elif key == "format":
            # Remove unsupported format fields for Vertex AI
            # Vertex AI only supports "enum" and "date-time" formats for STRING type
            if value in ["enum", "date-time"]:
                sanitized[key] = value
            # Skip unsupported formats like "uuid", "uri", etc. by not adding them to sanitized
        elif key in ["anyOf", "oneOf"]:
            # Remove unsupported constructs entirely, replace with simple string type
            sanitized.update(_simplify_union_type(value, max_depth, current_depth))
            continue  # Skip the original key
        elif key == "properties" and isinstance(value, dict):
            # Recursively sanitize properties but limit depth
            sanitized[key] = {}
            for prop_name, prop_schema in value.items():
                sanitized[key][prop_name] = _sanitize_schema_recursive(prop_schema, max_depth, current_depth + 1)
        elif key == "items" and isinstance(value, dict):
            # Sanitize array item schemas
            sanitized[key] = _sanitize_schema_recursive(value, max_depth, current_depth + 1)
        elif isinstance(value, dict):
            # Recursively sanitize nested objects
            sanitized[key] = _sanitize_schema_recursive(value, max_depth, current_depth + 1)
        elif isinstance(value, list):
            # Handle lists that might contain schemas
            sanitized[key] = [
                _sanitize_schema_recursive(item, max_depth, current_depth + 1) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            # Keep primitive values as-is
            sanitized[key] = value
    
    return sanitized

def _get_first_valid_type(type_list: list) -> str:
    """Get the first valid type from a list, preferring non-null types."""
    if not type_list:
        return "string"
    
    # Prefer non-null types
    for t in type_list:
        if t != "null":
            return t
    
    # If only null, default to string
    return "string"

def _simplify_union_type(union_value: any, max_depth: int, current_depth: int) -> dict:
    """Simplify anyOf/oneOf constructs to a basic type."""
    if isinstance(union_value, list) and union_value:
        # Take the first option and sanitize it
        first_option = union_value[0]
        if isinstance(first_option, dict):
            return _sanitize_schema_recursive(first_option, max_depth, current_depth)
    
    # Fallback to string type
    return {"type": "string", "description": "Union type (simplified for compatibility)"}

# -------------------------------------------------------------------- Custom JSON encoder
class MCPEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle dataclasses
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        # Handle Pydantic models
        if hasattr(obj, "model_dump_json"):
            return json.loads(obj.model_dump_json())
        # Handle ChatCompletion related objects
        if hasattr(obj, '__class__') and 'ChatCompletion' in str(type(obj)):
            return self._handle_chat_completion_object(obj)
        # Handle CallToolResult objects and other custom classes
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        # Handle datetime objects
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        # Handle other non-serializable objects
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # Fallback to string representation
    
    def _handle_chat_completion_object(self, obj):
        """Handle ChatCompletion related objects safely"""
        try:
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                return obj.dict()
            elif hasattr(obj, '__dict__'):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            else:
                return str(obj)
        except Exception:
            return str(obj)

# -------------------------------------------------------------------- helpers
def tool_to_obj(t):
    if isinstance(t, dict):
        return t                          # already JSON-ready
    if dataclasses.is_dataclass(t):
        return dataclasses.asdict(t)      # MCP canonical form
    return json.loads(t.model_dump_json()) if hasattr(t, "model_dump_json") else str(t)

def ts() -> str: return time.strftime("%H:%M:%S", time.localtime())
def log(msg: str): print(f"[{ts()}] {msg}", file=sys.stderr)

# ==================================================================== StderrCapture
class StderrCapture:
    """A TextIO wrapper that captures stderr output and stores it in a buffer."""
    
    def __init__(self, server_name: str, stderr_buffer: list, original_stderr: TextIO):
        self.server_name = server_name
        self.stderr_buffer = stderr_buffer
        self.original_stderr = original_stderr
        self._buffer_limit = 1000
    
    def write(self, text: str) -> int:
        """Write text to both the original stderr and capture buffer."""
        # Write to original stderr first
        count = self.original_stderr.write(text)
        
        # Split text into lines and add to buffer with timestamps
        if text:
            lines = text.splitlines(keepends=True)
            for line in lines:
                if line.strip():  # Only capture non-empty lines
                    entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "line": f"[{self.server_name}] {line.rstrip()}"
                    }
                    self.stderr_buffer.append(entry)
                    
                    # Limit buffer size
                    if len(self.stderr_buffer) > self._buffer_limit:
                        del self.stderr_buffer[0:len(self.stderr_buffer) - self._buffer_limit]
        
        return count
    
    def flush(self):
        """Flush the original stderr."""
        return self.original_stderr.flush()
    
    def close(self):
        """Close the original stderr."""
        return self.original_stderr.close()
    
    def fileno(self):
        """Return the file descriptor of the original stderr."""
        return self.original_stderr.fileno()
    
    def readable(self) -> bool:
        return False
    
    def writable(self) -> bool:
        return True
    
    def seekable(self) -> bool:
        return False
    
    def isatty(self) -> bool:
        """Check if the original stderr is a TTY."""
        return self.original_stderr.isatty()
    
    def __getattr__(self, name):
        """Delegate any other attribute access to the original stderr."""
        return getattr(self.original_stderr, name)

# ==================================================================== Child
class Child:
    def __init__(self, name: str, spec: dict):
        self.name   = name
        self.spec   = spec
        self.state  : State = "stopped"
        t_explicit = spec.get("transport") or spec.get("type")
        if t_explicit:
            self.transport: Transport = t_explicit
        elif "url" in spec:
            self.transport = "http"
        else:
            self.transport = "stdio"
            
        # Create service profile for adaptive behavior
        self.profile = ServiceProfile(name, spec, self.transport)
        
        # Core MCP client state (SDK-managed)
        self.session     : Optional[ClientSession] = None
        self.session_at  = 0.0
        self._transport_context = None
        self._session_context = None
        
        # Health and lifecycle management
        self._health     = None
        self.retries     = 0
        self.started_at  = None
        self._cmd_q      : asyncio.Queue = asyncio.Queue()
        self._runner     = asyncio.create_task(self._loop())
        self.healthy     = False
        self.restart_count = 0
        self.last_error   = None
        self.last_tool_request = 0.0
        self.health_failures = 0
        self.last_restart_time = 0
        
        # Output capture (for UI display)
        self.stdout_buffer = []
        self.stderr_buffer = []
        self.last_output_renewal = None
        self._output_buffer_limit = 1000
        
        # Tools caching
        self._cached_tools = None
        self._tools_cache_time = 0

        # Service profile for adaptive settings
        self.service_profile = ServiceProfile(name, spec, self.transport)

    async def start(self):
        # Check if runner exists and is still running
        if not self._runner or self._runner.done():
            log(f"{self.name}: Creating new runner task for start")
            self._runner = asyncio.create_task(self._loop())
        
        await self._cmd_q.put(("start",))

    async def stop(self):
        await self._cmd_q.put(("stop",))
        
    async def cleanup(self):
        """Comprehensive cleanup of all resources."""
        # Cancel the main loop
        if self._runner and not self._runner.done():
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
                
        # Stop the child
        await self._do_stop()
        
        # Clear buffers
        self.stdout_buffer.clear()
        self.stderr_buffer.clear()
        
        log(f"{self.name}: Cleanup complete")

    def _setup_oauth_auth(self):
        """
        Setup OAuth authentication using MCP SDK's built-in OAuth 2.1 support.
        Uses persistent file-based token storage for seamless deployment.
        """
        try:
            from pydantic import AnyUrl
            
            # Create persistent token storage
            token_storage = FileTokenStorage(self.name)
            
            # Default OAuth client metadata for automatic authentication
            client_metadata = OAuthClientMetadata(
                client_name=f"Fractalic MCP Client - {self.name}",
                redirect_uris=[AnyUrl("http://localhost:0/callback")],  # Use dynamic port
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope="user",  # Default scope, server will specify what's needed
            )
            
            # OAuth redirect and callback handlers for interactive authentication
            async def handle_redirect(auth_url: str):
                """Handle OAuth authorization redirect - open browser automatically"""
                import webbrowser
                import platform
                
                log(f"{self.name}: OAuth authentication required")
                log(f"{self.name}: Opening browser for authorization...")
                
                try:
                    # Try to open the browser automatically
                    if platform.system() == "Darwin":  # macOS
                        os.system(f'open "{auth_url}"')
                    elif platform.system() == "Windows":
                        os.system(f'start "" "{auth_url}"')
                    else:  # Linux
                        os.system(f'xdg-open "{auth_url}"')
                    
                    log(f"{self.name}: Browser opened. Please complete authorization and return to this application.")
                except Exception as e:
                    log(f"{self.name}: Could not open browser automatically: {e}")
                    log(f"{self.name}: Please manually visit: {auth_url}")
                
            async def handle_callback():
                """Handle OAuth callback - automatic local server approach"""
                log(f"{self.name}: Starting local callback server...")
                
                try:
                    # Start a temporary local HTTP server for OAuth callback
                    import asyncio
                    from aiohttp import web, ClientTimeout
                    import socket
                    
                    # Find an available port automatically
                    def find_free_port():
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(('', 0))
                            s.listen(1)
                            port = s.getsockname()[1]
                        return port
                    
                    callback_port = find_free_port()
                    callback_data = {"code": None, "state": None, "error": None}
                    
                    async def callback_handler(request):
                        """Handle the OAuth callback request"""
                        callback_data["code"] = request.query.get("code")
                        callback_data["state"] = request.query.get("state") 
                        callback_data["error"] = request.query.get("error")
                        
                        if callback_data["code"]:
                            html = """
                            <html><body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                                <h2 style="color: green;">✅ Authorization Successful!</h2>
                                <p>You can close this window and return to Fractalic.</p>
                                <script>setTimeout(() => window.close(), 3000);</script>
                            </body></html>
                            """
                        else:
                            html = f"""
                            <html><body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                                <h2 style="color: red;">❌ Authorization Failed</h2>
                                <p>Error: {callback_data['error']}</p>
                                <p>Please try again or contact support.</p>
                            </body></html>
                            """
                            
                        return web.Response(text=html, content_type="text/html")
                    
                    # Create temporary web app
                    app = web.Application()
                    app.router.add_get("/callback", callback_handler)
                    
                    # Start server on available port
                    runner = web.AppRunner(app)
                    await runner.setup()
                    site = web.TCPSite(runner, "localhost", callback_port)
                    await site.start()
                    
                    callback_url = f"http://localhost:{callback_port}/callback"
                    log(f"{self.name}: Local callback server started on {callback_url}")
                    log(f"{self.name}: Waiting for OAuth authorization (timeout: 120s)...")
                    
                    # Update redirect URI to use the actual port
                    client_metadata.redirect_uris = [AnyUrl(callback_url)]
                    
                    # Wait for callback with timeout
                    timeout = 120  # 2 minutes
                    start_time = time.time()
                    
                    while time.time() - start_time < timeout:
                        if callback_data["code"] or callback_data["error"]:
                            break
                        await asyncio.sleep(0.5)
                    
                    # Cleanup
                    await runner.cleanup()
                    
                    if callback_data["code"]:
                        log(f"{self.name}: OAuth authorization successful ✅")
                        return callback_data["code"], callback_data["state"]
                    elif callback_data["error"]:
                        log(f"{self.name}: OAuth authorization failed: {callback_data['error']}")
                        return "", None
                    else:
                        log(f"{self.name}: OAuth authorization timed out")
                        return "", None
                        
                except Exception as e:
                    log(f"{self.name}: Local callback server failed: {e}")
                    # Fallback to console input
                    return await self._console_oauth_callback()
                    
            async def _console_oauth_callback(self):
                """Fallback console-based OAuth callback"""
                log(f"{self.name}: ============ OAUTH AUTHORIZATION REQUIRED ============")
                log(f"{self.name}: ")
                log(f"{self.name}: 🌐 A browser window should have opened automatically.")
                log(f"{self.name}: 🔐 Please complete the authorization in your browser.")
                log(f"{self.name}: ")
                log(f"{self.name}: 📋 AFTER authorizing, the browser will show a URL like:")
                log(f"{self.name}: 📋 http://localhost:XXXX/callback?code=abc123&state=xyz")
                log(f"{self.name}: ")
                log(f"{self.name}: 📝 Copy that ENTIRE URL and paste it below:")
                log(f"{self.name}: =====================================================")
                
                try:
                    import sys
                    if hasattr(sys, 'ps1'):  # Interactive shell
                        callback_url = input(f"[{self.name}] 📝 Paste the full callback URL here: ")
                    else:
                        log(f"{self.name}: ❌ Non-interactive mode - manual OAuth not supported")
                        log(f"{self.name}: 💡 Run Fractalic interactively for OAuth authentication")
                        return "", None
                        
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(callback_url.strip())
                    params = parse_qs(parsed.query)
                    
                    code = params.get("code", [None])[0]
                    state = params.get("state", [None])[0]
                    
                    if code:
                        log(f"{self.name}: ✅ Authorization code received successfully!")
                        return code, state
                    else:
                        log(f"{self.name}: ❌ No authorization code found in URL")
                        log(f"{self.name}: 💡 Make sure you copied the complete URL with ?code=...")
                        return "", None
                        
                except Exception as e:
                    log(f"{self.name}: ❌ Console OAuth callback failed: {e}")
                    return "", None
            
            # Create OAuth provider using SDK with persistent storage
            oauth_provider = OAuthClientProvider(
                server_url=self.spec["url"],  # MCP server will provide OAuth endpoints
                client_metadata=client_metadata,
                storage=token_storage,  # Use file-based storage
                redirect_handler=handle_redirect,
                callback_handler=handle_callback,
            )
            
            log(f"{self.name}: OAuth 2.1 configured with persistent storage at {token_storage.storage_dir}")
            return oauth_provider
            
        except Exception as e:
            log(f"{self.name}: OAuth setup failed, will try without authentication: {e}")
            return None

    # REMOVED: All custom STDIO/HTTP code - now handled by MCP SDK
    # The SDK provides:
    # - Automatic transport selection (stdio_client, streamablehttp_client, sse_client)
    # - Built-in error handling and retry logic
    # - OAuth 2.1 authentication support
    # - Proper session lifecycle management

    async def _loop(self):
        while True:
            msg = await self._cmd_q.get()
            if msg[0] == "start":
                await self._do_start()
            elif msg[0] == "stop":
                await self._do_stop()
            elif msg[0] == "exit":
                await self._do_stop()
                break

    async def _do_start(self):
        if self.state == "running":
            return
        
        self.state = "starting"
        self.retries = 0
        
        try:
            # Handle startup delay
            startup_delay = int(self.spec.get("env", {}).get("STARTUP_DELAY", "0"))
            if startup_delay > 0:
                log(f"Waiting {startup_delay}ms for {self.name} to initialize...")
                await asyncio.sleep(startup_delay / 1000)
            
            # Spawn process if needed
            await self._spawn_if_needed()
            
            # Try to establish session with retries
            # Use adaptive retry count based on service profile
            base_retry_count = int(self.spec.get("env", {}).get("RETRY_COUNT", "3"))
            retry_count = max(base_retry_count, self.profile.retry_count)
            retry_delay = int(self.spec.get("env", {}).get("RETRY_DELAY", "2000")) / 1000
            
            for attempt in range(retry_count):
                try:
                    await self._ensure_session(force=True)
                    # Test session by getting cached tools (avoids repeated list_tools calls)
                    tools = await asyncio.wait_for(self.tools(), timeout=15)
                    if tools and (not isinstance(tools, dict) or "error" not in tools):
                        log(f"{self.name} tools available after {attempt + 1} attempts")
                        break
                except Exception as e:
                    error_msg = str(e)
                    if "500 internal server error" in error_msg.lower() and self.profile.is_external:
                        log(f"Attempt {attempt + 1} failed for {self.name} (external service error): {e}")
                        if self.profile.is_third_party_api:
                            log(f"{self.name}: This appears to be a temporary external API issue, will retry...")
                    else:
                        log(f"Attempt {attempt + 1} failed for {self.name}: {e}")
                    await self._close_session()
                    if attempt < retry_count - 1:
                        # Use longer delay for external service errors
                        delay = retry_delay * 2 if ("500 internal server error" in error_msg.lower() and self.profile.is_external) else retry_delay
                        await asyncio.sleep(delay)
            else:
                # All attempts failed
                if self.profile.is_external:
                    raise Exception(f"Failed to connect to external service '{self.name}' after {retry_count} attempts. This is likely a temporary issue with the external API.")
                else:
                    raise Exception(f"Failed to get tools after {retry_count} attempts")
            
            # Start health monitoring
            self._health = asyncio.create_task(self._health_loop())
            self.state = "running"
            self.healthy = True
            log(f"{self.name} ↑ ({self.transport})")
            
        except Exception as e:
            self.state = "errored"
            self.last_error = str(e)
            log(f"{self.name} failed to start: {e}")
            
            # Clean up failed startup
            await self._close_session()
            if self.proc:
                try:
                    log(f"Killing {self.name} (pid {self.pid}) after failed startup.")
                    self.proc.kill()
                    await asyncio.wait_for(self.proc.wait(), timeout=5)
                except Exception as kill_exc:
                    log(f"Kill failed for {self.name}: {kill_exc}")
                
                if self.pid:
                    try:
                        import signal
                        log(f"Trying os.kill on {self.name} (pid {self.pid}) after failed startup.")
                        os.kill(self.pid, signal.SIGKILL)
                    except Exception as oskill_exc:
                        log(f"os.kill failed for {self.name}: {oskill_exc}")
                
            self.proc, self.pid = None, None
            
            # Don't retry automatically - let supervisor handle retries
            log(f"{self.name} is now marked as errored.")

    async def _do_stop(self):
        if self.state == "stopped":
            return
        self.state = "stopping"
        
        # Cancel health check first
        if self._health:
            self._health.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health
        
        # Cancel output capture tasks
        if self._stdout_task:
            self._stdout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stdout_task
            self._stdout_task = None
            
        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
            
        if hasattr(self, '_log_monitor_task') and self._log_monitor_task:
            self._log_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._log_monitor_task
            self._log_monitor_task = None
        
        # Close session
        await self._close_session()
        
        # Terminate process
        if self.proc:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self.proc.kill()
                    await self.proc.wait()
                except ProcessLookupError:
                    pass  # Process already gone
            
            # Explicitly close pipes
            for pipe in [self.proc.stdin, self.proc.stdout, self.proc.stderr]:
                if pipe:
                    try:
                        # Check if pipe is already closed using a safer method
                        if hasattr(pipe, 'is_closing') and pipe.is_closing():
                            continue
                        pipe.close()
                        # Wait for pipe to actually close
                        if hasattr(pipe, 'wait_closed'):
                            await pipe.wait_closed()
                    except Exception:
                        pass
        
        self.proc, self.pid = None, None
        self.state          = "stopped"
        self.healthy        = False
        self.session_at     = 0.0  # Invalidate session timestamp
        
        log(f"{self.name} ↓")
        
        if self._runner:
            self._runner.cancel()

    async def _do_restart(self):
        """Properly restart the child without breaking the event loop."""
        log(f"{self.name}: Starting restart sequence")
        
        # Track restart time and count
        self.last_restart_time = time.time()
        self.restart_count += 1
        
        # First, do a controlled stop (without cancelling runner)
        await self._controlled_stop()
        
        # Small delay to ensure cleanup is complete
        await asyncio.sleep(0.5)
        
        # Check if the runner is still alive, recreate if needed
        if self._runner and self._runner.done():
            log(f"{self.name}: Creating new runner task for restart")
            self._runner = asyncio.create_task(self._loop())
        
        # Reset retry count for fresh start
        self.retries = 0
        
        # Now restart by sending start command to the queue
        await self._cmd_q.put(("start",))
        log(f"{self.name}: Restart command queued")

    async def _controlled_stop(self):
        """Stop the child without cancelling the main runner task."""
        if self.state == "stopped":
            return
        self.state = "stopping"
        
        # Cancel health check first
        if self._health:
            self._health.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health
            self._health = None
        
        # Cancel output capture tasks
        if self._stdout_task:
            self._stdout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stdout_task
            self._stdout_task = None
            
        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
            
        if hasattr(self, '_log_monitor_task') and self._log_monitor_task:
            self._log_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._log_monitor_task
            self._log_monitor_task = None
        
        # Close session
        await self._close_session()
        
        # Terminate process
        if self.proc:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self.proc.kill()
                    await self.proc.wait()
                except ProcessLookupError:
                    pass  # Process already gone
            
            # Explicitly close pipes
            for pipe in [self.proc.stdin, self.proc.stdout, self.proc.stderr]:
                if pipe:
                    try:
                        if hasattr(pipe, 'is_closing') and pipe.is_closing():
                            continue
                        pipe.close()
                        if hasattr(pipe, 'wait_closed'):
                            await pipe.wait_closed()
                    except Exception:
                        pass
        
        self.proc, self.pid = None, None
        self.state = "stopped"
        self.healthy = False
        self.session_at = 0.0
        
        log(f"{self.name} ↓ (controlled stop)")

    async def _spawn_if_needed(self):
        # For HTTP transport, no subprocess is needed
        if self.transport == "http":
            return
        
        # For STDIO transport, we don't create our own subprocess
        # The MCP stdio_client will create its own subprocess
        # We'll capture output differently in the session setup
        if self.transport == "stdio":
            return
            
        # For other transports, create subprocess with capture
        if self.proc and self.proc.returncode is None:
            return
        try:
            env = {**os.environ, **self.spec.get("env", {})}
            command_parts = shlex.split(self.spec["command"])
            args = self.spec.get("args", [])
            log(f"Spawning {self.name} with command: {command_parts} {args}")
            
            # --- Enhanced: Capture stdout and stderr with MCP server name prefixes ---
            import datetime
            def iso_now():
                return datetime.datetime.now().isoformat()
                    
            async def capture_output(stream, buffer, stream_name):
                try:
                    while True:
                        if not stream or stream.at_eof():
                            break
                        
                        line = await stream.readline()
                        if not line:
                            break
                            
                        decoded = line.decode('utf-8', errors='replace').rstrip('\n\r')
                        # Skip empty lines to reduce noise
                        if not decoded.strip():
                            continue
                            
                        entry = {"timestamp": iso_now(), "line": decoded}
                        buffer.append(entry)
                        if len(buffer) > self._output_buffer_limit:
                            del buffer[0:len(buffer)-self._output_buffer_limit]
                        self.last_output_renewal = entry["timestamp"]
                        
                        # Always prefix with MCP server name
                        if stream_name == 'stderr':
                            print(f"[{self.name}:err] {decoded}", flush=True)
                        else:
                            print(f"[{self.name}] {decoded}", flush=True)
                            
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log(f"{self.name} {stream_name} capture error: {e}")
            
            # Create subprocess with explicit PIPE redirection
            self.proc = await asyncio.create_subprocess_exec(
                *command_parts,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            self.pid = self.proc.pid
            self.started_at = time.time()
            
            # Start capture tasks immediately after process creation
            self._stdout_task = asyncio.create_task(capture_output(self.proc.stdout, self.stdout_buffer, 'stdout'))
            self._stderr_task = asyncio.create_task(capture_output(self.proc.stderr, self.stderr_buffer, 'stderr'))
            
        except Exception as e:
            log(f"Error spawning {self.name}: {e}")
            raise

    async def _ensure_session(self, force=False):
        """
        Create MCP session using the official SDK instead of custom implementation.
        This replaces all custom transport and session management code.
        """
        
        # STEP 1: Check if we can reuse the existing session
        if (not force and self.session
                and time.time() - self.session_at < SESSION_TTL
                and self.healthy):
            return  # Session is good, reuse it
                
        # STEP 2: Clean up any existing session
        await self._close_session()
        
        try:
            # STEP 3: Use SDK for transport and session creation
            if self.transport == "http":
                # Try without OAuth first, SDK will handle OAuth automatically if required
                oauth_auth = None
                
                try:
                    # Use SDK's HTTP clients - OAuth will be triggered automatically if needed
                    if "/sse" in self.spec["url"]:
                        # SSE transport via SDK
                        self._transport_context = sse_client(self.spec["url"], auth=oauth_auth)
                        read_stream, write_stream = await self._transport_context.__aenter__()
                    else:
                        # Streamable HTTP transport via SDK
                        self._transport_context = streamablehttp_client(self.spec["url"], auth=oauth_auth)
                        read_stream, write_stream, _ = await self._transport_context.__aenter__()
                
                except Exception as auth_error:
                    # Check if this is an OAuth authentication error
                    if "unauthorized" in str(auth_error).lower() or "401" in str(auth_error):
                        log(f"{self.name}: Server requires OAuth authentication, setting up...")
                        oauth_auth = self._setup_oauth_auth()
                        
                        # Retry with OAuth
                        if "/sse" in self.spec["url"]:
                            self._transport_context = sse_client(self.spec["url"], auth=oauth_auth)
                            read_stream, write_stream = await self._transport_context.__aenter__()
                        else:
                            self._transport_context = streamablehttp_client(self.spec["url"], auth=oauth_auth)
                            read_stream, write_stream, _ = await self._transport_context.__aenter__()
                    else:
                        # Not an auth error, re-raise
                        raise
                
                # Create session using SDK
                self._session_context = ClientSession(
                    read_stream,
                    write_stream,
                    client_info={"name": "Fractalic MCP Manager", "version": "0.3.0"},
                )
                self.session = await self._session_context.__aenter__()
                
            else:
                # STDIO transport via SDK
                server_params = StdioServerParameters(
                    command=self.spec["command"],
                    args=self.spec.get("args", []),
                    env=self.spec.get("env", {})
                )
                
                # Use SDK's stdio client instead of custom implementation
                self._transport_context = stdio_client(server_params)
                read_stream, write_stream = await self._transport_context.__aenter__()
                
                self._session_context = ClientSession(read_stream, write_stream)
                self.session = await self._session_context.__aenter__()
            
            # STEP 4: Initialize session using SDK
            init_timeout = self.profile.init_timeout
            log(f"{self.name}: Starting SDK-based MCP session initialization (timeout: {init_timeout}s)")
            await asyncio.wait_for(self.session.initialize(), timeout=init_timeout)
            log(f"{self.name}: SDK session initialization completed successfully")
            
            # STEP 5: Mark session as successfully established
            self.session_at = time.time()
            self.started_at = self.started_at or self.session_at
            self.healthy = True
            
        except Exception as e:
            # STEP 6: Clean up on any failure
            await self._close_session()
            error_str = str(e).lower()
            if "500 internal server error" in error_str and self.profile.is_external:
                error_msg = f"External service error (HTTP 500): {e}. This is likely a temporary issue with the external API."
            else:
                error_msg = f"SDK session initialization failed: {e}"
            
            log(f"{self.name}: {error_msg}")
            self.last_error = error_msg
            raise Exception(error_msg)

    async def _close_session(self):
        """
        Close MCP session using SDK's cleanup mechanisms.
        The SDK handles all transport and resource cleanup automatically.
        """
        if hasattr(self, '_session_context') and self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                log(f"{self.name}: Warning during SDK session cleanup: {e}")
            finally:
                self._session_context = None
                
        if hasattr(self, '_transport_context') and self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except Exception as e:
                log(f"{self.name}: Warning during SDK transport cleanup: {e}")
            finally:
                self._transport_context = None
                
        # Reset session state
        self.session = None
        self.healthy = False
        self.session_at = 0.0

    async def _health_loop(self):
        # Give newly started processes more time before first health check
        startup_delay = max(30, HEALTH_INT * 2)  # At least 30 seconds
        await asyncio.sleep(startup_delay)
        
        consecutive_failures = 0
        
        while True:
            try:
                await asyncio.sleep(HEALTH_INT)
                
                # Skip health check if already stopping/stopped
                if self.state in ["stopping", "stopped", "errored"]:
                    break
                
                # Skip health check if recently restarted (give it time to stabilize)
                if hasattr(self, 'restart_count') and self.restart_count > 0:
                    time_since_restart = time.time() - getattr(self, 'last_restart_time', 0)
                    if time_since_restart < 60:  # Wait 60 seconds after restart
                        log(f"{self.name}: Skipping health check, recently restarted ({time_since_restart:.1f}s ago)")
                        continue
                
                # Try lightweight health check first
                health_ok = False
                
                # For all server types, try session-based health check first
                if self.session:
                    try:
                        # If session exists, try a simple ping first
                        if hasattr(self.session, 'ping'):
                            await asyncio.wait_for(self.session.ping(), timeout=10)
                            health_ok = True
                        else:
                            # If no ping available, try cached tools with longer timeout
                            await asyncio.wait_for(self.tools(), timeout=15)
                            health_ok = True
                    except Exception as e:
                        log(f"{self.name}: Session-based health check failed: {e}")
                        health_ok = False
                        
                        # For stdio servers, also check if process is still running
                        if self.transport == "stdio" and self.proc and self.proc.returncode is not None:
                            log(f"{self.name}: Process has exited (returncode: {self.proc.returncode})")
                        elif self.transport == "stdio" and not self.proc:
                            log(f"{self.name}: No process handle available")
                else:
                    # No session, try to establish one
                    try:
                        await self._ensure_session()
                        health_ok = True
                    except Exception as e:
                        log(f"{self.name}: Failed to establish session: {e}")
                        health_ok = False
                
                if health_ok:
                    self.healthy = True
                    consecutive_failures = 0
                    self.health_failures = 0  # Reset global counter too
                else:
                    self.healthy = False
                    consecutive_failures += 1
                    self.health_failures += 1
                    
                    # Be more lenient - require more failures before restarting
                    if consecutive_failures >= 3:
                        log(f"{self.name} failed health check {consecutive_failures} times consecutively")
                        
                        # Only restart if we haven't restarted too recently
                        time_since_last_restart = time.time() - getattr(self, 'last_restart_time', 0)
                        if time_since_last_restart < 120:  # Don't restart more than once every 2 minutes
                            log(f"{self.name}: Skipping restart, last restart was {time_since_last_restart:.1f}s ago")
                            consecutive_failures = 0  # Reset to prevent immediate retry
                            continue
                        
                        # If we've had too many total failures, mark as errored
                        # Be more tolerant for external HTTP services which can have temporary outages
                        failure_limit = self.profile.health_failure_limit
                        if self.health_failures >= failure_limit:
                            log(f"{self.name} exceeded health failure limit ({self.health_failures}/{failure_limit}), marking as errored")
                            
                            # Cancel output capture tasks first
                            if self._stdout_task:
                                self._stdout_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await self._stdout_task
                                self._stdout_task = None
                                
                            if self._stderr_task:
                                self._stderr_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await self._stderr_task
                                self._stderr_task = None
                            
                            await self._close_session()
                            if self.proc:
                                try:
                                    log(f"Attempting graceful kill of {self.name} (pid {self.pid})")
                                    self.proc.kill()
                                    await self.proc.wait()
                                except Exception as kill_exc:
                                    log(f"Graceful kill failed for {self.name}: {kill_exc}")
                            self.proc, self.pid = None, None
                            self.state = "errored"
                            log(f"{self.name} is now marked as errored and will not be restarted.")
                            break
                        else:
                            log(f"{self.name} scheduling restart due to health failures")
                            await self._schedule_retry()
                            break
                    else:
                        log(f"{self.name}: Health check failed ({consecutive_failures}/3), will retry")
                
            except asyncio.CancelledError:
                log(f"{self.name}: Health check cancelled")
                break
            except Exception as e:
                log(f"{self.name}: Health check loop error: {e}")
                # Don't break on unexpected errors, just continue
                consecutive_failures += 1

    async def _schedule_retry(self):
        # Track restart time
        self.last_restart_time = time.time()
        
        # Cancel output capture tasks before retry
        if self._stdout_task:
            self._stdout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stdout_task
            self._stdout_task = None
            
        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
            
        await self._close_session()
        if self.proc:
            with contextlib.suppress(Exception):
                self.proc.kill()
                await self.proc.wait()
                # Close pipes explicitly
                for pipe in [self.proc.stdin, self.proc.stdout, self.proc.stderr]:
                    if pipe:
                        try:
                            # Check if pipe is already closed using a safer method
                            if hasattr(pipe, 'is_closing') and pipe.is_closing():
                                continue
                            pipe.close()
                            if hasattr(pipe, 'wait_closed'):
                                await pipe.wait_closed()
                        except Exception:
                            pass
                            
        # Use adaptive retry limit based on service characteristics
        max_retries = self.profile.max_retries
            
        if self.retries >= max_retries:
            self.state = "errored"
            log(f"{self.name} exceeded retries ({self.retries}/{max_retries}) → errored")
            return
            
        self.retries += 1
        backoff = min(BACKOFF_BASE ** self.retries, 60)  # Cap at 60 seconds
        self.state = "retrying"
        log(f"{self.name} retrying in {backoff}s …")
        await asyncio.sleep(backoff)
        self.restart_count += 1           # Increment restart count before retry
        await self._do_start()

    async def list_tools(self):
        try:
            await self._ensure_session()
            return await asyncio.wait_for(self.session.list_tools(), TIMEOUT_INITIAL)
        except Exception as e:
            error_msg = f"Failed to list tools: {str(e) if str(e) else 'Unknown error'}"
            self.last_error = error_msg
            log(f"{self.name}: {error_msg}")
            # Invalidate session on error
            self.session_at = 0.0
            self.healthy = False
            raise

    async def tools(self):
        """Get tools with caching and rate limiting to reduce repeated list_tools() calls."""
        try:
            # Apply rate limiting for high-activity services
            current_time = time.time()
            time_since_last_request = current_time - self.last_tool_request
            if time_since_last_request < self.profile.tool_request_cooldown:
                # Return cached tools if we're in cooldown period
                if hasattr(self, '_cached_tools') and self._cached_tools:
                    return self._cached_tools
                else:
                    # If no cache but in cooldown, wait for cooldown to expire
                    sleep_time = self.profile.tool_request_cooldown - time_since_last_request
                    log(f"{self.name}: Rate limiting tool request, waiting {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
            
            self.last_tool_request = time.time()
            
            # Check if cached tools are available and recent (within 5 minutes for better resilience)
            cache_timeout = 300  # 5 minutes to reduce repeated calls during health check cycles
            if (hasattr(self, '_cached_tools') and self._cached_tools and 
                time.time() - getattr(self, '_tools_cache_time', 0) < cache_timeout):
                return self._cached_tools
            
            # Cache expired or doesn't exist, refresh it
            await self._ensure_session()
            tl = await asyncio.wait_for(self.session.list_tools(), TIMEOUT_INITIAL)
            tools_list = [tool_to_obj(t) for t in tl.tools]
            
            # Apply Vertex AI schema sanitization only if enabled
            if ENABLE_SCHEMA_SANITIZATION:
                sanitized_tools = [sanitize_tool_schema(tool) for tool in tools_list]
            else:
                sanitized_tools = tools_list  # No sanitization
                
            # Cache the results
            self._cached_tools = sanitized_tools
            self._tools_cache_time = time.time()
            
            return sanitized_tools
            
        except Exception as e:
            error_msg = f"Failed to get tools: {str(e) if str(e) else 'Unknown error'}"
            self.last_error = error_msg
            log(f"{self.name}: {error_msg}")
            # Clear temporary reset flag on real errors to ensure cache gets invalidated
            if hasattr(self, '_temporary_session_reset'):
                delattr(self, '_temporary_session_reset')
            # Invalidate session on error
            self.session_at = 0.0
            self.healthy = False
            raise

    async def call_tool(self, tool: str, args: dict):
        try:
            await self._ensure_session()
            
            log(f"{self.name}: Calling tool '{tool}' with timeout {TIMEOUT_INITIAL}s")
            result = await asyncio.wait_for(
                self.session.call_tool(tool, args), TIMEOUT_INITIAL)
            return result
        except asyncio.TimeoutError as e:
            error_msg = f"Tool '{tool}' timed out after {TIMEOUT_INITIAL}s"
            log(f"{self.name}: {error_msg}")
            self.last_error = error_msg
            # Invalidate session on timeout
            self.session_at = 0.0
            self.healthy = False
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Tool '{tool}' failed: {str(e) if str(e) else 'Unknown error'}"
            log(f"{self.name}: {error_msg}")
            self.last_error = error_msg
            # Invalidate session on error
            self.session_at = 0.0
            self.healthy = False
            raise
    
    def info(self):
        return {
            "state":      self.state,
            "pid":        self.pid,
            "transport":  self.transport,
            "retries":    self.retries,
            "uptime":     round(time.time() - self.started_at, 1) if self.started_at else None,
            "healthy":    self.healthy,    # Expose health status
            "restarts":   self.restart_count,  # Expose restart count
            "last_error": self.last_error,     # Expose last error if any
            # --- New fields ---
            "stdout":     self.stdout_buffer[-50:],  # Last 50 lines, each with timestamp
            "stderr":     self.stderr_buffer[-50:],  # Last 50 lines, each with timestamp
            "last_output_renewal": self.last_output_renewal,
        }

    async def get_tools_info(self):
        """
        Return tool count and token count of the schema for this child/server.
        Uses LiteLLM token counting when USE_LITELLM_TOKEN_COUNT=True (aligned with Fractalic execution).
        """
        if self.state != "running":
            return {"tool_count": 0, "token_count": 0, "tools_error": f"MCP state is {self.state}"}
        try:
            # Use cached tools() method instead of direct list_tools()
            cached_tools = await self.tools()
            # Convert to the format expected by get_tools_info
            tools_list = cached_tools if isinstance(cached_tools, list) else []
            if tools_list == self._last_tools_list:
                return {
                    "tool_count": len(self._last_tools_list),
                    "token_count": self._last_token_count
                }
            
            # Calculate token count based on configuration
            if USE_LITELLM_TOKEN_COUNT and LITELLM_AVAILABLE:
                try:
                    # Convert MCP tools to OpenAI format for LiteLLM
                    openai_tools = []
                    for tool in tools_list:
                        openai_tool = {
                            "type": "function",
                            "function": {
                                "name": tool.get("name", "unknown"),
                                "description": tool.get("description", ""),
                                "parameters": tool.get("inputSchema", {})
                            }
                        }
                        openai_tools.append(openai_tool)
                    
                    # Use LiteLLM to count tokens (uses gpt-4 as reference model)
                    token_count = litellm.token_counter(model="gpt-4", messages=[], tools=openai_tools)
                    
                except Exception as e:
                    # Fallback to tiktoken if LiteLLM fails
                    schema_json = json.dumps(tools_list)
                    token_count = len(TOKENIZER.encode(schema_json))
            else:
                # Direct tiktoken approach
                schema_json = json.dumps(tools_list)
                token_count = len(TOKENIZER.encode(schema_json))
            
            self._last_tools_list = tools_list
            self._last_token_count = token_count
            self._last_schema_json = json.dumps(tools_list)
            return {"tool_count": len(tools_list), "token_count": token_count}
        except Exception as e:
            return {"tool_count": 0, "token_count": 0, "tools_error": str(e)}

# ==================================================================== Supervisor
class Supervisor:
    def __init__(self, file: Path = CONF_PATH):
        try:
            if file.exists():
                cfg = json.loads(file.read_text())
            else:
                # Use default empty config if file doesn't exist
                log(f"Config file {file} not found, using default empty configuration")
                cfg = {"mcpServers": {}}
        except Exception as e:
            log(f"Error reading config file {file}: {e}, using default empty configuration")
            cfg = {"mcpServers": {}}
        
        self.children = {n: Child(n, spec) for n, spec in cfg["mcpServers"].items()}

    async def start(self, tgt):
        if tgt == "all":
            # Launch all children as background tasks so API can start even if some fail
            startup_tasks = []
            for c in self.children.values():
                task = asyncio.create_task(c.start())
                startup_tasks.append(task)
            
            # Don't wait for all to complete, just fire and forget
            # The API should be available even if some servers fail to start
            log(f"Started {len(startup_tasks)} MCP servers in background")
            
        else:
            c = self.children.get(tgt)
            if not c: raise web.HTTPNotFound(text=f"{tgt} unknown")
            await c.start()

    async def stop(self, tgt):
        if tgt == "all":
            # Stop all children with proper cleanup
            cleanup_tasks = []
            for c in self.children.values():
                cleanup_tasks.append(c.cleanup())
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        else:
            c = self.children.get(tgt)
            if not c: 
                raise web.HTTPNotFound(text=f"{tgt} unknown")
            await c.cleanup()
    
    async def restart(self, tgt):
        """Restart an MCP server by stopping it and starting it again."""
        if tgt == "all":
            # Restart all children
            for c in self.children.values():
                await c._do_restart()
        else:
            c = self.children.get(tgt)
            if not c:
                raise web.HTTPNotFound(text=f"{tgt} unknown")
            await c._do_restart()
    
    async def stop_local_only(self):
        """Stop only local stdio servers, skip remote HTTP servers."""
        local_children = [c for c in self.children.values() if c.transport == "stdio"]
        if local_children:
            log(f"Stopping {len(local_children)} local servers: {[c.name for c in local_children]}")
            cleanup_tasks = [c.cleanup() for c in local_children]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        else:
            log("No local servers to stop")

    async def status(self):
        # Gather base info
        base = {n: c.info() for n, c in self.children.items()}
        # Gather tools info (tool count, token count) for each child
        tools_info = {}
        for n, c in self.children.items():
            tools_info[n] = await c.get_tools_info()
        # Merge tools info into base info
        for n in base:
            base[n].update(tools_info[n])
        return base

    async def tools(self):
        out = {}
        for n, c in self.children.items():
            if c.state != "running":
                # Provide more specific error messages
                error_detail = c.last_error or f"MCP state is {c.state}"
                out[n] = {"error": error_detail, "tools": []}
                continue
            try:
                # Use cached tools info if available to reduce repeated list_tools calls
                tools_info = await c.get_tools_info()
                if "tools_error" in tools_info:
                    out[n] = {"error": tools_info["tools_error"], "tools": []}
                    continue
                    
                # Always use cached tools() method to maintain consistent caching behavior
                sanitized_tools = await c.tools()
                out[n] = {"tools": sanitized_tools}
            except Exception as e:
                error_msg = str(e) if str(e) else f"Unknown error from {n}"
                log(f"Error getting tools from {n}: {error_msg}")
                out[n] = {"error": error_msg, "tools": []}
                # Store the error for future reference
                c.last_error = error_msg
        return out

    async def call_tool(self, name: str, args: Dict[str, Any]):
        for c in self.children.values():
            if c.state == "errored":
                continue  # Skip errored children
            try:
                # Use cached tools() method instead of direct list_tools calls
                cached_tools = await c.tools()
                if any(tool.get('name') == name for tool in cached_tools):
                    return await c.call_tool(name, args)
            except Exception:
                pass
        raise web.HTTPNotFound(text=f"tool {name!r} not found")

    async def _each(self, meth, tgt):
        if tgt == "all":
            await asyncio.gather(*(getattr(c, meth)() for c in self.children.values()))
        else:
            c = self.children.get(tgt)
            if not c: raise web.HTTPNotFound(text=f"{tgt} unknown")
            await getattr(c, meth)()

# ==================================================================== aiohttp façade
def build_app(sup: Supervisor, stop_event: asyncio.Event):
    app = web.Application()
    
    # Setup CORS
    cors = cors_setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        ),
        "http://localhost:3000": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add routes
    app.router.add_get ("/status",      lambda r: _json(r, sup.status()))
    app.router.add_get ("/tools",       lambda r: _await_json(r, sup.tools()))
    app.router.add_get ("/list_tools",  lambda r: _await_json(r, sup.tools()))
    app.router.add_post("/start/{n}",   lambda r: _mut(r, sup, "start"))
    app.router.add_post("/stop/{n}",    lambda r: _mut(r, sup, "stop"))
    app.router.add_post("/restart/{n}", lambda r: _mut(r, sup, "restart"))
    app.router.add_post("/call_tool",   lambda r: _call(r, sup))
    app.router.add_post("/add_server",  lambda r: _add_server(r, sup))
    app.router.add_post("/delete_server", lambda r: _delete_server(r, sup))
    app.router.add_post("/kill",        lambda r: _kill(r, sup, stop_event))
    
    # Configure CORS for all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def _json(_, coro):
    return web.json_response(await coro, dumps=lambda obj: json.dumps(obj, cls=MCPEncoder))

async def _await_json(_, coro):
    return web.json_response(await coro, dumps=lambda obj: json.dumps(obj, cls=MCPEncoder))

async def _mut(req, sup, act):
    await getattr(sup, act)(req.match_info["n"])
    return web.json_response(await sup.status(), dumps=lambda obj: json.dumps(obj, cls=MCPEncoder))

async def _call(req, sup):
    body = await req.json()
    res  = await sup.call_tool(body["name"], body.get("arguments", {}))
    return web.json_response(res, dumps=lambda obj: json.dumps(obj, cls=MCPEncoder))

async def _add_server(req, sup: Supervisor):
    """Add a new MCP server configuration"""
    try:
        body = await req.json()
        
        # Handle case where frontend sends JSON as a string in jsonConfig field
        if "jsonConfig" in body and isinstance(body["jsonConfig"], str):
            try:
                # Parse the JSON string to get the actual configuration
                body = json.loads(body["jsonConfig"])
            except json.JSONDecodeError as e:
                return web.json_response(
                    {"success": False, "error": "Fractalic MCP manager: Invalid JSON in jsonConfig field"}, 
                    status=400,
                    dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
                )
        
        # Handle different JSON formats from frontend
        if "mcpServers" in body and isinstance(body["mcpServers"], dict):
            # Format: {"mcpServers": {"server-name": {...}}}
            servers = body["mcpServers"]
            if len(servers) != 1:
                return web.json_response(
                    {"success": False, "error": "Fractalic MCP manager: When using mcpServers format, exactly one server must be provided"}, 
                    status=400,
                    dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
                )
            
            # Extract the server name and config from the nested structure
            server_name, server_config = next(iter(servers.items()))
            name = server_name
            # For nested format, the entire server_config becomes our config
            # and we need to extract URL if it exists, or build it from config
            if "url" in server_config:
                url = server_config["url"]
                config = {k: v for k, v in server_config.items() if k != "url"}
            else:
                # For non-HTTP servers (stdio), there's no URL
                config = server_config
                url = None
        elif "mcp" in body and isinstance(body["mcp"], dict) and "servers" in body["mcp"] and isinstance(body["mcp"]["servers"], dict):
            # Format: {"mcp": {"servers": {"server-name": {...}}}}
            servers = body["mcp"]["servers"]
            if len(servers) != 1:
                return web.json_response(
                    {"success": False, "error": "Fractalic MCP manager: When using mcp.servers format, exactly one server must be provided"}, 
                    status=400,
                    dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
                )
            
            # Extract the server name and config from the nested structure
            server_name, server_config = next(iter(servers.items()))
            name = server_name
            # For nested format, the entire server_config becomes our config
            # and we need to extract URL if it exists, or build it from config
            if "url" in server_config:
                url = server_config["url"]
                config = {k: v for k, v in server_config.items() if k != "url"}
            else:
                # For non-HTTP servers (stdio), there's no URL
                config = server_config
                url = None
        else:
            # Standard format with name, url, and config at top level
            name = body.get("name")
            url = body.get("url")
            config = body.get("config", {})
            
            # Validate required fields for standard format
            if not name:
                return web.json_response(
                    {"success": False, "error": "Fractalic MCP manager: Server name is required"}, 
                    status=400,
                    dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
                )
            
            # For standard format, URL is required (mcpServers format allows stdio servers without URL)
            if not url:
                return web.json_response(
                    {"success": False, "error": "Fractalic MCP manager: Server URL is required"}, 
                    status=400,
                    dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
                )
        
        # Read current configuration
        try:
            if CONF_PATH.exists():
                current_config = json.loads(CONF_PATH.read_text())
            else:
                # Create default config if file doesn't exist
                current_config = {"mcpServers": {}}
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Fractalic MCP manager: Failed to read server configuration: {str(e)}"}, 
                status=500,
                dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
            )
        
        # Check for duplicate names
        if name in current_config.get("mcpServers", {}):
            return web.json_response(
                {"success": False, "error": f"Fractalic MCP manager: Server with name '{name}' already exists"}, 
                status=409,
                dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
            )
        
        # Prepare server configuration
        server_config = {}
        
        # Add URL if provided (for HTTP servers)
        if url:
            server_config["url"] = url
        
        # Add other configuration parameters
        if config:
            # Handle different config types
            config_type = config.get("type", "manual")
            transport = config.get("transport", "http" if url else "stdio")
            auth = config.get("auth")
            capabilities = config.get("capabilities", [])
            
            # For stdio servers, we need command and args
            if transport == "stdio" or not url:
                if "command" in config:
                    server_config["command"] = config["command"]
                if "args" in config:
                    server_config["args"] = config["args"]
                if "env" in config:
                    server_config["env"] = config["env"]
            
            # Add transport if specified
            if transport and transport != "http":
                server_config["transport"] = transport
            
            # Add auth if provided
            if auth:
                server_config["auth"] = auth
            
            # Add any other config parameters that aren't handled above
            for key, value in config.items():
                if key not in ["type", "transport", "auth", "capabilities", "command", "args", "env"]:
                    server_config[key] = value
        
        # Add server to configuration
        if "mcpServers" not in current_config:
            current_config["mcpServers"] = {}
        
        current_config["mcpServers"][name] = server_config
        
        # Save updated configuration
        try:
            CONF_PATH.write_text(json.dumps(current_config, indent=2))
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Fractalic MCP manager: Failed to save server configuration: {str(e)}"}, 
                status=500,
                dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
            )
        
        # Log the operation
        log(f"Added new MCP server '{name}' with config: {server_config}")
        
        # DYNAMIC SERVER LAUNCH: Create and start the new server immediately
        try:
            # Create new Child object for the server
            new_child = Child(name, server_config)
            
            # Add to supervisor's children dictionary
            sup.children[name] = new_child
            
            # Start the server in background (non-blocking)
            asyncio.create_task(new_child.start())
            
            log(f"Fractalic MCP manager: Launched new server '{name}' dynamically")
            launch_status = "launched"
            
        except Exception as launch_error:
            # If dynamic launch fails, log but still return success since config was saved
            log(f"Fractalic MCP manager: Failed to launch '{name}' dynamically: {launch_error}")
            launch_status = "config_saved_only"
        
        # Return success response with launch status
        response_data = {
            "success": True,
            "message": "Fractalic MCP manager: Server added successfully",
            "launch_status": launch_status,
            "server": {
                "name": name,
                "url": url,
                "config": server_config,
                "added_at": datetime.datetime.now().isoformat()
            }
        }
        
        return web.json_response(
            response_data, 
            status=201,
            dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
        )
        
    except json.JSONDecodeError:
        return web.json_response(
            {"success": False, "error": "Fractalic MCP manager: Invalid JSON in request body"}, 
            status=400,
            dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
        )
    except Exception as e:
        log(f"Error adding server: {str(e)}")
        return web.json_response(
            {"success": False, "error": f"Fractalic MCP manager: Internal server error: {str(e)}"}, 
            status=500,
            dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
        )

async def _delete_server(req, sup: Supervisor):
    """Delete an MCP server configuration"""
    try:
        # Get server name from request body
        body = await req.json()
        name = body.get("name")
        
        if not name:
            return web.json_response(
                {"success": False, "error": "Fractalic MCP manager: Server name is required"}, 
                status=400,
                dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
            )
        
        # Read current configuration
        try:
            if CONF_PATH.exists():
                current_config = json.loads(CONF_PATH.read_text())
            else:
                # Create default config if file doesn't exist
                current_config = {"mcpServers": {}}
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Fractalic MCP manager: Failed to read server configuration: {str(e)}"}, 
                status=500,
                dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
            )
        
        # Check if server exists
        if name not in current_config.get("mcpServers", {}):
            return web.json_response(
                {"success": False, "error": f"Fractalic MCP manager: Server '{name}' not found"}, 
                status=404,
                dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
            )
        
        # Stop the server if it's running
        try:
            if name in sup.children:
                child = sup.children[name]
                await child.cleanup()  # Stop and clean up the server
                log(f"Stopped running server '{name}' before deletion")
        except Exception as e:
            log(f"Warning: Failed to stop server '{name}' before deletion: {e}")
            # Continue with deletion even if stopping fails
        
        # Remove server from configuration
        server_config = current_config["mcpServers"].pop(name)
        
        # Save updated configuration
        try:
            CONF_PATH.write_text(json.dumps(current_config, indent=2))
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Fractalic MCP manager: Failed to save server configuration: {str(e)}"}, 
                status=500,
                dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
            )
        
        # Remove from supervisor's children if present
        if name in sup.children:
            del sup.children[name]
        
        # Log the operation
        log(f"Deleted MCP server '{name}' with config: {server_config}")
        
        # Return success response
        response_data = {
            "success": True,
            "message": "Fractalic MCP manager: Server deleted successfully",
            "server": {
                "name": name,
                "config": server_config,
                "deleted_at": datetime.datetime.now().isoformat()
            }
        }
        
        return web.json_response(
            response_data, 
            status=200,
            dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
        )
        
    except Exception as e:
        log(f"Error deleting server: {str(e)}")
        return web.json_response(
            {"success": False, "error": f"Fractalic MCP manager: Internal server error: {str(e)}"}, 
            status=500,
            dumps=lambda obj: json.dumps(obj, cls=MCPEncoder)
        )

async def _kill(req, sup: Supervisor, stop_ev: asyncio.Event):
    # Respond immediately, then trigger shutdown asynchronously
    async def delayed_shutdown():
        # Small delay to ensure response is sent
        await asyncio.sleep(0.1)
        # 1) stop only local stdio servers (faster, safer)
        await sup.stop_local_only()
        # 2) tell the main loop in run_serve() to exit
        stop_ev.set()
    
    # Start shutdown in background
    asyncio.create_task(delayed_shutdown())
    
    # Return response immediately
    return web.json_response({"status": "Fractalic MCP manager: shutting-down"}, dumps=lambda obj: json.dumps(obj, cls=MCPEncoder))

# ==================================================================== runners
async def run_serve(port: int):
    # Set up global exception handler for unhandled task exceptions
    def exception_handler(loop, context):
        exception = context.get('exception')
        if exception:
            error_msg = str(exception)
            # Suppress common async context errors that don't affect functionality
            if ("cancel scope" in error_msg or "different task" in error_msg or 
                "Attempted to exit cancel scope" in error_msg or
                "unhandled errors in a TaskGroup" in error_msg):
                # These are common async context errors during shutdown - silently ignore
                return
        # Log other unhandled exceptions normally
        loop.default_exception_handler(context)
    
    # Install the exception handler
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(exception_handler)
    
    sup = Supervisor(); await sup.start("all")
    stop_ev = asyncio.Event()
    runner = web.AppRunner(build_app(sup, stop_ev)); await runner.setup()
    site   = web.TCPSite(runner, "127.0.0.1", port); await site.start()
    log(f"API http://127.0.0.1:{port}  – Ctrl-C to quit")

    def _stop(*_): stop_ev.set()

    # cross-platform signal handling
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _stop)

    await stop_ev.wait()
    log("shutting down …")
    
    # Improved shutdown sequence with better signal handling
    try:
        # Force immediate shutdown without waiting for graceful cleanup
        await asyncio.wait_for(sup.stop_local_only(), timeout=3.0)
        
        # Clean up the web runner
        await asyncio.wait_for(runner.cleanup(), timeout=2.0)
        
        # Minimal cleanup time
        await asyncio.sleep(0.1)
        gc.collect()
        
    except asyncio.TimeoutError:
        log("Warning: Shutdown timed out, forcing exit")
        # Force exit immediately
        sys.exit(0)
    except Exception as e:
        log(f"Warning: Error during shutdown: {e}")
    
    log("Shutdown complete")

async def client_call(port: int, verb: str, tgt: Optional[str] = None):
    url = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as s:
        try:
            if verb in ("status", "tools"):
                r = await s.get(f"{url}/{verb}"); print(json.dumps(await r.json(), indent=2))
            elif verb in ("start", "stop", "restart"):
                await s.post(f"{url}/{verb}/{tgt}")
            elif verb == "kill":
                await s.post(f"{url}/kill")
            else:
                raise SystemExit(f"unknown verb {verb}")
        except aiohttp.ClientConnectorError:
            print("Error: Could not connect to server (is it running?)")

# ==================================================================== CLI
def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve"); sub.add_parser("status"); sub.add_parser("tools")
    sub.add_parser("kill")
    for v in ("start", "stop", "restart"):
        sc = sub.add_parser(v); sc.add_argument("target")
    
    # Tools command
    dump = sub.add_parser("tools-dump")
    dump.add_argument("output_file", help="Path to output JSON file")
    dump.add_argument("target", nargs="?", default="all", help="Optional: server name (default: all)")
    
    # OAuth management commands
    oauth_list = sub.add_parser("oauth-list", help="List OAuth tokens and their status")
    oauth_list.add_argument("--storage-dir", help="OAuth storage directory (optional)")
    
    oauth_migrate = sub.add_parser("oauth-migrate", help="Migrate OAuth tokens for deployment")
    oauth_migrate.add_argument("source", help="Source directory containing OAuth tokens")
    oauth_migrate.add_argument("target", help="Target directory for deployment")
    
    oauth_setup = sub.add_parser("oauth-setup", help="Setup OAuth for all HTTP servers")
    oauth_setup.add_argument("--server", help="Setup OAuth for specific server only")
    
    return p

def main():
    a = _parser().parse_args()
    if a.cmd == "serve":
        asyncio.run(run_serve(a.port))
    elif a.cmd == "tools-dump":
        # Dump tools schema to file using HTTP API
        async def dump_tools():
            url = f"http://127.0.0.1:{a.port}"
            async with aiohttp.ClientSession() as s:
                r = await s.get(f"{url}/tools")
                all_tools = await r.json()
                if a.target == "all":
                    tools = all_tools
                else:
                    tools = {a.target: all_tools.get(a.target, {"error": "Not found", "tools": []})}
                with open(a.output_file, "w", encoding="utf-8") as f:
                    json.dump(tools, f, indent=2)
                print(f"Tools schema dumped to {a.output_file}")
        asyncio.run(dump_tools())
    elif a.cmd == "oauth-list":
        # List OAuth tokens
        tokens = list_oauth_tokens(a.storage_dir)
        if not tokens:
            print("No OAuth tokens found.")
        else:
            print(f"OAuth tokens in {a.storage_dir or OAUTH_STORAGE_DIR}:")
            for server_name, info in tokens.items():
                if "error" in info:
                    print(f"  ❌ {server_name}: {info['error']}")
                else:
                    status = "✅ Valid" if not info["is_expired"] else "⚠️ Expired"
                    print(f"  {status} {server_name}:")
                    print(f"     Access Token: {'✓' if info['has_access_token'] else '✗'}")
                    print(f"     Refresh Token: {'✓' if info['has_refresh_token'] else '✗'}")
                    if info["expires_at"]:
                        expires = datetime.datetime.fromtimestamp(info["expires_at"])
                        print(f"     Expires: {expires.isoformat()}")
                    print(f"     Last Modified: {info['last_modified']}")
    elif a.cmd == "oauth-migrate":
        # Migrate OAuth tokens
        success = migrate_oauth_tokens(a.source, a.target)
        if success:
            print(f"✅ OAuth tokens migrated from {a.source} to {a.target}")
        else:
            print(f"❌ OAuth token migration failed")
            sys.exit(1)
    elif a.cmd == "oauth-setup":
        # Setup OAuth for servers
        print("🔐 OAuth Setup for MCP Servers")
        print("This will trigger OAuth authentication for HTTP servers...")
        print("⚠️  Make sure no other Fractalic instance is running!")
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
            
        with open(config_path) as f:
            config = json.load(f)
            
        # Find HTTP servers
        http_servers = []
        for name, spec in config.get("mcpServers", {}).items():
            if "url" in spec:  # HTTP server
                if a.server is None or name == a.server:
                    http_servers.append((name, spec))
                    
        if not http_servers:
            if a.server:
                print(f"❌ Server '{a.server}' not found or not an HTTP server")
            else:
                print("ℹ️  No HTTP servers found in configuration")
            sys.exit(0)
            
        print(f"Found {len(http_servers)} HTTP server(s) for OAuth setup:")
        for name, spec in http_servers:
            print(f"  - {name}: {spec['url']}")
        print()
        
        # Setup OAuth for each server
        async def setup_oauth():
            for name, spec in http_servers:
                print(f"🔑 Setting up OAuth for {name}...")
                try:
                    child = Child(name, spec)
                    # Try to connect - this will trigger OAuth if needed
                    await child._ensure_session(force=True)
                    print(f"✅ {name}: OAuth setup completed")
                    await child.cleanup()
                except Exception as e:
                    print(f"❌ {name}: OAuth setup failed: {e}")
                print()
                
        asyncio.run(setup_oauth())
        print("🎉 OAuth setup completed!")
    else:
        asyncio.run(client_call(a.port, a.cmd, getattr(a, "target", None)))

if __name__ == "__main__":
    main()
