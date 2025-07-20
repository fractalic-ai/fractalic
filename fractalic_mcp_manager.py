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

import errno
import tiktoken

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
        
        self.proc        = None
        self.pid         = None
        self.session     : Optional[ClientSession] = None
        self.session_at  = 0.0
        self._exit_stack = None
        self._health     = None
        self.retries     = 0
        self.started_at  = None
        self._cmd_q      : asyncio.Queue = asyncio.Queue()
        self._runner     = asyncio.create_task(self._loop())
        self.healthy     = False          # Track liveness separately from readiness
        self.restart_count = 0            # Track total number of restarts
        self.last_error   = None          # Store last error message
        self.last_tool_request = 0.0      # Track last tool request time for rate limiting
        self.health_failures = 0          # Track consecutive health check failures
        self.last_restart_time = 0        # Track when last restart occurred
        # --- New fields for output capture ---
        self.stdout_buffer = []  # List of dicts: {"timestamp": str, "line": str}
        self.stderr_buffer = []  # List of dicts: {"timestamp": str, "line": str}
        self.last_output_renewal = None  # ISO8601 string of last output change
        self._output_buffer_limit = 1000  # Max lines to keep per buffer
        # --- Output capture tasks ---
        self._stdout_task = None
        self._stderr_task = None
        # --- Caching for tools_info ---
        self._last_tools_list = None
        self._last_token_count = None
        self._last_schema_json = None
        # --- Tools caching ---
        self._cached_tools = None
        self._tools_cache_time = 0

        # --- Service profile for adaptive settings ---
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

    async def _setup_stdio_monitoring(self, transport):
        """Setup monitoring for stdio subprocess stderr output."""
        try:
            # The transport is a tuple (read_stream, write_stream)
            read_stream, write_stream = transport
            
            # Try to access the underlying process through various methods
            process = None
            
            # Method 1: Try to access via stream internals
            for stream in [read_stream, write_stream]:
                if hasattr(stream, '_transport'):
                    transport_obj = stream._transport
                    if hasattr(transport_obj, 'get_extra_info'):
                        try:
                            subprocess_obj = transport_obj.get_extra_info('subprocess')
                            if subprocess_obj:
                                process = subprocess_obj
                                break
                        except:
                            pass
                    
                    # Try other common process attributes
                    for attr in ['_process', '_protocol']:
                        if hasattr(transport_obj, attr):
                            obj = getattr(transport_obj, attr)
                            if hasattr(obj, 'pid'):  # Looks like a process
                                process = obj
                                break
                            elif hasattr(obj, '_process'):
                                process = obj._process
                                break
                    
                    if process:
                        break
            
            if process and hasattr(process, 'pid'):
                self.proc = process
                self.pid = process.pid
                log(f"{self.name}: Found subprocess PID {self.pid}")
                
                # If the process has stderr available, monitor it
                if hasattr(process, 'stderr') and process.stderr:
                    log(f"{self.name}: Starting stderr monitoring")
                    self._stderr_task = asyncio.create_task(self._monitor_stderr(process.stderr))
                else:
                    log(f"{self.name}: No stderr available for monitoring")
            else:
                log(f"{self.name}: Could not access subprocess for stderr monitoring")
                
        except Exception as e:
            log(f"{self.name}: Failed to setup stdio monitoring: {e}")

    async def _create_stderr_wrapper(self, command: str, args: list) -> str:
        """Create a wrapper script that captures stderr for a stdio server."""
        try:
            # Create wrapper script path
            wrapper_path = f"/tmp/mcp_{self.name}_wrapper.sh"
            log_file_path = f"/tmp/mcp_{self.name}_stderr.log"
            
            # Build the command with proper escaping
            escaped_args = [shlex.quote(arg) for arg in args]
            full_command = f"{shlex.quote(command)} {' '.join(escaped_args)}"
            
            # Create wrapper script content
            wrapper_content = f"""#!/bin/bash

# Auto-generated stderr capture wrapper for MCP server: {self.name}
SERVER_NAME="{self.name}"
LOG_FILE="{log_file_path}"

# Clean up old log file
rm -f "$LOG_FILE"

# Write initial marker
echo "[$SERVER_NAME] Starting MCP server..." >> "$LOG_FILE"

# Execute the actual server with stderr redirected to log file only
# We preserve stdin/stdout for MCP protocol and only redirect stderr
exec {full_command} 2>> "$LOG_FILE"
"""

            # Write wrapper script
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_content)
                
            # Make executable
            os.chmod(wrapper_path, 0o755)
            
            log(f"{self.name}: Created stderr wrapper: {wrapper_path}")
            return wrapper_path
            
        except Exception as e:
            log(f"{self.name}: Failed to create stderr wrapper: {e}")
            # Fall back to original command
            return command

    async def _monitor_stderr(self, stderr_stream):
        """Monitor stderr stream and capture output."""
        try:
            buffer = b""
            while True:
                try:
                    # Read from stderr stream
                    chunk = await stderr_stream.read(1024)
                    if not chunk:
                        break  # EOF - process ended
                    
                    buffer += chunk
                    
                    # Process complete lines
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        decoded = line.decode('utf-8', errors='replace').rstrip()
                        
                        if decoded:
                            # Create prefixed line for both console and buffer
                            prefixed_line = f"[{self.name}:stderr] {decoded}"
                            print(prefixed_line, flush=True)
                            
                            # Add to stderr buffer for UI
                            entry = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "line": prefixed_line
                            }
                            self.stderr_buffer.append(entry)
                            if len(self.stderr_buffer) > self._output_buffer_limit:
                                del self.stderr_buffer[0:len(self.stderr_buffer) - self._output_buffer_limit]
                            self.last_output_renewal = entry["timestamp"]
                            
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log(f"{self.name}: Error reading stderr: {e}")
                    break
                    
        except Exception as e:
            log(f"{self.name}: stderr monitoring error: {e}")

    async def _setup_log_monitoring(self):
        """Setup monitoring for stderr log file created by wrapper script."""
        try:
            # Only monitor stdio servers (HTTP servers don't need stderr capture)
            if self.transport != "stdio":
                return
                
            log_file_path = f"/tmp/mcp_{self.name}_stderr.log"
            
            # Only monitor if log file monitoring is not already active
            if not hasattr(self, '_log_monitor_task') or self._log_monitor_task is None:
                self._log_monitor_task = asyncio.create_task(self._monitor_log_file(log_file_path))
                log(f"{self.name}: Started log file monitoring: {log_file_path}")
        except Exception as e:
            log(f"{self.name}: Failed to setup log monitoring: {e}")

    async def _monitor_log_file(self, log_file_path):
        """Monitor a log file for new stderr content."""
        try:
            last_position = 0
            
            while True:
                try:
                    # Check if file exists and read new content
                    if os.path.exists(log_file_path):
                        with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                            f.seek(last_position)
                            new_content = f.read()
                            last_position = f.tell()
                            
                            if new_content:
                                # Process new lines
                                lines = new_content.splitlines()
                                for line in lines:
                                    if line.strip():
                                        # Add to stderr buffer
                                        entry = {
                                            "timestamp": datetime.datetime.now().isoformat(),
                                            "line": line  # Line already has server prefix from wrapper
                                        }
                                        self.stderr_buffer.append(entry)
                                        if len(self.stderr_buffer) > self._output_buffer_limit:
                                            del self.stderr_buffer[0:len(self.stderr_buffer) - self._output_buffer_limit]
                                        self.last_output_renewal = entry["timestamp"]
                    
                    # Wait before checking again
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log(f"{self.name}: Error monitoring log file: {e}")
                    await asyncio.sleep(5)  # Wait longer on error
                    
        except Exception as e:
            log(f"{self.name}: log file monitoring error: {e}")

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
        Ensures a valid MCP session is available for communication.
        
        This method handles session lifecycle management including:
        - Session reuse when possible to avoid connection overhead
        - Health checks to verify session validity
        - Graceful session recreation when needed
        - Proper cleanup and resource management
        
        Args:
            force (bool): If True, forces creation of a new session even if current one seems valid
        """
        
        # STEP 1: Check if we can reuse the existing session
        # We can reuse if: not forced, session exists, session is fresh, and service is healthy
        if (not force and self.session
                and time.time() - self.session_at < SESSION_TTL
                and self.healthy):  # Only reuse if healthy
            try:
                # Perform a quick health check on the existing session
                # Some MCP servers support ping for lightweight connectivity testing
                if hasattr(self.session, 'ping'):
                    await asyncio.wait_for(self.session.ping(), timeout=5)
                # If ping not available, trust the session is valid (health check will catch issues)
                return  # Session is good, reuse it
            except Exception as e:
                log(f"{self.name}: Session health check failed: {e}, creating new session")
                force = True  # Force new session creation
                # Mark this as a temporary reset to preserve cache
                self._temporary_session_reset = True
                
        # STEP 2: Clean up any existing session and prepare for new one
        # Close current session properly and initialize new context manager
        await self._close_session()
        self._exit_stack = contextlib.AsyncExitStack()
        
        try:
            # STEP 3: Create appropriate transport based on service configuration
            # Handle HTTP-based and stdio-based MCP servers differently
            if self.transport == "http":
                # HTTP transport: distinguish between SSE and regular HTTP endpoints
                if "/sse" in self.spec["url"]:
                    # Server-Sent Events (SSE) client for real-time streaming
                    # Returns only read_stream and write_stream (no additional info)
                    read_stream, write_stream = await self._exit_stack.enter_async_context(
                        sse_client(self.spec["url"])
                    )
                else:
                    # Regular HTTP client for request/response communication
                    # Returns read_stream, write_stream, and additional connection info
                    read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                        streamablehttp_client(self.spec["url"])
                    )
                # Create MCP session using the HTTP streams
                self.session = await self._exit_stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        client_info={"name": "Fractalic MCP Manager", "version": "0.3.0"},
                    )
                )
            else:
                # STDIO transport: communicate with local process via stdin/stdout
                # This is for MCP servers that run as separate processes
                
                # Create stderr capture wrapper for stdio servers
                original_command = self.spec["command"]
                original_args = self.spec.get("args", [])
                wrapper_script = await self._create_stderr_wrapper(original_command, original_args)
                
                # Use default stderr (don't interfere with subprocess creation)
                stdio_ctx = stdio_client(
                    StdioServerParameters(
                        command=wrapper_script,
                        args=[],  # All args are embedded in the wrapper script
                        env=self.spec.get("env", {})
                    )
                    # Note: not using errlog parameter to avoid issues with subprocess creation
                )
                
                # Create transport and session
                transport = await self._exit_stack.enter_async_context(stdio_ctx)
                self.session = await self._exit_stack.enter_async_context(
                    ClientSession(*transport)
                )
                
                # Try to access the subprocess created by stdio_client to monitor stderr
                # This approach uses process monitoring to capture stderr
                await self._setup_stdio_monitoring(transport)
                
                # Also setup log file monitoring for stderr (fallback approach)
                await self._setup_log_monitoring()
            
            # STEP 4: Perform MCP handshake with adaptive timeout handling
            # The handshake establishes the MCP protocol and confirms server capabilities
            try:
                # Use timeout based on service profile (external services get longer timeouts)
                init_timeout = self.profile.init_timeout
                    
                log(f"{self.name}: Starting MCP session initialization (timeout: {init_timeout}s)")
                await asyncio.wait_for(self.session.initialize(), timeout=init_timeout)
                log(f"{self.name}: MCP session initialization completed successfully")
            except asyncio.TimeoutError:
                # Handshake took too long - likely server is unresponsive
                error_msg = f"MCP session initialization timed out after {init_timeout}s"
                log(f"{self.name}: {error_msg}")
                self.last_error = error_msg
                raise Exception(error_msg)
            except Exception as e:
                # Handle different types of handshake failures with context-aware messaging
                error_str = str(e).lower()
                if "500 internal server error" in error_str and self.profile.is_external:
                    # External API is having issues - this is often temporary
                    error_msg = f"External service error (HTTP 500): {e}. This is likely a temporary issue with the external API."
                    log(f"{self.name}: {error_msg}")
                    self.last_error = error_msg
                    # For external service errors, mark as retriable
                    raise Exception(error_msg)
                else:
                    # Generic handshake failure - could be protocol mismatch, auth issues, etc.
                    error_msg = f"MCP session initialization failed: {e}"
                    log(f"{self.name}: {error_msg}")
                    self.last_error = error_msg
                    raise
            
            # STEP 5: Mark session as successfully established
            # Update timestamps and health status after successful handshake
            self.session_at = time.time()
            self.started_at = self.started_at or self.session_at
            self.healthy = True               # Mark as healthy after successful handshake
            # Clear temporary reset flag after successful session establishment
            if hasattr(self, '_temporary_session_reset'):
                delattr(self, '_temporary_session_reset')
            
        except Exception as e:
            # STEP 6: Clean up on any failure during session creation
            # Ensure we don't leave partial sessions or resources hanging
            await self._close_session()
            raise

    async def _close_session(self):
        """Close the session and cleanup all resources properly."""
        try:
            # Cancel any ongoing tasks first
            if hasattr(self, '_health') and self._health and not self._health.done():
                self._health.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._health
                    
            # Cancel output capture tasks
            if self._stdout_task and not self._stdout_task.done():
                self._stdout_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._stdout_task
                    
            if self._stderr_task and not self._stderr_task.done():
                self._stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._stderr_task
                    
            # Close session - check if it has a close method first
            if self.session:
                try:
                    if hasattr(self.session, 'close'):
                        await asyncio.wait_for(self.session.close(), timeout=3.0)
                    else:
                        # MCP ClientSession doesn't have close method, just clear reference
                        log(f"{self.name}: Session cleanup (no close method available)")
                except asyncio.TimeoutError:
                    log(f"{self.name}: Session close timed out")
                except Exception as e:
                    log(f"{self.name}: Error closing session: {e}")
                    
            # Close exit stack with better error handling for cross-task issues
            if self._exit_stack:
                try:
                    # Store reference and clear immediately to prevent cross-task access
                    exit_stack = self._exit_stack
                    self._exit_stack = None
                    
                    # Wrap in shield to prevent cancellation during cleanup
                    await asyncio.shield(
                        asyncio.wait_for(exit_stack.aclose(), timeout=5.0)
                    )
                except asyncio.TimeoutError:
                    log(f"{self.name}: Exit stack close timed out")
                except asyncio.CancelledError:
                    # Handle cancellation during close gracefully
                    log(f"{self.name}: Exit stack close was cancelled")
                except Exception as e:
                    # Suppress common asyncio context errors that don't affect functionality
                    error_msg = str(e)
                    if ("cancel scope" in error_msg or "different task" in error_msg or 
                        "Attempted to exit cancel scope" in error_msg or
                        "unhandled errors in a TaskGroup" in error_msg):
                        # These are common async context errors that don't affect functionality
                        pass  # Silently ignore these specific errors
                    else:
                        log(f"{self.name}: Error closing exit stack: {e}")
                    
        except Exception as e:
            log(f"{self.name}: Error during session cleanup: {e}")
        finally:
            # Always reset these regardless of errors
            self.session = None
            self._exit_stack = None
            self._stdout_task = None
            self._stderr_task = None
            self._health = None
            # Only clear cached tools if session was actually broken (not just a health check reset)
            # Keep cache for temporary reconnections to reduce repeated tool list generation
            if not hasattr(self, '_temporary_session_reset'):
                self._cached_tools = None
                self._tools_cache_time = 0
            self.session_at = 0.0
            self.healthy = False

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
    # Add tools-dump command
    dump = sub.add_parser("tools-dump")
    dump.add_argument("output_file", help="Path to output JSON file")
    dump.add_argument("target", nargs="?", default="all", help="Optional: server name (default: all)")
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
    else:
        asyncio.run(client_call(a.port, a.cmd, getattr(a, "target", None)))

if __name__ == "__main__":
    main()
