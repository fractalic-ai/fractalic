#!/usr/bin/env python3
# fractalic_mcp_manager.py – Fractalic MCP supervisor (auto-transport, health, back-off)
#
# CLI -----------------------------------------------------
#   python fractalic_mcp_manager.py serve        [--port 5859]
#   python fractalic_mcp_manager.py status       [--port 5859]
#   python fractalic_mcp_manager.py tools        [--port 5859]
#   python fractalic_mcp_manager.py start NAME   [--port 5859]
#   python fractalic_mcp_manager.py stop  NAME   [--port 5859]
# ---------------------------------------------------------
from __future__ import annotations

import argparse, asyncio, contextlib, dataclasses, datetime, json, os, shlex, signal, subprocess, sys, time, gc
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import aiohttp
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions, CorsViewMixin

from mcp.client.session          import ClientSession
from mcp.client.stdio            import stdio_client, StdioServerParameters
from mcp.client.streamable_http  import streamablehttp_client
from mcp.client.sse              import sse_client

import errno
import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# -------------------------------------------------------------------- constants
CONF_PATH    = Path(__file__).parent / "mcp_servers.json"
DEFAULT_PORT = 5859

State     = Literal["starting", "running", "retrying", "stopped", "errored"]
Transport = Literal["stdio", "http"]

TIMEOUT_INITIAL = 120      # s – Increased timeout for slow operations like Zapier
HEALTH_INT    = 45         # s – between health probes (increased to avoid heartbeat clashes)
SESSION_TTL   = 3600       # s – refresh session after this period
MAX_RETRY     = 5
BACKOFF_BASE  = 2          # exponential back-off

# Configuration flags
ENABLE_SCHEMA_SANITIZATION = True  # Set to True to enable Vertex AI schema sanitization

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
        # Handle CallToolResult objects and other custom classes
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)

# -------------------------------------------------------------------- helpers
def tool_to_obj(t):
    if isinstance(t, dict):
        return t                          # already JSON-ready
    if dataclasses.is_dataclass(t):
        return dataclasses.asdict(t)      # MCP canonical form
    return json.loads(t.model_dump_json()) if hasattr(t, "model_dump_json") else str(t)

def ts() -> str: return time.strftime("%H:%M:%S", time.localtime())
def log(msg: str): print(f"[{ts()}] {msg}", file=sys.stderr)

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
        self.health_failures = 0          # Track consecutive health check failures
        # --- New fields for output capture ---
        self.stdout_buffer = []  # List of dicts: {"timestamp": str, "line": str}
        self.stderr_buffer = []  # List of dicts: {"timestamp": str, "line": str}
        self.last_output_renewal = None  # ISO8601 string of last output change
        self._output_buffer_limit = 1000  # Max lines to keep per buffer
        # --- Caching for tools_info ---
        self._last_tools_list = None
        self._last_token_count = None
        self._last_schema_json = None

    async def start(self):
        if self.state == "running": return
        self.state = "starting"; self.retries = 0
        startup_delay = int(self.spec.get("env", {}).get("STARTUP_DELAY", "0"))
        if startup_delay > 0:
            log(f"Waiting {startup_delay}ms for {self.name} to initialize...")
            await asyncio.sleep(startup_delay / 1000)
        await self._spawn_if_needed()
        retry_count = int(self.spec.get("env", {}).get("RETRY_COUNT", "3"))
        retry_delay = int(self.spec.get("env", {}).get("RETRY_DELAY", "2000")) / 1000
        for attempt in range(retry_count):
            try:
                tools = await asyncio.wait_for(self.list_tools(), timeout=10)
                if tools and not isinstance(tools, dict) or "error" not in tools:
                    log(f"{self.name} tools available after {attempt + 1} attempts")
                    self.state = "running"
                    self.healthy = True
                    return
            except Exception as e:
                log(f"Attempt {attempt + 1} failed for {self.name}: {e}")
            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay)
        # If we reach here, all attempts failed: kill process, mark errored
        log(f"{self.name} failed to expose tools after {retry_count} attempts, killing process and marking as errored.")
        await self._close_session()
        if self.proc:
            try:
                log(f"Killing {self.name} (pid {self.pid}) after failed tool exposure.")
                self.proc.kill()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except Exception as kill_exc:
                log(f"Kill failed for {self.name}: {kill_exc}")
            if self.pid:
                try:
                    import signal
                    log(f"Trying os.kill on {self.name} (pid {self.pid}) after failed tool exposure.")
                    os.kill(self.pid, signal.SIGKILL)
                except Exception as oskill_exc:
                    log(f"os.kill failed for {self.name}: {oskill_exc}")
        self.proc, self.pid = None, None
        self.state = "errored"
        self.last_error = f"Failed to get tools after {retry_count} attempts"
        log(f"{self.name} is now marked as errored and will not be restarted.")
        return  # Always return so supervisor can proceed

    async def stop(self):
        await self._cmd_q.put(("stop",))

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
        try:
            await self._spawn_if_needed()
            await self._ensure_session(force=True)
            self._health = asyncio.create_task(self._health_loop())
            self.state   = "running"
            log(f"{self.name} ↑ ({self.transport})")
        except Exception as e:
            self.state = "errored"
            log(f"{self.name} failed to start: {e}")
            await self._schedule_retry()

    async def _do_stop(self):
        if self.state == "stopped":
            return
        self.state = "stopping"
        if self._health:
            self._health.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health
        await self._close_session()
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
            for pipe in [self.proc.stdin, self.proc.stdout, self.proc.stderr]:
                if pipe:
                    try:
                        pipe.close()
                    except Exception:
                        pass
        self.proc, self.pid = None, None
        self.state          = "stopped"
        log(f"{self.name} ↓")
        if self._runner:
            self._runner.cancel()

    async def _spawn_if_needed(self):
        if self.transport == "http":
            return
        if self.proc and self.proc.returncode is None:
            return
        try:
            env = {**os.environ, **self.spec.get("env", {})}
            command_parts = shlex.split(self.spec["command"])
            args = self.spec.get("args", [])
            log(f"Spawning {self.name} with command: {command_parts} {args}")
            
            self.proc = await asyncio.create_subprocess_exec(
                *command_parts,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self.pid = self.proc.pid
            self.started_at = time.time()
            
            # --- New: Capture stdout and stderr output with timestamps ---
            import datetime
            def iso_now():
                return datetime.datetime.now().isoformat()
            async def capture_output(stream, buffer, stream_name):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='replace').rstrip('\n')
                    entry = {"timestamp": iso_now(), "line": decoded}
                    buffer.append(entry)
                    # Limit buffer size
                    if len(buffer) > self._output_buffer_limit:
                        del buffer[0:len(buffer)-self._output_buffer_limit]
                    self.last_output_renewal = entry["timestamp"]
                    # Add prefix with process name, stream, and timestamp
                    print(f"[{entry['timestamp']}] [{self.name} {stream_name}] {decoded}")
                    log(f"{self.name} {stream_name}: {decoded}")
            
            asyncio.create_task(capture_output(self.proc.stdout, self.stdout_buffer, 'stdout'))
            asyncio.create_task(capture_output(self.proc.stderr, self.stderr_buffer, 'stderr'))
            
        except Exception as e:
            log(f"Error spawning {self.name}: {e}")
            raise

    async def _ensure_session(self, force=False):
        if (not force and self.session
                and time.time() - self.session_at < SESSION_TTL):
            return
        await self._close_session()
        self._exit_stack = contextlib.AsyncExitStack()
        if self.transport == "http":
            # Check if this is an SSE endpoint (generic detection based on URL path)
            if self.spec["url"].endswith("/sse"):
                # Use SSE client for SSE endpoints (returns only read_stream, write_stream)
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    sse_client(self.spec["url"])
                )
            else:
                # Use streamable HTTP client for other HTTP services (returns read_stream, write_stream, _)
                read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                    streamablehttp_client(self.spec["url"])
                )
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    client_info={"name": "Fractalic MCP Manager", "version": "0.3.0"},
                )
            )
        else:
            stdio_ctx = stdio_client(StdioServerParameters(
                command=self.spec["command"],
                args=self.spec.get("args", []),
                env=self.spec.get("env", {})
            ))
            transport = await self._exit_stack.enter_async_context(stdio_ctx)
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(*transport)
            )
        # Mandatory MCP handshake with timeout
        try:
            # Add timeout specifically for session initialization to debug Zapier issues
            init_timeout = 30  # seconds
            log(f"{self.name}: Starting MCP session initialization (timeout: {init_timeout}s)")
            await asyncio.wait_for(self.session.initialize(), timeout=init_timeout)
            log(f"{self.name}: MCP session initialization completed successfully")
        except asyncio.TimeoutError:
            error_msg = f"MCP session initialization timed out after {init_timeout}s"
            log(f"{self.name}: {error_msg}")
            self.last_error = error_msg
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"MCP session initialization failed: {e}"
            log(f"{self.name}: {error_msg}")
            self.last_error = error_msg
            raise
        
        self.session_at = time.time()
        self.started_at = self.started_at or self.session_at
        self.healthy = True               # Mark as healthy after successful handshake

    async def _close_session(self):
        if self._exit_stack:
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
        self.session, self._exit_stack = None, None

    async def _health_loop(self):
        while True:
            try:
                await asyncio.sleep(HEALTH_INT)
                await self._ensure_session()
                await asyncio.wait_for(self.session.list_tools(), timeout=TIMEOUT_INITIAL / 3)
                self.healthy = True        # Mark as healthy after successful tool list
                self.health_failures = 0   # Reset on success
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.healthy = False       # Mark as unhealthy before retry
                self.last_error = str(e)   # Store the error message
                self.health_failures += 1
                log(f"{self.name} unhealthy: {e} (failure {self.health_failures})")
                if self.health_failures >= 2:
                    log(f"{self.name} failed health check twice, killing process and marking as errored.")
                    await self._close_session()
                    if self.proc:
                        try:
                            log(f"Attempting graceful kill of {self.name} (pid {self.pid})")
                            self.proc.kill()
                            await self.proc.wait()
                        except Exception as kill_exc:
                            log(f"Graceful kill failed for {self.name}: {kill_exc}")
                        # Fallback: try os.kill if process still exists
                        if self.pid:
                            try:
                                log(f"Trying os.kill on {self.name} (pid {self.pid})")
                                os.kill(self.pid, signal.SIGKILL)
                            except Exception as oskill_exc:
                                log(f"os.kill failed for {self.name}: {oskill_exc}")
                    self.proc, self.pid = None, None
                    self.state = "errored"
                    log(f"{self.name} is now marked as errored and will not be restarted.")
                    break
                else:
                    await self._schedule_retry()
                    break

    async def _schedule_retry(self):
        await self._close_session()
        if self.proc:
            with contextlib.suppress(Exception):
                self.proc.kill()
                await self.proc.wait()
        if self.retries >= MAX_RETRY:
            self.state = "errored"
            log(f"{self.name} exceeded retries → errored")
            return
        self.retries += 1
        backoff = BACKOFF_BASE ** self.retries
        self.state   = "retrying"
        log(f"{self.name} retrying in {backoff}s …")
        await asyncio.sleep(backoff)
        self.restart_count += 1           # Increment restart count before retry
        await self._do_start()

    async def list_tools(self):
        await self._ensure_session()
        return await asyncio.wait_for(self.session.list_tools(), TIMEOUT_INITIAL)

    async def call_tool(self, tool: str, args: dict):
        await self._ensure_session()
        
        log(f"{self.name}: Calling tool '{tool}' with timeout {TIMEOUT_INITIAL}s")
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool, args), TIMEOUT_INITIAL)
            log(f"{self.name}: Tool '{tool}' completed successfully")
            return result
        except asyncio.TimeoutError as e:
            error_msg = f"Tool '{tool}' timed out after {TIMEOUT_INITIAL}s"
            log(f"{self.name}: {error_msg}")
            raise Exception(error_msg) from e
        except Exception as e:
            log(f"{self.name}: Tool '{tool}' failed: {e}")
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
        """Return tool count and token count of the schema for this child/server."""
        if self.state != "running":
            return {"tool_count": 0, "token_count": 0, "tools_error": f"MCP state is {self.state}"}
        try:
            tl = await self.list_tools()
            tools_list = [tool_to_obj(t) for t in tl.tools]
            if tools_list == self._last_tools_list:
                return {
                    "tool_count": len(self._last_tools_list),
                    "token_count": self._last_token_count
                }
            schema_json = json.dumps(tools_list)
            token_count = len(TOKENIZER.encode(schema_json))
            self._last_tools_list = tools_list
            self._last_token_count = token_count
            self._last_schema_json = schema_json
            return {"tool_count": len(tools_list), "token_count": token_count}
        except Exception as e:
            return {"tool_count": 0, "token_count": 0, "tools_error": str(e)}

# ==================================================================== Supervisor
class Supervisor:
    def __init__(self, file: Path = CONF_PATH):
        cfg = json.loads(file.read_text())
        self.children = {n: Child(n, spec) for n, spec in cfg["mcpServers"].items()}

    async def start(self, tgt):
        if tgt == "all":
            # Launch all children as background tasks so API can start even if some fail
            for c in self.children.values():
                asyncio.create_task(c.start())
        else:
            c = self.children.get(tgt)
            if not c: raise web.HTTPNotFound(text=f"{tgt} unknown")
            await c.start()

    async def stop(self, tgt):
        await self._each("stop", tgt)

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
                out[n] = {"error": f"MCP state is {c.state}", "tools": []}
                continue
            try:
                tl = await c.list_tools()
                tools_list = [tool_to_obj(t) for t in tl.tools]
                # Apply Vertex AI schema sanitization only if enabled
                if ENABLE_SCHEMA_SANITIZATION:
                    sanitized_tools = [sanitize_tool_schema(tool) for tool in tools_list]
                else:
                    sanitized_tools = tools_list  # No sanitization
                out[n] = {"tools": sanitized_tools}
            except Exception as e:
                log(f"Error getting tools from {n}: {e}")
                out[n] = {"error": str(e) or f"Unknown error from {n}", "tools": []}
        return out

    async def call_tool(self, name: str, args: Dict[str, Any]):
        for c in self.children.values():
            if c.state == "errored":
                continue  # Skip errored children
            try:
                tl = await c.list_tools()
                if any(t.name == name for t in tl.tools):
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
    app.router.add_post("/call_tool",   lambda r: _call(r, sup))
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

async def _kill(req, sup: Supervisor, stop_ev: asyncio.Event):
    # 1) stop all child servers
    await sup.stop("all")
    # 2) tell the main loop in run_serve() to exit
    stop_ev.set()
    return web.json_response({"status": "shutting-down"}, dumps=lambda obj: json.dumps(obj, cls=MCPEncoder))

# ==================================================================== runners
async def run_serve(port: int):
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
    await sup.stop("all"); await runner.cleanup()
    await asyncio.sleep(0.1)  # Give time for all async cleanup
    gc.collect()              # Force garbage collection
    await asyncio.sleep(0.1)  # Allow any finalizers to run

async def client_call(port: int, verb: str, tgt: Optional[str] = None):
    url = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as s:
        try:
            if verb in ("status", "tools"):
                r = await s.get(f"{url}/{verb}"); print(json.dumps(await r.json(), indent=2))
            elif verb in ("start", "stop"):
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
    for v in ("start", "stop"):
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
