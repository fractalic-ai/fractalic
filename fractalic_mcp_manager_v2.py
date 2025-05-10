#!/usr/bin/env python3
# fractalic_mcp_manager_v2.py  –  improved version with proper session management
#
# CLI ─────────────────────────────────────────────────────────
#   python fractalic_mcp_manager_v2.py serve        [--port 5859]
#   python fractalic_mcp_manager_v2.py status       [--port 5859]
#   python fractalic_mcp_manager_v2.py tools        [--port 5859]
#   python fractalic_mcp_manager_v2.py start NAME   [--port 5859]
#   python fractalic_mcp_manager_v2.py stop  NAME   [--port 5859]
#
# settings.toml for Fractalic
#   [settings]
#   mcpServers = ["http://127.0.0.1:5859"]
#
# Config file
#   mcp_servers.json   (same schema you showed – ports optional)
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, asyncio, json, os, signal, socket, subprocess, sys, time, shlex
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from contextlib import AsyncExitStack

import aiohttp
from aiohttp import web
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client as MCPHttpClient

CONF_PATH    = Path(__file__).parent / "mcp_servers.json"
DEFAULT_PORT = 5859

State        = Literal["starting", "running", "retrying", "stopped", "errored"]
Transport    = Literal["stdio", "http"]
MAX_RETRY    = 2          # keep short
HEALTH_INT   = 10         # seconds
TIMEOUT_RPC  = 30         # increase timeout for RPC calls

def ts() -> str: return time.strftime("%H:%M:%S", time.localtime())

def log(msg: str):
    print(f"[{ts()}] {msg}")

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0)); return s.getsockname()[1]

class Child:
    def __init__(self, spec):
        self.spec = spec
        self.state = "stopped"
        self.proc = None
        self.pid = None
        self.last_error = None
        self.session = None
        self._task_group = None
        self._health = None
        self._cleanup_lock = asyncio.Lock()
        self.exit_stack = AsyncExitStack()

    async def start(self):
        if self.state == "running":
            return
            
        log(f"Starting {self.spec['name']}...")
        self.state = "starting"
        
        try:
            await self._spawn()
            log(f"Started {self.spec['name']}")
        except Exception as e:
            self.last_error = str(e)
            self.state = "stopped"
            log(f"Error starting {self.spec['name']}: {e}")
            raise

    async def _spawn(self):
        if self.proc:
            return

        log(f"Starting process for {self.spec['name']}...")
        
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *shlex.split(self.spec["command"]),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.pid = self.proc.pid
            
            log(f"Process started with PID {self.pid}")
            
            await self._initialize_session()
            
            self._health = asyncio.create_task(self._health_check())
            self.state = "running"
            
        except Exception as e:
            self.last_error = str(e)
            self.state = "stopped"
            log(f"Error starting process: {e}")
            await self._cleanup()
            raise

    async def _initialize_session(self):
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("Process not properly initialized")

        server_params = StdioServerParameters(
            command=self.spec["command"],
            args=self.spec.get("args", []),
            env=self.spec.get("env", {})
        )

        try:
            # Create a new task group for this session
            self._task_group = asyncio.TaskGroup()
            async with self._task_group as tg:
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                self.stdio, self.write = stdio_transport
                self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
                log(f"Session initialized for {self.spec['name']}")
        except Exception as e:
            log(f"Error initializing session: {e}")
            await self._cleanup()
            raise

    async def _cleanup(self):
        async with self._cleanup_lock:
            log(f"Cleaning up resources for {self.spec['name']}...")
            
            # Cancel health check task
            if self._health:
                self._health.cancel()
                try:
                    await self._health
                except asyncio.CancelledError:
                    pass
                self._health = None

            # Cancel task group
            if self._task_group:
                try:
                    await self._task_group.__aexit__(None, None, None)
                except Exception as e:
                    log(f"Error cancelling task group: {e}")
                self._task_group = None

            # Close session
            if self.session:
                try:
                    await self.session.close()
                except Exception as e:
                    log(f"Error closing session: {e}")
                self.session = None

            # Terminate process
            if self.proc:
                try:
                    self.proc.terminate()
                    try:
                        await asyncio.wait_for(self.proc.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self.proc.kill()
                        await self.proc.wait()
                except Exception as e:
                    log(f"Error terminating process: {e}")
                self.proc = None
                self.pid = None

            # Close exit stack
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                log(f"Error closing exit stack: {e}")

            self.state = "stopped"
            log(f"Cleanup completed for {self.spec['name']}")

    async def stop(self):
        log(f"Stopping {self.spec['name']}...")
        self.state = "stopping"
        await self._cleanup()
        log(f"Stopped {self.spec['name']}")

    async def _health_check(self):
        while True:
            try:
                if not self.proc or self.proc.returncode is not None:
                    log(f"Process {self.spec['name']} terminated unexpectedly")
                    await self.stop()
                    break
                    
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log(f"Health check error for {self.spec['name']}: {e}")
                await self.stop()
                break

    def info(self):
        return {
            "state": self.state,
            "pid": self.pid,
            "last_error": self.last_error
        }

    async def list_tools(self):
        try:
            if not self.session:
                await self._initialize_session()
            return await self.session.list_tools()
        except Exception as e:
            self.last_error = str(e)
            return {"error": str(e)}

    async def call_tool(self, name: str, arguments: dict):
        try:
            if not self.session:
                await self._initialize_session()
            return await self.session.call_tool(name, arguments)
        except Exception as e:
            self.last_error = str(e)
            return {"error": str(e)}

class Supervisor:
    def __init__(self, cfg_file: Path = CONF_PATH):
        self.cfg = json.loads(cfg_file.read_text())
        self.children = {n: Child(spec)
                        for n, spec in self.cfg["mcpServers"].items()}

    async def start(self, tgt): await self._each("start", tgt)
    async def stop(self, tgt): await self._each("stop", tgt)
    async def status(self): return {n: c.info() for n, c in self.children.items()}

    async def tools(self):
        out = {}
        for n, c in self.children.items():
            try:
                log(f"Getting tools for {n}")
                tools_task = asyncio.create_task(c.list_tools())
                try:
                    result = await asyncio.wait_for(tools_task, TIMEOUT_RPC)
                    out[n] = result
                    log(f"Successfully got tools for {n}")
                except asyncio.TimeoutError:
                    out[n] = {"error": "Operation timed out"}
                    log(f"Timeout getting tools for {n}")
                except Exception as e:
                    out[n] = {"error": str(e)}
                    log(f"Error getting tools for {n}: {e}")
            except Exception as e:
                out[n] = {"error": str(e)}
                log(f"Error getting tools for {n}: {e}")
        return out

    async def call_tool(self, name: str, args: Dict[str, Any]):
        for c in self.children.values():
            try:
                tl = await c.list_tools()
                if any(t["name"] == name for t in tl):
                    return await c.call_tool(name, args)
            except Exception:
                pass
        raise web.HTTPNotFound(text=f"tool {name!r} not found")

    async def _each(self, act, tgt):
        if tgt == "all":
            await asyncio.gather(*(getattr(c, act)() for c in self.children.values()))
        else:
            c = self.children.get(tgt)
            if not c: raise web.HTTPNotFound(text=f"{tgt} unknown")
            await getattr(c, act)()

def build_app(sup: Supervisor):
    app = web.Application()
    app.router.add_get("/status", lambda r: _status(r, sup))
    app.router.add_get("/tools", lambda r: _tools(r, sup))
    app.router.add_post("/start/{name}", lambda r: _mut(r, sup, "start"))
    app.router.add_post("/stop/{name}", lambda r: _mut(r, sup, "stop"))
    app.router.add_post("/call_tool", lambda r: _call(r, sup))
    return app

async def _status(req, sup):
    return web.json_response(await sup.status())

async def _tools(req, sup):
    try:
        results = await sup.tools()
        serialized_results = {}
        for service_name, tools_result in results.items():
            if isinstance(tools_result, dict) and "error" in tools_result:
                serialized_results[service_name] = tools_result
            else:
                try:
                    serialized_results[service_name] = {
                        "tools": [str(tool) for tool in tools_result.tools]
                    }
                except Exception as e:
                    serialized_results[service_name] = {"error": f"Failed to serialize tools: {str(e)}"}
        return web.json_response(serialized_results)
    except Exception as e:
        log(f"Error in tools endpoint: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def _mut(req, sup, act):
    await getattr(sup, act)(req.match_info["name"])
    return web.json_response(await sup.status())

async def _call(req, sup):
    body = await req.json()
    res = await sup.call_tool(body["name"], body.get("arguments", {}))
    return web.json_response(res)

async def run_serve(port: int):
    try:
        log("Initializing supervisor...")
        sup = Supervisor()
        
        log("Starting all servers...")
        await sup.start("all")
        
        log("Setting up web application...")
        runner = web.AppRunner(build_app(sup))
        await runner.setup()
        
        log(f"Starting web server on port {port}...")
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        
        print(f"[{ts()}] API on http://127.0.0.1:{port} – Ctrl-C to quit")
        
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        
        def handle_signal():
            if not stop_event.is_set():
                log("Received shutdown signal")
                stop_event.set()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, handle_signal)
            
        try:
            await stop_event.wait()
        finally:
            log("Shutting down servers...")
            await sup.stop("all")
            await runner.cleanup()
            log("Shutdown complete")
            
    except Exception as e:
        log(f"Error in run_serve: {e}")
        raise

async def client_call(port: int, verb: str, tgt: Optional[str] = None):
    url = f"http://127.0.0.1:{port}"
    
    async with aiohttp.ClientSession() as s:
        try:
            if verb == "status":
                r = await asyncio.wait_for(s.get(f"{url}/status"), TIMEOUT_RPC)
                print(json.dumps(await r.json(), indent=2))
                
            elif verb == "tools":
                r = await asyncio.wait_for(s.get(f"{url}/tools"), TIMEOUT_RPC)
                print(json.dumps(await r.json(), indent=2))
                
            elif verb in ("start", "stop"):
                if not tgt:
                    print(f"Error: {verb} requires a target")
                    sys.exit(1)
                    
                r = await asyncio.wait_for(s.post(f"{url}/{verb}/{tgt}"), TIMEOUT_RPC)
                if r.status != 200:
                    print(f"Error: {r.status} {await r.text()}")
                    sys.exit(1)
                    
            elif verb == "kill":
                r = await asyncio.wait_for(s.post(f"{url}/kill"), TIMEOUT_RPC)
                if r.status != 200:
                    print(f"Error: {r.status} {await r.text()}")
                    sys.exit(1)
                    
        except asyncio.TimeoutError:
            print(f"Error: Operation timed out after {TIMEOUT_RPC}s")
            sys.exit(1)
        except aiohttp.ClientError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

async def kill_server(port: int):
    log("Initiating server shutdown...")
    
    # First try to stop all servers gracefully with a short timeout
    try:
        url = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession() as s:
            try:
                # Use a shorter timeout for graceful shutdown attempt
                r = await asyncio.wait_for(s.post(f"{url}/stop/all"), timeout=5.0)
                if r.status == 200:
                    log("Servers stopped gracefully")
                    await asyncio.sleep(1)  # Give servers time to cleanup
            except asyncio.TimeoutError:
                log("Graceful shutdown timed out")
            except Exception as e:
                log(f"Error during graceful shutdown: {e}")
    except Exception:
        pass

    # Force kill any remaining processes
    try:
        # First try SIGTERM
        subprocess.run(["pkill", "-TERM", "-f", "python fractalic_mcp_manager_v2.py"], check=False)
        await asyncio.sleep(2)  # Give processes time to handle SIGTERM
        
        # Then check if any processes remain
        result = subprocess.run(["pgrep", "-f", "python fractalic_mcp_manager_v2.py"], 
                              capture_output=True, text=True, check=False)
        
        # If processes still exist, use SIGKILL
        if result.returncode == 0:
            log("Some processes still running, forcing termination...")
            subprocess.run(["pkill", "-KILL", "-f", "python fractalic_mcp_manager_v2.py"], check=False)
            
        log("Server processes terminated")
    except Exception as e:
        log(f"Warning: Error killing processes: {e}")
    
    # Final verification
    try:
        result = subprocess.run(["pgrep", "-f", "python fractalic_mcp_manager_v2.py"], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            log("Warning: Some processes may still be running")
        else:
            log("All processes successfully terminated")
    except Exception as e:
        log(f"Warning: Error verifying process termination: {e}")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    sub.add_parser("kill")
    for v in ("start", "stop"):
        c = sub.add_parser(v)
        c.add_argument("target")
    sub.add_parser("status")
    sub.add_parser("tools")
    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "serve":
        asyncio.run(run_serve(args.port))
    elif args.cmd == "kill":
        asyncio.run(kill_server(args.port))
    else:
        asyncio.run(client_call(args.port, args.cmd, getattr(args, "target", None)))

if __name__ == "__main__":
    main() 