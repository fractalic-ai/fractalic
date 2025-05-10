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
        self._cmd_queue = asyncio.Queue()
        self._main_task = None
        self._resources = None
        self._session_init_time = None
        self._SESSION_TTL = 3600
        self._cleanup_event = asyncio.Event()
        self._health = None
        self._CLEANUP_TIMEOUT = 3.0
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            cmd, args = await self._cmd_queue.get()
            if cmd == "start":
                await self._do_start()
            elif cmd == "stop":
                await self._do_stop()
            elif cmd == "exit":
                await self._do_stop()
                break

    async def start(self):
        await self._cmd_queue.put(("start", None))

    async def stop(self):
        await self._cmd_queue.put(("stop", None))
        await self._cleanup_event.wait()
        self._cleanup_event.clear()

    async def _do_start(self):
        if self.state == "running":
            return
        self.state = "starting"
        try:
            await self._spawn()
            self.state = "running"
        except Exception as e:
            self.last_error = str(e)
            self.state = "stopped"
            log(f"Error starting {self.spec['name']}: {e}")
            await self._do_stop()

    async def _spawn(self):
        if self.proc:
            return
        log(f"Starting process for {self.spec['name']}...")
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

    async def _initialize_session(self):
        server_params = StdioServerParameters(
            command=self.spec["command"],
            args=self.spec.get("args", []),
            env=self.spec.get("env", {})
        )
        self._resources = {}
        try:
            exit_stack = AsyncExitStack()
            stdio_context = stdio_client(server_params)
            stdio_transport = await exit_stack.enter_async_context(stdio_context)
            http_client = await exit_stack.enter_async_context(
                MCPHttpClient(f"http://localhost:{self.spec.get('port', DEFAULT_PORT)}")
            )
            session = await exit_stack.enter_async_context(
                ClientSession(stdio_transport[0], stdio_transport[1])
            )
            self._resources = {
                'exit_stack': exit_stack,
                'session': session,
                'stdio': stdio_transport[0],
                'write': stdio_transport[1],
                'stdio_context': stdio_context,
                'http_client': http_client
            }
            self._session_init_time = time.time()
            log(f"Session initialized for {self.spec['name']}")
        except Exception as e:
            log(f"Error initializing session: {e}")
            self._session_init_time = None
            if self._resources and self._resources.get('exit_stack'):
                try:
                    await self._resources['exit_stack'].aclose()
                except Exception as e:
                    log(f"Error closing exit stack during initialization: {e}")
            self._resources = None
            await self._do_stop()
            raise

    async def _do_stop(self):
        log(f"Stopping {self.spec['name']}...")
        self.state = "stopping"
        if self._health:
            self._health.cancel()
            try:
                await asyncio.wait_for(self._health, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._health = None
        if self._resources:
            try:
                await asyncio.wait_for(self._resources['exit_stack'].aclose(), timeout=self._CLEANUP_TIMEOUT)
            except Exception as e:
                log(f"Error closing exit stack: {e}")
            self._resources = None
        self._session_init_time = None
        if self.proc:
            try:
                if self.proc.returncode is None:
                    self.proc.terminate()
                    try:
                        await asyncio.wait_for(self.proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        log(f"Process termination timeout for {self.spec['name']}, sending SIGKILL")
                        self.proc.kill()
                        try:
                            await asyncio.wait_for(self.proc.wait(), timeout=1.0)
                        except asyncio.TimeoutError:
                            log(f"Failed to kill process {self.spec['name']}")
            except Exception as e:
                log(f"Error terminating process: {e}")
            finally:
                for pipe in [self.proc.stdin, self.proc.stdout, self.proc.stderr]:
                    if pipe:
                        try:
                            pipe.close()
                        except Exception:
                            pass
                self.proc = None
                self.pid = None
        self.state = "stopped"
        self._cleanup_event.set()
        log(f"Cleanup completed for {self.spec['name']}")

    def info(self):
        return {
            "state": self.state,
            "pid": self.pid,
            "last_error": self.last_error
        }

    async def list_tools(self):
        if not self._resources:
            await self._initialize_session()
        return await self._resources['session'].list_tools()

    async def call_tool(self, name: str, arguments: dict):
        if not self._resources:
            await self._initialize_session()
        return await self._resources['session'].call_tool(name, arguments)

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
    runner = None
    sup = None
    try:
        log("Initializing supervisor...")
        sup = Supervisor()
        log("Starting all servers...")
        await sup.start("all")
        log("Setting up web application...")
        runner = web.AppRunner(build_app(sup))
        await runner.setup()
        log(f"Starting web server on port {port}...")
        try:
            site = web.TCPSite(runner, "127.0.0.1", port)
            await site.start()
        except OSError as e:
            if e.errno == 48:
                log(f"Port {port} is already in use. Exiting.")
                return
            else:
                raise
        print(f"[{ts()}] API on http://127.0.0.1:{port} – Ctrl-C to quit")
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
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
            try:
                await sup.stop("all")
            except Exception as e:
                log(f"Error stopping servers: {e}")
            if runner is not None:
                try:
                    await runner.cleanup()
                except Exception as e:
                    log(f"Error cleaning up runner: {e}")
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.remove_signal_handler(sig)
                except Exception:
                    pass
            log("Shutdown complete")
    except Exception as e:
        log(f"Error in run_serve: {e}")
        if sup is not None:
            try:
                await sup.stop("all")
            except Exception as cleanup_err:
                log(f"Error during cleanup after error: {cleanup_err}")
        if runner is not None:
            try:
                await runner.cleanup()
            except Exception as cleanup_err:
                log(f"Error cleaning up runner after error: {cleanup_err}")
        raise

async def client_call(port: int, verb: str, tgt: Optional[str] = None):
    url = f"http://127.0.0.1:{port}"
    
    async with aiohttp.ClientSession() as session:
        try:
            if verb == "status":
                async with session.get(f"{url}/status") as r:
                    r = await asyncio.wait_for(r.json(), TIMEOUT_RPC)
                    print(json.dumps(r, indent=2))
                
            elif verb == "tools":
                async with session.get(f"{url}/tools") as r:
                    r = await asyncio.wait_for(r.json(), TIMEOUT_RPC)
                    print(json.dumps(r, indent=2))
                
            elif verb in ("start", "stop"):
                if not tgt:
                    print(f"Error: {verb} requires a target")
                    sys.exit(1)
                    
                async with session.post(f"{url}/{verb}/{tgt}") as r:
                    if r.status != 200:
                        print(f"Error: {r.status} {await r.text()}")
                        sys.exit(1)
                    
            elif verb == "kill":
                async with session.post(f"{url}/kill") as r:
                    if r.status != 200:
                        print(f"Error: {r.status} {await r.text()}")
                        sys.exit(1)
                    
        except asyncio.TimeoutError:
            print(f"Error: Operation timed out after {TIMEOUT_RPC}s")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

async def kill_server(port: int):
    log("Initiating server shutdown...")
    
    # First try to stop all servers gracefully through the API
    try:
        url = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{url}/stop/all", timeout=5) as r:
                    if r.status == 200:
                        log("Servers stopped gracefully through API")
                        await asyncio.sleep(1)
            except Exception as e:
                log(f"Could not stop servers through API: {e}")
    except Exception:
        pass

    # Get all Python processes related to our script
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*fractalic_mcp_manager_v2.py"],
            capture_output=True, text=True, check=False
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            log(f"Found {len(pids)} processes to terminate")
            
            # First try SIGTERM
            for pid in pids:
                try:
                    pid = int(pid)
                    os.kill(pid, signal.SIGTERM)
                    log(f"Sent SIGTERM to process {pid}")
                except ProcessLookupError:
                    continue
                except Exception as e:
                    log(f"Error sending SIGTERM to {pid}: {e}")
            
            # Wait a bit and check if processes are still running
            await asyncio.sleep(2)
            
            # Check remaining processes
            result = subprocess.run(
                ["pgrep", "-f", "python.*fractalic_mcp_manager_v2.py"],
                capture_output=True, text=True, check=False
            )
            
            if result.returncode == 0:
                remaining_pids = result.stdout.strip().split('\n')
                if remaining_pids:
                    log(f"{len(remaining_pids)} processes still running, sending SIGKILL")
                    for pid in remaining_pids:
                        try:
                            pid = int(pid)
                            os.kill(pid, signal.SIGKILL)
                            log(f"Sent SIGKILL to process {pid}")
                        except ProcessLookupError:
                            continue
                        except Exception as e:
                            log(f"Error sending SIGKILL to {pid}: {e}")
            
            # Final verification
            await asyncio.sleep(1)
            result = subprocess.run(
                ["pgrep", "-f", "python.*fractalic_mcp_manager_v2.py"],
                capture_output=True, text=True, check=False
            )
            
            if result.returncode == 0:
                log("Warning: Some processes may still be running")
            else:
                log("All processes successfully terminated")
    except Exception as e:
        log(f"Error during process termination: {e}")
        
    # Clean up any remaining port bindings
    try:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"http://127.0.0.1:{port}/status", timeout=1) as r:
                    if r.status == 200:
                        log(f"Warning: Port {port} is still in use")
            except:
                log(f"Port {port} is free")
    except Exception:
        pass

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