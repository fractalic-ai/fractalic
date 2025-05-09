#!/usr/bin/env python3
# fractalic_mcp_manager_v2.py  –  Anthropic SDK version
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
#   mcp_servers.json   (same schema as before)
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, asyncio, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import aiohttp
from aiohttp import web
from anthropic import Anthropic
from anthropic.types import Tool, ToolUse
from anthropic.types.tool_use import ToolUseResult

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

class MCPClient:
    def __init__(self, name: str, spec: Dict[str, Any]):
        self.name = name
        self.spec = spec
        self.transport: Transport = spec.get("transport", "stdio")
        self.port: Optional[int] = spec.get("port")
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.client: Optional[Anthropic] = None
        self.state: State = "stopped"
        self.retries = 0
        self.last_error: Optional[str] = None
        self._health: Optional[asyncio.Task] = None
        self.tools: Optional[list[Tool]] = None

    async def start(self):
        if self.state == "running": return
        self.state = "starting"
        self.retries = 0
        await self._spawn()

    async def stop(self):
        self.state = "stopped"
        if self._health:
            self._health.cancel()
            try:
                await self._health
            except asyncio.CancelledError:
                pass
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 3)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        self.proc = None
        self.client = None
        self.tools = None

    def info(self):
        return {
            "state": self.state,
            "pid": self.proc.pid if self.proc else None,
            "mode": self.transport,
            "port": self.port,
            "last_error": self.last_error
        }

    async def _spawn(self):
        cmd = [self.spec["command"], *self.spec.get("args", [])]
        env = os.environ.copy()
        env.update(self.spec.get("env", {}))

        if self.transport == "http":
            self.port = self.port or free_port()
            env["PORT"] = str(self.port)
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
            )
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=env,
            )
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        self.state = "running"
        self._health = asyncio.create_task(self._health_loop())
        await self._refresh_tools()

    async def _health_loop(self):
        try:
            while True:
                await asyncio.sleep(HEALTH_INT)
                if not await self._healthy():
                    raise RuntimeError("health-check failed")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.last_error = str(exc)
            await self._retry()

    async def _healthy(self) -> bool:
        if self.proc is None or self.proc.returncode is not None:
            return False
        try:
            if self.transport == "stdio":
                if not self.proc.stdin or not self.proc.stdout:
                    return False
                try:
                    self.proc.stdin.write(b"\n")
                    await self.proc.stdin.drain()
                    return True
                except Exception as e:
                    self.last_error = str(e)
                    return False
            else:
                await self._refresh_tools()
                return True
        except Exception as e:
            self.last_error = str(e)
            return False

    async def _retry(self):
        if self.retries >= MAX_RETRY:
            self.state = "errored"
            return
        self.retries += 1
        self.state = "retrying"
        await self.stop()
        await asyncio.sleep(2)
        await self._spawn()

    async def _refresh_tools(self):
        try:
            if self.transport == "stdio":
                if not self.proc or self.proc.returncode is not None:
                    return {"error": "Process not running"}
                
                cmd = [self.spec["command"], *self.spec.get("args", [])]
                env = os.environ.copy()
                env.update(self.spec.get("env", {}))
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    env=env
                )
                
                try:
                    stdout, stderr = await proc.communicate()
                    if proc.returncode != 0:
                        raise RuntimeError(f"Process failed: {stderr.decode()}")
                    
                    tools_data = json.loads(stdout.decode())
                    self.tools = [Tool(**tool) for tool in tools_data.get("tools", [])]
                    return {"tools": self.tools}
                finally:
                    if proc.returncode is None:
                        proc.terminate()
                        try:
                            await asyncio.wait_for(proc.wait(), 3)
                        except asyncio.TimeoutError:
                            proc.kill()
                            await proc.wait()
            else:
                if not self.client:
                    return {"error": "Client not initialized"}
                response = await self.client.tools.list()
                self.tools = response.tools
                return {"tools": self.tools}
        except Exception as e:
            self.last_error = str(e)
            return {"error": str(e)}

    async def list_tools(self):
        if not self.tools:
            await self._refresh_tools()
        return {"tools": self.tools} if self.tools else {"error": "No tools available"}

    async def call_tool(self, name: str, arguments: dict):
        try:
            if not self.client:
                return {"error": "Client not initialized"}
            
            tool_use = ToolUse(
                id="1",
                type="tool_use",
                name=name,
                input=arguments
            )
            
            result = await self.client.tools.call(tool_use)
            return result.output
        except Exception as e:
            self.last_error = str(e)
            return {"error": str(e)}

class Supervisor:
    def __init__(self, cfg_file: Path = CONF_PATH):
        self.cfg = json.loads(cfg_file.read_text())
        self.children = {n: MCPClient(n, spec)
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
                if any(t.name == name for t in tl.get("tools", [])):
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
                        "tools": [tool.dict() for tool in tools_result.get("tools", [])]
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

async def client_call(port: int, verb: str, tgt: str | None):
    url = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as s:
        try:
            log(f"Making {verb} request to {url}")
            if verb == "status":
                r = await asyncio.wait_for(s.get(url + "/status"), TIMEOUT_RPC)
            elif verb == "tools":
                r = await asyncio.wait_for(s.get(url + "/tools"), TIMEOUT_RPC)
            elif verb in {"start", "stop"}:
                r = await asyncio.wait_for(s.post(f"{url}/{verb}/{tgt}"), TIMEOUT_RPC)
            else:
                return
            
            log(f"Got response with status {r.status}")
            if r.status == 200:
                data = await r.json()
                print(json.dumps(data, indent=2))
            else:
                try:
                    error_data = await r.json()
                    print(f"Error: {error_data.get('error', 'Unknown error')}", file=sys.stderr)
                except:
                    print(f"Error: HTTP {r.status}", file=sys.stderr)
                sys.exit(1)
        except asyncio.TimeoutError:
            print("Error: Operation timed out", file=sys.stderr)
            sys.exit(1)
        except aiohttp.ClientConnectionError:
            print("Error: Could not connect to server", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)

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

async def run_serve(port: int):
    sup = Supervisor()
    await sup.start("all")
    runner = web.AppRunner(build_app(sup))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    print(f"[{ts()}] API on http://127.0.0.1:{port} – Ctrl-C to quit")
    
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    
    def handle_signal():
        if not stop_event.is_set():
            stop_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)
    
    try:
        await stop_event.wait()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        await sup.stop("all")
        await runner.cleanup()

async def kill_server(port: int):
    try:
        subprocess.run(["pkill", "-f", f"python fractalic_mcp_manager_v2.py serve"], check=False)
        print("Server processes terminated")
    except Exception as e:
        print(f"Warning: Error killing processes: {e}")
    
    await asyncio.sleep(1)
    
    url = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession() as s:
            try:
                async with s.post(f"{url}/stop/all") as response:
                    if response.status == 200:
                        print("Server stopped gracefully")
            except:
                pass
    except:
        pass

def main():
    args = build_parser().parse_args()
    if args.cmd == "serve":
        asyncio.run(kill_server(args.port))
        
        if os.fork() == 0:
            try:
                os.setsid()
                asyncio.run(run_serve(args.port))
            except Exception as e:
                print(f"Error in server process: {e}", file=sys.stderr)
            sys.exit(0)
        else:
            print(f"Server started in background on port {args.port}")
            print("Use 'python fractalic_mcp_manager_v2.py kill' to stop the server")
    elif args.cmd == "kill":
        asyncio.run(kill_server(args.port))
    else:
        asyncio.run(client_call(args.port, args.cmd, getattr(args, "target", None)))

if __name__ == "__main__":
    main() 