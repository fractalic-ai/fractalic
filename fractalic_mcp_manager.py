#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fractalic_mcp_manager.py –  Supervisor & API for local MCP servers
------------------------------------------------------------------
CLI:
  python fractalic_mcp_manager.py serve [--port 6000]
  python fractalic_mcp_manager.py status
  python fractalic_mcp_manager.py start all
  python fractalic_mcp_manager.py stop  mcp-server-fetch
  python fractalic_mcp_manager.py tools
"""
from __future__ import annotations

import argparse, asyncio, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import aiohttp
from aiohttp import ClientSession, web

# ───────────────────────────── constants ────────────────────────────
CONF_PATH    = Path(__file__).parent / "mcp_servers.json"
LOG_DIR      = Path(__file__).parent / "logs" ; LOG_DIR.mkdir(exist_ok=True)
DEFAULT_PORT = 5859

State = Literal["starting", "running", "retrying", "stopped", "errored"]


def ts() -> str: return time.strftime("%H:%M:%S", time.localtime())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ──────────────────────── child-process wrapper ─────────────────────
class Child:
    """
    Wraps one MCP server (HTTP or STDIO).
    *Health-checks itself every 15 s and auto-restarts (max 3 retries).*
    """
    MAX_RETRY   = 3
    CHECK_EVERY = 15           # health-probe interval
    RPC_TIMEOUT = 15           # seconds

    _STDIO_METHODS = ("tools/list", "tools.list", "list_tools")

    def __init__(self, name: str, spec: Dict[str, Any]) -> None:
        self.name, self.spec = name, spec
        self.port: Optional[int] = spec.get("port")
        self.mode: Literal["http", "stdio"] = "http" if self.port else "stdio"

        self.proc:   Optional[subprocess.Popen[str]] = None
        self.state:  State = "stopped"
        self.retries = 0
        self.last_error: Optional[str] = None
        self._check_task: Optional[asyncio.Task] = None
        self._rpc_id = 0

    # ── lifecycle ───────────────────────────────────────────────────
    async def start(self):
        if self.state in {"running", "starting"}:
            return
        self.state, self.retries = "starting", 0
        await self._spawn()

    async def stop(self):
        self.state = "stopped"
        if self._check_task:
            self._check_task.cancel()
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.to_thread(self.proc.wait), 5)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.proc = None

    def info(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "pid": self.proc.pid if self.proc else None,
            "mode": self.mode,
            "port": self.port,
            "last_error": self.last_error,
        }

    # ── spawn / health-loop ─────────────────────────────────────────
    async def _spawn(self):
        cmd = [self.spec["command"], *self.spec.get("args", [])]
        env = os.environ.copy(); env.update(self.spec.get("env", {}))

        if self.mode == "http":
            self.port = self.port or free_port()
            env.setdefault("PORT", str(self.port))

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE  if self.mode == "stdio" else subprocess.DEVNULL,
            stdout=subprocess.PIPE if self.mode == "stdio" else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self.state = "running"
        self._check_task = asyncio.create_task(self._health_loop())

    async def _health_loop(self):
        try:
            while True:
                await asyncio.sleep(self.CHECK_EVERY)
                if not await self._healthy():
                    raise RuntimeError("health-check failed")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.last_error = str(exc)
            await self._restart()

    async def _healthy(self) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return False
        if self.mode == "http":
            try:
                r, _ = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", self.port), 2)
                r.close(); return True
            except Exception:
                return False
        return True  # STDIO: process alive ⇒ healthy

    async def _restart(self):
        if self.retries >= self.MAX_RETRY:
            self.state = "errored"; return
        self.retries += 1; self.state = "retrying"
        await self.stop()
        await asyncio.sleep(2 ** (self.retries - 1))
        await self._spawn()

    # ── tool-discovery ──────────────────────────────────────────────
    async def list_tools(self):
        if self.state != "running":
            raise RuntimeError("not running")
        return (
            await http_list_tools(self.port) if self.mode == "http"
            else await self._stdio_list_tools()
        )

    async def _stdio_list_tools(self):
        if not (self.proc and self.proc.stdin and self.proc.stdout):
            raise RuntimeError("stdio handles missing")

        errors: list[str] = []
        for m in self._STDIO_METHODS:
            try:
                self._rpc_id += 1
                req = {"id": self._rpc_id, "method": m, "jsonrpc": "2.0"}
                self.proc.stdin.write(json.dumps(req) + "\n"); self.proc.stdin.flush()

                raw = await asyncio.wait_for(
                    asyncio.to_thread(self.proc.stdout.readline), self.RPC_TIMEOUT)
                resp = json.loads(raw)

                if "error" in resp:
                    raise RuntimeError(resp["error"])
                return resp.get("result", resp)   # success!
            except Exception as exc:            # keep last error
                errors.append(f"{m}: {exc}")

        raise RuntimeError(" ; ".join(errors))


# ────────────────────────── helpers ────────────────────────────────
async def http_list_tools(port: int):
    async with ClientSession() as s:
        # try /tools/list first, fall back to /list_tools
        for path in ("/tools/list", "/list_tools"):
            try:
                r = await s.get(f"http://127.0.0.1:{port}{path}", timeout=5)
                if r.status == 404:
                    continue
                r.raise_for_status()
                return await r.json()
            except Exception:
                continue
        raise RuntimeError("no /tools/list or /list_tools endpoint")


# ───────────────────────── supervisor ──────────────────────────────
class Supervisor:
    def __init__(self, cfg_path: Path = CONF_PATH):
        self.cfg = json.loads(cfg_path.read_text())
        self.children = {n: Child(n, spec)
                         for n, spec in self.cfg["mcpServers"].items()}

    async def start(self, tgt: str):
        if tgt == "all":
            await asyncio.gather(*(c.start() for c in self.children.values()))
        else:
            await self._get(tgt).start()

    async def stop(self, tgt: str):
        if tgt == "all":
            await asyncio.gather(*(c.stop() for c in self.children.values()))
        else:
            await self._get(tgt).stop()

    async def status(self): return {n: c.info() for n, c in self.children.items()}

    async def tools(self):
        out = {}
        for n, c in self.children.items():
            try:  out[n] = await c.list_tools()
            except Exception as e: out[n] = {"error": str(e)}
        return out

    def _get(self, n): return self.children.get(n) or web.HTTPNotFound(text=f"{n} unknown")


# ────────────────────── aiohttp API server ─────────────────────────
def build_app(sup: Supervisor):
    app = web.Application()

    async def _status(_): return web.json_response(await sup.status())
    async def _tools (_): return web.json_response(await sup.tools())

    async def _mut(req, act):
        tgt = req.match_info["name"]
        await getattr(sup, act)(tgt)
        return web.json_response(await sup.status())

    app.router.add_get ("/status", _status)
    app.router.add_get ("/tools",  _tools)
    app.router.add_post("/start/{name}", lambda r: _mut(r, "start"))
    app.router.add_post("/stop/{name}",  lambda r: _mut(r, "stop"))
    return app


# ────────────────────────── CLI client ─────────────────────────────
async def client_call(port: int, verb: str, tgt: Optional[str]):
    url = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as s:
        try:
            if verb == "status":  r = await s.get(url + "/status")
            elif verb == "tools": r = await s.get(url + "/tools")
            else:                 r = await s.post(f"{url}/{verb}/{tgt}")
            if r.status != 200:  print(await r.text(), file=sys.stderr)
            else:                print(json.dumps(await r.json(), indent=2))
        except aiohttp.ClientConnectionError:
            print(f"Supervisor not running on port {port}", file=sys.stderr)


# ───────────────────────── arg-parsing ─────────────────────────────
def build_parser():
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--port", type=int, default=DEFAULT_PORT, help="API port")

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=argparse.SUPPRESS)

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", parents=[p0])

    for v in ("start", "stop"):
        sp = sub.add_parser(v, parents=[p0]); sp.add_argument("target")
    sub.add_parser("status", parents=[p0])
    sub.add_parser("tools",  parents=[p0])
    return ap


# ───────────────────────── entry-points ────────────────────────────
async def run_serve(port: int):
    sup = Supervisor()
    await sup.start("all")
    print(f"[{ts()}] serving API on http://127.0.0.1:{port}")

    runner = web.AppRunner(build_app(sup)); await runner.setup()
    site   = web.TCPSite(runner, "127.0.0.1", port); await site.start()

    try:
        while True: await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\nCtrl-C – shutting down…")
    finally:
        await sup.stop("all"); await runner.cleanup()


def main():
    args = build_parser().parse_args()

    if args.cmd == "serve":
        asyncio.run(run_serve(args.port))
    else:
        asyncio.run(client_call(args.port, args.cmd, getattr(args, "target", None)))


if __name__ == "__main__":
    main()
