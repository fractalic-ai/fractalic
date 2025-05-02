#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tool_registry.py
───────────────────────────────────────────────────────────────────────────────
Single source of truth for *all* tools visible to the LLM.

Logic
─────
1. Load explicit *.yaml* manifests (if any) under tools/.
2. Autodiscover *.py / *.sh that lack a YAML:
      – introspect via cli_introspect.sniff()
      – synthesize a manifest on the fly
3. Merge in manifests from MCP servers (optional).
4. Expose:
      registry[name](**kwargs) → result
      registry.generate_schema() → list[dict]  (OpenAI format)
"""
from __future__ import annotations
import json, sys, importlib, subprocess, yaml, textwrap
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional

from .cli_introspect import sniff as sniff_cli
from .mcp_client import list_tools as mcp_list, call_tool as mcp_call

class ToolRegistry(dict):
    def __init__(self,
                 tools_dir: str | Path = "tools",
                 mcp_servers: Optional[List[str]] = None):
        super().__init__()
        self._manifests: List[Dict[str, Any]] = []
        self.tools_dir = Path(tools_dir).expanduser()
        self.mcp_servers = mcp_servers or []
        self.rescan()

    def rescan(self):
        self.clear()
        self._manifests.clear()
        self._load_yaml_manifests()
        self._autodiscover_cli()
        self._load_mcp()

    def generate_schema(self) -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": m["name"],
                "description": m["description"],
                "parameters": m["parameters"],
            },
        } for m in self._manifests]

    def _load_yaml_manifests(self):
        for y in self.tools_dir.rglob("*.yaml"):
            m = yaml.safe_load(y.read_text())
            m["_src"] = str(y.relative_to(self.tools_dir))
            self._register(m, explicit=True)

    def _autodiscover_cli(self):
        for src in self.tools_dir.rglob("*"):
            if src.is_dir() or src.suffix not in {".py", ".sh"}:
                continue
            if src.with_suffix(".yaml").exists():
                continue
            name = src.stem
            cmd_type = "python-cli" if src.suffix == ".py" else "bash-cli"
            schema, desc, runner = sniff_cli(src, cmd_type)
            manifest = {
                "name": name,
                "description": desc,
                "command": cmd_type,
                "entry": str(src),
                "parameters": schema,
                "_auto": True,
            }
            self._register(manifest, runner_override=runner)

    def _load_mcp(self):
        for srv in self.mcp_servers:
            try:
                for m in mcp_list(srv):
                    m["_mcp"] = srv
                    self._register(m, from_mcp=True)
            except Exception as e:
                print(f"[ToolRegistry] MCP {srv} skipped: {e}", file=sys.stderr)

    def _register(self, meta: Dict[str, Any],
                  explicit=False, runner_override: Callable | None = None,
                  from_mcp=False):
        name = meta["name"]
        if name in self:
            if explicit and self[name].__dict__.get("_auto"):
                pass
            else:
                return

        cmd = meta.get("command", "python")
        if runner_override:
            runner = runner_override

        elif cmd == "python" and ":" in meta["entry"]:
            mod, fn = meta["entry"].split(":")
            runner = getattr(importlib.import_module(mod), fn)

        elif cmd in {"python-cli", "bash-cli"}:
            schema, desc, runner = sniff_cli(Path(meta["entry"]), cmd)
            meta.setdefault("parameters", schema)
            # strip accidental 'help' field
            meta["parameters"]["properties"].pop("help", None)

            meta.setdefault("description", desc)

        elif cmd == "mcp":
            srv = meta.get("_mcp") or meta["mcp_server"]
            runner = lambda **kw: mcp_call(srv, name, kw)

        else:
            path = Path(meta["entry"])
            runner = lambda **kw: subprocess.run(
                [path, *map(str, kw.values())],
                capture_output=True, text=True, check=True).stdout

        self[name] = runner
        self._manifests.append(meta)
