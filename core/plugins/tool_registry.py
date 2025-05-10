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
import os
import logging

from .cli_introspect import sniff as sniff_cli
from .mcp_client import list_tools as mcp_list, call_tool as mcp_call
from core.config import Config

# Add the missing ToolParameterParser class
class ToolParameterParser:
    """Convert tool parameters to command line arguments"""
    def __init__(self, properties: Dict[str, Any]):
        self.properties = properties
    
    def convert_to_cli_args(self, kw: Dict[str, Any]) -> List[str]:
        """Convert keyword arguments to CLI arguments"""
        args = []
        for k, v in kw.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:  # Only add flag if True
                    args.append(flag)
            else:
                args.extend([flag, str(v)])
        return args

from core.utils import load_settings # Ensure load_settings is imported if needed elsewhere, though Config should handle it

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
        print(f"[ToolRegistry] MCP servers to load: {self.mcp_servers}")
        for srv in self.mcp_servers:
            try:
                response = mcp_list(srv)
                print(f"[ToolRegistry] MCP {srv} raw response: {response} (type: {type(response)})")
                if not response:
                    print(f"[ToolRegistry] MCP {srv} returned empty or None response.")
                for m in response:
                    print(f"[ToolRegistry] Registering MCP tool manifest: {m}")
                    m["_mcp"] = srv
                    self._register(m, from_mcp=True)
                    print(f"[ToolRegistry] Registered MCP tool: {m.get('name', '<no name>')} from {srv}")
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

        elif meta.get("type") == "cli":
            path = Path(meta["entry"])
            def runner_with_env(**kw):
                env = None
                # Inject environment variables from settings.toml if present
                if Config.TOML_SETTINGS and 'environment' in Config.TOML_SETTINGS:
                    env = os.environ.copy()
                    for env_var in Config.TOML_SETTINGS['environment']:
                        if 'key' in env_var and 'value' in env_var:
                            env[env_var['key']] = env_var['value']
                    # --- BEGIN DEBUG PRINT ---
                    print(f"DEBUG: Injecting env for {name}: { {k: ('***' if 'KEY' in k.upper() else v) for k, v in env.items() if k == 'TAVILY_API_KEY'} }", file=sys.stderr)
                    # --- END DEBUG PRINT ---
                else:
                    # --- BEGIN DEBUG PRINT ---
                    print(f"DEBUG: NOT Injecting env for {name}. Config.TOML_SETTINGS: {Config.TOML_SETTINGS is not None}, 'environment' in settings: {'environment' in Config.TOML_SETTINGS if Config.TOML_SETTINGS else 'N/A'}", file=sys.stderr)
                    # --- END DEBUG PRINT ---

                # Convert boolean args to flags, handle other types
                args = [str(path)]
                parser = ToolParameterParser(meta["parameters"]["properties"])
                cli_args = parser.convert_to_cli_args(kw)
                args.extend(cli_args)

                try:
                    # Use the potentially modified env
                    result = subprocess.run(
                        args, # Pass the constructed args list
                        capture_output=True, text=True, check=True, env=env
                    )
                    return result.stdout
                except subprocess.CalledProcessError as e:
                    # Log or return stderr for better debugging in case of tool error
                    error_message = f"Tool '{name}' failed with exit code {e.returncode}.\nArgs: {' '.join(args)}\nStderr: {e.stderr}"
                    logging.error(error_message)
                    # Return a dictionary indicating error, including stderr
                    return {"error": error_message, "stderr": e.stderr}
                except FileNotFoundError:
                    error_message = f"Tool '{name}' executable not found at {path}."
                    logging.error(error_message)
                    return {"error": error_message}

            runner = runner_with_env
            self.tools[name] = {"meta": meta, "runner": runner}
            logging.debug(f"Registered CLI tool: {name}")

        else:
            path = Path(meta["entry"])
            def runner_with_env(**kw):
                env = None
                # Inject environment variables from settings.toml if present
                if Config.TOML_SETTINGS and 'environment' in Config.TOML_SETTINGS:
                    env = os.environ.copy()
                    for env_var in Config.TOML_SETTINGS['environment']:
                        if 'key' in env_var and 'value' in env_var:
                            env[env_var['key']] = env_var['value']
                return subprocess.run(
                    [path, *map(str, kw.values())],
                    capture_output=True, text=True, check=True, env=env
                ).stdout
            runner = runner_with_env

        self[name] = runner
        self._manifests.append(meta)
