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
        """Generate OpenAI-compatible schema for all registered tools."""
        schema = []
        for m in self._manifests:
            # Skip invalid manifests
            if not isinstance(m, dict) or "name" not in m:
                continue
            # Create the function schema
            function_schema = {
                "name": m["name"],
                "description": m.get("description", ""),
                "parameters": m.get("parameters", {"type": "object", "properties": {}})
            }
            # Add to schema list
            schema.append({
                "type": "function",
                "function": function_schema
            })
        return schema

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
            cmd_type = "python-cli" if src.suffix == ".py" else "bash-cli"
            schema, desc, runner = sniff_cli(src, cmd_type)
            
            # Check if sniff_cli returned simple tool detection (schema will be dict, not list)
            if isinstance(schema, dict) and schema.get("description") == "Simple JSON tool - accepts JSON input, returns JSON output":
                # This is a simple JSON tool
                name = src.stem
                manifest = {
                    "name": name,
                    "description": desc,
                    "command": "simple-json",
                    "entry": str(src),
                    "parameters": schema,
                    "_auto": True,
                    "_simple": True,  # Mark as simple tool
                }
                self._register(manifest, runner_override=runner)
                continue
            
            if isinstance(schema, list):
                # Register each manifest in the list (multi-tool)
                for manifest in schema:
                    # Ensure the manifest has the required fields
                    if not isinstance(manifest, dict) or "name" not in manifest:
                        print(f"[ToolRegistry] Invalid manifest format in {src}: {manifest}")
                        continue
                    manifest["entry"] = str(src)
                    manifest["command"] = cmd_type
                    manifest["_auto"] = True
                    
                    # Create a specific runner for this tool that includes the tool name as first argument
                    tool_name = manifest["name"]
                    exec_prefix = [sys.executable, str(src)] if cmd_type == "python-cli" else [str(src)]
                    
                    def make_tool_runner(name, prefix):
                        def run_tool(**kw):
                            argv = prefix[:] + [name]  # Add the tool name as the first argument
                            for k, v in kw.items():
                                flag = f"--{k.replace('_','-')}"
                                if isinstance(v, bool):
                                    if v:
                                        argv.append(flag)
                                else:
                                    argv += [flag, str(v)]

                            # inject settings.toml environment if present
                            env = None
                            if Config.TOML_SETTINGS and 'environment' in Config.TOML_SETTINGS:
                                env = os.environ.copy()
                                for item in Config.TOML_SETTINGS['environment']:
                                    if 'key' in item and 'value' in item:
                                        env[item['key']] = item['value']

                            out = subprocess.run(
                                argv,
                                capture_output=True,
                                text=True,
                                env=env
                            )
                            if out.returncode != 0:
                                raise RuntimeError(out.stderr.strip() or out.stdout.strip())
                            try:
                                return json.loads(out.stdout)
                            except json.JSONDecodeError:
                                # For fractalic tools, return raw stdout instead of wrapping in JSON
                                return out.stdout.strip()
                        return run_tool
                    
                    tool_runner = make_tool_runner(tool_name, exec_prefix)
                    self._register(manifest, runner_override=tool_runner)
                # If multi-tool, do NOT register the script stem as a tool
                continue
            # Single-tool fallback
            name = src.stem
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
                print(f"[ToolRegistry] Attempting to load tools from {srv}")
                response = mcp_list(srv)
                print(f"[ToolRegistry] MCP {srv} raw response: {response}")
                
                if not response:
                    print(f"[ToolRegistry] MCP {srv} returned empty or None response.")
                    continue
                
                # Check if all services have errors
                if isinstance(response, dict) and all(isinstance(service_data, dict) and "error" in service_data for service_data in response.values()):
                    print(f"[ToolRegistry] All services have errors. Attempting to restart MCP server...")
                    try:
                        import subprocess, time
                        # Kill existing server
                        subprocess.run(["pkill", "-f", "fractalic_mcp_manager_v2"], check=False)
                        time.sleep(2)
                        
                        # Start new server with increased timeout
                        proc = subprocess.Popen([
                            "python3", 
                            "fractalic_mcp_manager_v2.py", 
                            "serve",
                            "--port", "5859"
                        ])
                        
                        # Wait for server to start
                        time.sleep(5)
                        
                        # Try again with increased timeout
                        response = mcp_list(srv)
                        if not response or all("error" in service_data for service_data in response.values()):
                            print(f"[ToolRegistry] Server restart failed. Last error: {response}")
                            continue
                            
                    except Exception as restart_err:
                        print(f"[ToolRegistry] Failed to restart MCP server: {restart_err}")
                        continue
                
                # Process tools from each service
                if isinstance(response, dict):
                    for service_name, service_data in response.items():
                        if isinstance(service_data, dict) and "error" in service_data:
                            print(f"[ToolRegistry] Error in service {service_name}: {service_data['error']}")
                            continue
                            
                        if not isinstance(service_data, dict) or "tools" not in service_data:
                            print(f"[ToolRegistry] Invalid service data format for {service_name}: {service_data}")
                            continue
                            
                        tools = service_data.get("tools", [])
                        if not tools:
                            print(f"[ToolRegistry] No tools found for service {service_name}")
                            continue
                            
                        print(f"[ToolRegistry] Processing {len(tools)} tools from service: {service_name}")
                        for tool in tools:
                            if "name" not in tool:
                                print(f"[ToolRegistry] Tool missing name: {tool}")
                                continue
                                
                            print(f"[ToolRegistry] Registering MCP tool manifest: {tool}")
                            tool["_mcp"] = srv
                            tool["_service"] = service_name
                            self._register(tool, from_mcp=True)
                            print(f"[ToolRegistry] Registered MCP tool: {tool.get('name')} from {srv} ({service_name})")
                else:
                    print(f"[ToolRegistry] Invalid response format from {srv}: {type(response)}")
                    
            except Exception as e:
                print(f"[ToolRegistry] Error loading MCP server {srv}: {e}", file=sys.stderr)

    def _register(self, meta: Dict[str, Any],
                  explicit=False, runner_override: Callable | None = None,
                  from_mcp=False):
        name = meta["name"]
        # Only print a summary list of tool names after all registration is done
        if not hasattr(self, '_tool_names'):  # Track tool names for summary
            self._tool_names = []
        self._tool_names.append(name)

        if name in self:
            if explicit and self[name].__dict__.get("_auto"):
                pass
            else:
                print(f"[ToolRegistry] Tool '{name}' already registered, skipping")
                return

        # Special handling for MCP tools (no 'entry' field, use mcp_call)
        if from_mcp:
            srv = meta.get("_mcp") or meta.get("mcp_server")
            if not srv:
                print(f"[ToolRegistry] MCP tool '{name}' missing server information, skipping")
                return
                
            # Set up the runner function to call the MCP server
            runner = lambda **kw: mcp_call(srv, name, kw)
            self[name] = runner
            
            # Patch manifest to OpenAI schema style if needed
            # Use 'inputSchema' as 'parameters' if present
            if "inputSchema" in meta and "parameters" not in meta:
                meta["parameters"] = meta["inputSchema"]
                
            # Ensure parameters exists and has the right structure
            if "parameters" not in meta or not isinstance(meta["parameters"], dict):
                print(f"[ToolRegistry] MCP tool '{name}' missing valid parameters schema, creating empty schema")
                meta["parameters"] = {"type": "object", "properties": {}, "required": []}
                
            # Add to manifests list so it appears in the schema sent to the LLM
            self._manifests.append(meta)
            return

        cmd = meta.get("command", "python")
        if runner_override:
            runner = runner_override

        elif cmd == "simple-json":
            # Simple JSON in/out tool
            path = Path(meta["entry"])
            def simple_json_runner(**kw):
                json_input = json.dumps(kw)
                env = None
                if Config.TOML_SETTINGS and 'environment' in Config.TOML_SETTINGS:
                    env = os.environ.copy()
                    for item in Config.TOML_SETTINGS['environment']:
                        if 'key' in item and 'value' in item:
                            env[item['key']] = item['value']
                
                result = subprocess.run(
                    [sys.executable, str(path), json_input],
                    capture_output=True, text=True, env=env, timeout=30
                )
                if result.returncode != 0:
                    try:
                        error_data = json.loads(result.stderr)
                        raise RuntimeError(json.dumps(error_data))
                    except json.JSONDecodeError:
                        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
                
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {"result": result.stdout.strip()}
            runner = simple_json_runner

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

        # At the end of rescan, print summary if this is the last tool
        if hasattr(self, '_tool_names') and len(self._tool_names) == len(self._manifests):
            print(f"[ToolRegistry] Discovered tools: {', '.join(self._tool_names)}")
            del self._tool_names
