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

# Import sanitization function for Gemini compatibility
def _sanitize_schema_for_gemini(schema: dict, max_depth: int = 6, current_depth: int = 0) -> dict:
    """Sanitize JSON schema for Gemini/Vertex AI compatibility."""
    if current_depth >= max_depth:
        return {"type": "string", "description": "Complex nested data (simplified for compatibility)"}
    
    if not isinstance(schema, dict):
        return schema
    
    sanitized = {}
    
    for key, value in schema.items():
        if key == "type" and isinstance(value, list):
            # Convert array types to single type - use first non-null type
            non_null_types = [t for t in value if t != "null"]
            sanitized[key] = non_null_types[0] if non_null_types else "string"
        elif key == "format":
            # Remove unsupported format fields for Vertex AI
            if value in ["enum", "date-time"]:
                sanitized[key] = value
            # Skip unsupported formats by not adding them
        elif key in ["anyOf", "oneOf"]:
            # Replace with first option or fallback to string
            if isinstance(value, list) and value:
                first_option = value[0]
                if isinstance(first_option, dict):
                    sanitized.update(_sanitize_schema_for_gemini(first_option, max_depth, current_depth))
                    continue
            sanitized.update({"type": "string", "description": "Union type (simplified for compatibility)"})
            continue
        elif key == "properties" and isinstance(value, dict):
            sanitized[key] = {}
            for prop_name, prop_schema in value.items():
                sanitized[key][prop_name] = _sanitize_schema_for_gemini(prop_schema, max_depth, current_depth + 1)
        elif key == "items" and isinstance(value, dict):
            sanitized[key] = _sanitize_schema_for_gemini(value, max_depth, current_depth + 1)
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_schema_for_gemini(value, max_depth, current_depth + 1)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_schema_for_gemini(item, max_depth, current_depth + 1) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized

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
                
            # Get the parameters and apply Gemini sanitization if needed
            parameters = m.get("parameters", {"type": "object", "properties": {}})
            
            # Check if we're using a Gemini provider and sanitize accordingly
            if hasattr(Config, 'LLM_PROVIDER') and Config.LLM_PROVIDER and 'gemini' in Config.LLM_PROVIDER.lower():
                parameters = _sanitize_schema_for_gemini(parameters)
            
            # Create the function schema
            function_schema = {
                "name": m["name"],
                "description": m.get("description", ""),
                "parameters": parameters
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
            result = sniff_cli(src, cmd_type)
            
            # If sniff_cli returns None, skip this file (not a valid tool)
            if result == (None, None, None) or result is None:
                continue
                
            schema, desc, runner = result
            
            # If we got a valid result, register the tool
            if schema and desc and runner:
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

    def _load_mcp(self):
        # print(f"[ToolRegistry] MCP servers to load: {self.mcp_servers}")
        for srv in self.mcp_servers:
            try:
                # print(f"[ToolRegistry] Attempting to load tools from {srv}")
                response = mcp_list(srv)
                # print(f"[ToolRegistry] MCP {srv} raw response: {response}")
                
                if not response:
                    # print(f"[ToolRegistry] MCP {srv} returned empty or None response.")
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
                            
                        # print(f"[ToolRegistry] Processing {len(tools)} tools from service: {service_name}")
                        for tool in tools:
                            if "name" not in tool:
                                # print(f"[ToolRegistry] Tool missing name: {tool}")
                                continue
                                
                            # print(f"[ToolRegistry] Registering MCP tool manifest: {tool}")
                            tool["_mcp"] = srv
                            tool["_service"] = service_name
                            self._register(tool, from_mcp=True)
                            # print(f"[ToolRegistry] Registered MCP tool: {tool.get('name')} from {srv} ({service_name})")
                else:
                    # print(f"[ToolRegistry] Invalid response format from {srv}: {type(response)}")
                    pass
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
            
            def runner_with_error_handling(**kw):
                try:
                    return runner_with_env(**kw)
                except subprocess.CalledProcessError as e:
                    error_message = f"Tool failed with exit code {e.returncode}.\nStderr: {e.stderr}"
                    logging.error(error_message)
                    return {"error": error_message, "stderr": e.stderr}
                except FileNotFoundError:
                    error_message = f"Tool executable not found at {path}."
                    logging.error(error_message)
                    return {"error": error_message}
                except Exception as e:
                    error_message = f"Unexpected tool execution error: {str(e)}"
                    logging.error(error_message)
                    return {"error": error_message}
            
            runner = runner_with_error_handling

        self[name] = runner
        self._manifests.append(meta)

        # At the end of rescan, print summary if this is the last tool
        if hasattr(self, '_tool_names') and len(self._tool_names) == len(self._manifests):
            # print(f"[ToolRegistry] Discovered tools: {', '.join(self._tool_names)}")
            del self._tool_names
