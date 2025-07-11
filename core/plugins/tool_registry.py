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
import uuid

from .cli_introspect import sniff as sniff_cli
from .mcp_client import list_tools as mcp_list, call_tool as mcp_call
from core.config import Config

# ─────────────────────────── Tool Execution Configuration ──────────────────────────
TOOL_EXECUTION_TIMEOUT = None  # Default: No timeout. Set to number of seconds for timeout.
# Examples:
# TOOL_EXECUTION_TIMEOUT = None   # No timeout (default)
# TOOL_EXECUTION_TIMEOUT = 30     # 30 second timeout
# TOOL_EXECUTION_TIMEOUT = 300    # 5 minute timeout

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
        # Store current execution context for fractalic_run tool
        self._current_ast = None
        self._current_file = None
        self._current_call_tree_node = None
        self._committed_files = None
        self._file_commit_hashes = None
        self._base_dir = None
        self._tool_loop_ast = None
        self.rescan()

    def rescan(self):
        self.clear()
        self._manifests.clear()
        self._load_yaml_manifests()
        self._autodiscover_cli()
        self._load_mcp()
        self._register_builtin_tools()

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
                
                # Set working directory to the tool's directory
                tool_dir = path.parent
                result = subprocess.run(
                    [sys.executable, str(path), json_input],
                    capture_output=True, text=True, env=env, timeout=TOOL_EXECUTION_TIMEOUT, cwd=str(tool_dir)
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
                    # print(f"DEBUG: Injecting env for {name}: { {k: ('***' if 'KEY' in k.upper() else v) for k, v in env.items() if k == 'TAVILY_API_KEY'} }", file=sys.stderr)
                    # --- END DEBUG PRINT ---
                else:
                    # --- BEGIN DEBUG PRINT ---
                    # print(f"DEBUG: NOT Injecting env for {name}. Config.TOML_SETTINGS: {Config.TOML_SETTINGS is not None}, 'environment' in settings: {'environment' in Config.TOML_SETTINGS if Config.TOML_SETTINGS else 'N/A'}", file=sys.stderr)
                    # --- END DEBUG PRINT ---
                    pass

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
    
    def set_execution_context(self, ast, current_file, call_tree_node, committed_files=None, file_commit_hashes=None, base_dir=None, tool_loop_ast=None, current_node=None):
        """Set current execution context for built-in tools like fractalic_run."""
        self._current_ast = ast
        self._current_file = current_file
        self._current_call_tree_node = call_tree_node
        self._committed_files = committed_files or set()
        self._file_commit_hashes = file_commit_hashes or {}
        self._base_dir = base_dir
        self._tool_loop_ast = tool_loop_ast
        self._current_node = current_node
    
    def _register_builtin_tools(self):
        """Register built-in tools like fractalic_run."""
        # Register fractalic_run tool
        fractalic_run_manifest = {
            "name": "fractalic_run",
            "description": "Execute a Fractalic script within current context",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string", 
                        "description": "Path to .md file to execute"
                    },
                    "prompt": {
                        "type": "string", 
                        "description": "Optional prompt text to prepend to execution"
                    },
                    "block_uri": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ], 
                        "description": "Block reference(s) to include from current context. Supports wildcards like 'section/*'"
                    },
                    "mode": {
                        "type": "string", 
                        "enum": ["append", "prepend", "replace"], 
                        "default": "append",
                        "description": "How to insert results back into context"
                    }
                },
                "required": ["file_path"]
            },
            "_builtin": True
        }
        
        self._register(fractalic_run_manifest, runner_override=self._handle_fractalic_run)
    
    def _handle_fractalic_run(self, **kwargs):
        """Handle fractalic_run tool calls."""
        try:
            # Import here to avoid circular imports
            from core.operations.runner import run
            from core.ast_md.node import Node, NodeType
            from core.ast_md.ast import get_ast_part_by_path, AST
            
            # Check if we have execution context
            if not self._current_ast or not self._current_file or not self._current_call_tree_node:
                return {
                    "error": "fractalic_run tool requires execution context to be set",
                    "status": "failed"
                }
            
            # Extract parameters
            file_path = kwargs.get("file_path")
            prompt = kwargs.get("prompt")
            block_uri = kwargs.get("block_uri")
            mode = kwargs.get("mode", "append")
            
            if not file_path:
                return {
                    "error": "file_path parameter is required",
                    "status": "failed"
                }
            
            # Resolve file path relative to current working directory
            import os
            if not os.path.isabs(file_path):
                # Make file path relative to current working directory
                file_path = os.path.join(os.getcwd(), file_path)
            
            if not os.path.exists(file_path):
                return {
                    "error": f"File not found: {file_path}",
                    "status": "failed"
                }
            
            # Build input AST if prompt or block_uri provided
            input_ast = None
            if prompt or block_uri:
                input_ast = AST("")
                
                if block_uri:
                    # Handle both string and array block_uri
                    try:
                        if isinstance(block_uri, list):
                            # Import the new function for array handling
                            from core.ast_md.ast import get_ast_parts_by_uri_array
                            block_ast = get_ast_parts_by_uri_array(self._current_ast, block_uri, use_hierarchy=any(uri.endswith("/*") for uri in block_uri), tool_loop_ast=self._tool_loop_ast)
                        else:
                            # Single string block_uri (existing behavior)
                            block_ast = get_ast_part_by_path(self._current_ast, block_uri, block_uri.endswith("/*"), tool_loop_ast=self._tool_loop_ast)
                        
                        # Update attribution for all nodes from block_uri to the parent @llm operation
                        if block_ast and block_ast.parser.nodes:
                            for node in block_ast.parser.nodes.values():
                                node.created_by = getattr(self._current_node, 'key', None)
                                node.created_by_file = getattr(self._current_node, 'created_by_file', None)
                        
                        input_ast = block_ast
                    except Exception as e:
                        return {
                            "error": f"Block reference '{block_uri}' not found: {str(e)}",
                            "status": "failed"
                        }
                
                if prompt:
                    # Create a prompt node with attribution to the parent @llm operation
                    prompt_node = Node(
                        type=NodeType.HEADING,
                        name="Input Parameters",
                        level=1,
                        content=f"# Input Parameters\n{prompt}",
                        id="InputParameters",
                        key=str(uuid.uuid4())[:8],
                        created_by=getattr(self._current_node, 'key', None),
                        created_by_file=getattr(self._current_node, 'created_by_file', None)
                    )
                    
                    if input_ast and input_ast.parser.nodes:
                        # Append prompt to existing blocks
                        from core.ast_md.ast import perform_ast_operation
                        from core.ast_md.node import OperationType
                        prompt_ast = AST("")
                        prompt_ast.parser.nodes = {prompt_node.key: prompt_node}
                        prompt_ast.parser.head = prompt_node
                        prompt_ast.parser.tail = prompt_node
                        
                        perform_ast_operation(
                            src_ast=prompt_ast,
                            src_path='',
                            src_hierarchy=False,
                            dest_ast=input_ast,
                            dest_path=input_ast.parser.tail.key,
                            dest_hierarchy=False,
                            operation=OperationType.APPEND
                        )
                    else:
                        # Use prompt as input
                        input_ast = AST("")
                        input_ast.parser.nodes = {prompt_node.key: prompt_node}
                        input_ast.parser.head = prompt_node
                        input_ast.parser.tail = prompt_node
            
            # Call run function directly to avoid AST insertion issues
            run_result, child_call_tree_node, ctx_file, ctx_hash, trc_file, trc_hash, branch_name, explicit_return = run(
                filename=file_path,
                param_node=input_ast,
                create_new_branch=False,  # Don't create new branch for tool execution
                p_parent_filename=self._current_file,
                p_parent_operation="fractalic_run",
                p_call_tree_node=self._current_call_tree_node,
                committed_files=self._committed_files,
                file_commit_hashes=self._file_commit_hashes,
                base_dir=self._base_dir
            )
            
            # Format response for tool calling interface
            response = {
                "status": "success",
                "explicit_return": explicit_return,
                "trace_info": {
                    "ctx_file": ctx_file,
                    "ctx_hash": ctx_hash,
                    "trc_file": trc_file,
                    "trc_hash": trc_hash,
                    "branch_name": branch_name
                }
            }
            
            # If there's an explicit return, try to extract the content
            if explicit_return and run_result:
                # Find the return content by looking for nodes with return results
                return_content = ""
                return_nodes_attribution = []
                
                for node in run_result.parser.nodes.values():
                    if hasattr(node, 'content') and node.content and '@return' not in node.content:
                        return_content += node.content + "\n"
                        
                        # Capture attribution metadata for later restoration including content for robust matching
                        return_nodes_attribution.append({
                            "created_by": getattr(node, 'created_by', None),
                            "created_by_file": getattr(node, 'created_by_file', None),
                            "node_id": getattr(node, 'id', None),
                            "node_key": getattr(node, 'key', None),
                            "content": node.content,  # Include full content for robust content-based matching
                            "content_hash": node.hash,  # Include content hash for fallback matching
                            "content_length": len(node.content) if node.content else 0
                        })
                
                response["return_content"] = return_content.strip()
                response["return_nodes_attribution"] = return_nodes_attribution
            else:
                response["message"] = "Script executed successfully"
            
            return response
            
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "status": "failed",
                "traceback": traceback.format_exc()
            }
    
    def _build_run_params(self, file_path, prompt=None, block_uri=None, mode="append"):
        """Build parameters dictionary in format expected by process_run."""
        # Parse file path to get directory and filename
        path_obj = Path(file_path)
        if path_obj.is_absolute():
            # For absolute paths, get directory and filename
            file_dir = str(path_obj.parent)
            filename = path_obj.name
        else:
            # For relative paths, assume current directory
            file_dir = "."
            filename = file_path
        
        params = {
            "file": {
                "path": file_dir,
                "file": filename
            },
            "mode": mode
        }
        
        # Add prompt if provided
        if prompt:
            params["prompt"] = prompt
        
        # Add block reference if provided  
        if block_uri:
            params["block"] = {
                "block_uri": block_uri,
                "nested_flag": block_uri.endswith("/*") if block_uri else False
            }
        
        return params
    
    def _format_tool_response(self, result):
        """Format process_run result for tool calling interface."""
        if result is None:
            return {
                "status": "success",
                "message": "Script executed successfully",
                "explicit_return": False
            }
        
        # result is a tuple: (next_node, call_tree_node, ctx_file, ctx_hash, trc_file, trc_hash, branch_name, explicit_return)
        if isinstance(result, tuple) and len(result) >= 8:
            next_node, call_tree_node, ctx_file, ctx_hash, trc_file, trc_hash, branch_name, explicit_return = result
            
            response = {
                "status": "success",
                "explicit_return": explicit_return,
                "trace_info": {
                    "ctx_file": ctx_file,
                    "ctx_hash": ctx_hash,
                    "trc_file": trc_file,
                    "trc_hash": trc_hash,
                    "branch_name": branch_name
                }
            }
            
            # If there's a return result, extract the content
            if explicit_return and next_node and hasattr(next_node, 'content'):
                response["return_content"] = next_node.content
            else:
                response["message"] = "Script executed successfully"
            
            return response
        else:
            return {
                "status": "success", 
                "message": "Script executed successfully",
                "explicit_return": False
            }
