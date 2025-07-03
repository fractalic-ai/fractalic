#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Better CLI introspection:

 • Captures an argparse.ArgumentParser even when the script keeps it local.
 • Skips the automatic -h / --help flag.
 • Detects `required=True` arguments.
 • Still falls back to GNU-style help-text regex for shell scripts.

Contract: the script must print *something* on "--help", but for Python we
                                    don't need to parse it anymore.
"""
from __future__ import annotations
import json, re, subprocess, sys, runpy, types, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any
import os
from core.config import Config

# ─────────────────────────── Tool Execution Configuration ──────────────────────────
TOOL_EXECUTION_TIMEOUT = None  # Default: No timeout. Set to number of seconds for timeout.
# Examples:
# TOOL_EXECUTION_TIMEOUT = None   # No timeout (default)
# TOOL_EXECUTION_TIMEOUT = 30     # 30 second timeout  
# TOOL_EXECUTION_TIMEOUT = 300    # 5 minute timeout

import subprocess
import json
import sys
import argparse
import re
from typing import Dict, Any, Optional, Tuple, List

# --- regex fallback (for bash or non-argparse pythons) ----------------
HELP_RE = re.compile(
    r"^\s*(?:-\w,\s*)?(--[\w-]+)(?:\s+([A-Z\[\]<>\w-]+))?", re.MULTILINE
)

def _extract_description_from_file(file_path: Path) -> str:
    """Extract description from Python file docstring or comments"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Look for module docstring
        in_docstring = False
        docstring_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and shebang
            if not line or line.startswith('#!'):
                continue
            
            # Start of docstring
            if line.startswith('"""') or line.startswith("'''"):
                if line.count('"""') == 2 or line.count("'''") == 2:
                    # Single line docstring
                    return line.strip('"""').strip("'''").strip()
                in_docstring = True
                docstring_lines.append(line[3:])
                continue
            
            # End of docstring
            if in_docstring and (line.endswith('"""') or line.endswith("'''")):
                docstring_lines.append(line[:-3])
                return ' '.join(docstring_lines).strip()
            
            # Inside docstring
            if in_docstring:
                docstring_lines.append(line)
                continue
            
            # Look for comment description
            if line.startswith('#') and 'description:' in line.lower():
                return line.split(':', 1)[1].strip()
            
            # Stop at first non-comment, non-docstring line
            if not line.startswith('#'):
                break
                
    except Exception:
        pass
    
    return f"Simple tool: {file_path.stem}"

def _from_help_text(txt: str) -> Tuple[Dict[str, Any], List[str], str]:
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    desc = next((l for l in lines if not l.lower().startswith("usage")), "(no description)")
    props, req = {}, []
    for flag, arg in HELP_RE.findall(txt):
        key = flag.lstrip("-")
        if key == "help":
            continue
        props[key] = {"type": "boolean" if not arg else "string"}
    return props, req, desc

# --- argparse capture -------------------------------------------------
def _capture_argparse(path: Path) -> argparse.ArgumentParser | None:
    captured: dict[str, argparse.ArgumentParser] = {}
    real_parse = argparse.ArgumentParser.parse_args

    def fake_parse(self, *a, **kw):
        captured["parser"] = self
        # ensure the script exits early
        raise SystemExit

    argparse.ArgumentParser.parse_args = fake_parse          # type: ignore
    sys.argv = [str(path), "--help"]
    try:
        runpy.run_path(str(path))
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse      # restore
    return captured.get("parser")

def _schema_from_parser(p: argparse.ArgumentParser):
    props, req = {}, []
    for act in p._actions:                                   # pylint: disable=protected-access
        if any(f in {"-h", "--help"} for f in act.option_strings):
            continue
        name = act.option_strings[-1].lstrip("-")
        entry = {"type": "boolean" if act.nargs == 0 else "string"}
        if act.help:
            entry["description"] = act.help
        props[name] = entry
        if getattr(act, "required", False):
            req.append(name)
    return props, req, p.description or "(no description)"

# --- build runner -----------------------------------------------------
def _make_runner(exec_prefix: List[str]) -> Callable:
    def run_cli(**kw):
        argv = exec_prefix[:]
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
            env=env  # <-- pass through injected env
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or out.stdout.strip())
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            return {"stdout": out.stdout.strip()}
    return run_cli

# --- public entry -----------------------------------------------------
def sniff(path: Path, kind: str):
    """
    Simplified tool discovery - only use simple JSON schema discovery
    kind : python-cli | bash-cli
    returns (schema, description, runner) or None if not a valid tool
    """
    path = path.expanduser().absolute()
    
    if kind == "python-cli":
        # Only try simple JSON in/out convention
        try:
            # Test if tool accepts JSON input and returns JSON output
            test_input = '{"__test__": true}'
            res = subprocess.run(
                [sys.executable, str(path), test_input],
                capture_output=True, text=True, timeout=0.2, check=False
            )
            
            if res.returncode == 0 and res.stdout:
                try:
                    # Check if output is valid JSON
                    json.loads(res.stdout)
                    
                    # Try to get detailed schema if available
                    schema = None
                    desc = None
                    
                    # Try schema dump first for detailed schema
                    try:
                        schema_res = subprocess.run(
                            [sys.executable, str(path), "--fractalic-dump-schema"],
                            capture_output=True, text=True, timeout=0.2, check=False
                        )
                        if schema_res.returncode == 0 and schema_res.stdout:
                            schema_data = json.loads(schema_res.stdout)
                            if isinstance(schema_data, dict) and "parameters" in schema_data:
                                schema = schema_data["parameters"]
                                desc = schema_data.get("description", "Simple JSON tool")
                                # print(f"[CLI Introspect] Got detailed schema for simple JSON tool: {len(schema.get('properties', {}))} properties")
                    except (json.JSONDecodeError, Exception):
                        pass
                    
                    # Fallback to generic schema if detailed schema not available
                    if not schema:
                        desc = _extract_description_from_file(path)
                        schema = {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": True,
                            "description": "Simple JSON tool - accepts JSON input, returns JSON output"
                        }
                    
                    # Create a simple runner that passes JSON directly
                    def simple_runner(**kw):
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
                    
                    return schema, desc, simple_runner
                    
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        
        # If simple JSON discovery fails, this is not a valid tool
        return None, None, None
        
    # For bash-cli, we don't support it in simplified mode
    return None, None, None

# --- Constants ---
SCHEMA_DUMP_FLAG = "--fractalic-dump-schema"

# --- Improved Argparse Introspection (Fallback) ---
def _get_type_from_action(action: argparse.Action) -> str:
    """Determine JSON schema type from argparse action."""
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "boolean"
    if action.type is int:
        return "integer"
    if action.type is float:
        return "number"
    # TODO: Add support for 'choices' -> enum?
    # Default to string for others (str, Path, etc.)
    return "string"

# --- Main Introspection Function ---
def introspect_script(script_path: str) -> Optional[Dict[str, Any]]:
    """
    Introspects a Python script to generate an OpenAI tool schema.
    Uses the updated sniff function with simple JSON convention support.
    """
    try:
        path = Path(script_path)
        
        # Use the updated sniff function with simple JSON support
        result = sniff(path, "python-cli")
        
        # Handle different return types from sniff
        if isinstance(result, tuple) and len(result) == 3:
            schema, description, runner = result
            
            # For simple JSON tools, mark them as such
            if schema and schema.get("description") == "Simple JSON tool - accepts JSON input, returns JSON output":
                tool_name = path.stem
                return {
                    "name": tool_name,
                    "description": description or f"Simple JSON tool: {tool_name}",
                    "command": "simple-json",
                    "parameters": schema,
                    "_simple": True
                }
            else:
                # Regular tool
                tool_name = path.stem
                return {
                    "name": tool_name,
                    "description": description or f"Tool: {tool_name}",
                    "command": "python-cli",
                    "parameters": schema
                }
        elif isinstance(result, list):
            # Multi-tool result
            return result
        else:
            print(f"[ERROR] cli_introspect: Unexpected result type from sniff for {script_path}", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"[ERROR] cli_introspect: Failed to introspect {script_path}: {e}", file=sys.stderr)
        return None
