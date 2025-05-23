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
    kind : python-cli | bash-cli
    returns (schema, description, runner)
    """
    path = path.expanduser().absolute()
    if kind == "python-cli":
        # 1) Try multi-schema dump
        try:
            res = subprocess.run(
                [sys.executable, str(path), "--fractalic-dump-multi-schema"],
                capture_output=True, text=True, timeout=3, check=False
            )
            if res.returncode == 0 and res.stdout:
                try:
                    parsed = json.loads(res.stdout)
                    if isinstance(parsed, list):
                        # Return the list of manifests (multi-tool)
                        return parsed, None, None
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        # 2) Fallback: single schema dump
        try:
            res = subprocess.run(
                [sys.executable, str(path), SCHEMA_DUMP_FLAG],
                capture_output=True, text=True, timeout=3, check=False
            )
            if res.returncode == 0 and res.stdout:
                try:
                    parsed = json.loads(res.stdout)
                    if isinstance(parsed, list):
                        # Return the list of manifests (multi-tool)
                        return parsed, None, None
                    params = parsed.get("parameters") if isinstance(parsed, dict) else None
                    desc = parsed.get("description") if isinstance(parsed, dict) else None
                    if params and desc is not None:
                        runner = _make_runner([sys.executable, str(path)])
                        return params, desc, runner
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        # 3) Fallback: argparse capture via --help
        parser = _capture_argparse(path)
        if parser:
            props, req, desc = _schema_from_parser(parser)
        else:
            help_txt = subprocess.run([sys.executable, str(path), "--help"],
                                      capture_output=True, text=True).stdout
            props, req, desc = _from_help_text(help_txt)
        schema = {"type": "object", "properties": props, "required": req}
        runner = _make_runner([sys.executable, str(path)])
        return schema, desc, runner
    # ---------- bash-cli -------------
    help_txt = subprocess.run([str(path), "--help"],
                              capture_output=True, text=True).stdout
    props, req, desc = _from_help_text(help_txt)
    schema = {"type": "object", "properties": props, "required": req}
    runner = _make_runner([str(path)])
    return schema, desc, runner

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

    1. Tries executing script with SCHEMA_DUMP_FLAG to get JSON schema directly.
    2. Falls back to executing with --help and parsing ArgumentParser.
    3. (Optional) Further fallback to regex parsing of --help text.
    """
    # 1. Try direct schema dump
    try:
        result = subprocess.run(
            [sys.executable, script_path, SCHEMA_DUMP_FLAG],
            capture_output=True,
            text=True,
            timeout=5, # Add a timeout
            check=False, # Don't raise exception on non-zero exit
        )
        if result.returncode == 0 and result.stdout:
            try:
                schema = json.loads(result.stdout)
                # If schema is a list, it's a multi-tool schema
                if isinstance(schema, list):
                    return schema
                # Basic validation: check for top-level keys
                if isinstance(schema, dict) and "name" in schema and "parameters" in schema:
                     # Wrap it in the standard function structure
                     return {
                         "type": "function",
                         "function": schema
                     }
                else:
                     print(f"[WARN] cli_introspect: Invalid schema structure from {script_path} {SCHEMA_DUMP_FLAG}", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"[WARN] cli_introspect: Failed to parse JSON from {script_path} {SCHEMA_DUMP_FLAG}: {e}", file=sys.stderr)
                # Optionally print result.stdout[:500] for debugging
        # else: # Optional: Log if dump command failed
        #    print(f"[DEBUG] cli_introspect: Schema dump failed for {script_path}. Exit={result.returncode}, Stderr={result.stderr.strip()}", file=sys.stderr)

    except FileNotFoundError:
         print(f"[ERROR] cli_introspect: Python executable not found at {sys.executable}", file=sys.stderr)
         return None
    except subprocess.TimeoutExpired:
         print(f"[WARN] cli_introspect: Timeout executing {script_path} {SCHEMA_DUMP_FLAG}", file=sys.stderr)
    except Exception as e:
         print(f"[WARN] cli_introspect: Error during schema dump for {script_path}: {e}", file=sys.stderr)


    # 2. Fallback: Capture ArgumentParser via --help
    print(f"[DEBUG] cli_introspect: Falling back to argparse capture for {script_path}", file=sys.stderr)
    parser = _capture_argparse(script_path)
    if parser:
        try:
            return _schema_from_parser(parser)
        except Exception as e:
            print(f"[WARN] cli_introspect: Error building schema from parser for {script_path}: {e}", file=sys.stderr)

    # 3. Fallback: Regex parsing (if _from_help_text exists and is desired)
    # print(f"[DEBUG] cli_introspect: Falling back to regex help parsing for {script_path}", file=sys.stderr)
    # help_text = _get_help_text(script_path) # Assuming _get_help_text exists
    # if help_text:
    #     return _from_help_text(script_path, help_text) # Assuming _from_help_text exists

    print(f"[ERROR] cli_introspect: Failed to generate schema for {script_path}", file=sys.stderr)
    return None
