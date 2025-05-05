#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Better CLI introspection:

 • Captures an argparse.ArgumentParser even when the script keeps it local.
 • Skips the automatic -h / --help flag.
 • Detects `required=True` arguments.
 • Still falls back to GNU-style help-text regex for shell scripts.

Contract: the script must print *something* on “--help”, but for Python we
                                    don’t need to parse it anymore.
"""
from __future__ import annotations
import json, re, subprocess, sys, runpy, types, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any
import os
from core.config import Config

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
