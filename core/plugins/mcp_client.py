#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thin wrapper for Model-Context-Protocol discovery/execution.
You can swap this for an official SDK later.
"""
from __future__ import annotations
import requests
from typing import Dict, Any, List

def list_tools(server: str) -> List[Dict[str, Any]]:
    return requests.get(f"{server.rstrip('/')}/list_tools", timeout=5).json()

def call_tool(server: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return requests.post(f"{server.rstrip('/')}/call_tool",
                         json={"name": name, "arguments": args},
                         timeout=30).json()
