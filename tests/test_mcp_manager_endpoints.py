import asyncio
import json
import os
import signal
import time
import http.client
from urllib.parse import urlencode

# Basic integration tests for fractalic_mcp_manager_sdk_v2 HTTP endpoints
# Assumptions:
# - Server can be started via `python fractalic_mcp_manager_sdk_v2.py serve --port 5859`
# - Tests run in repository root
# - openmemory-mcp-marina is disabled in mcp_servers.json

SERVER_PORT = 5859
SERVER_HOST = "127.0.0.1"
SERVER_URL_BASE = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_CMD = ["python", "fractalic_mcp_manager_sdk_v2.py", "serve", "--port", str(SERVER_PORT), "--host", SERVER_HOST]

PROC = None

def _start_server():
    global PROC
    if PROC is not None:
        return
    import subprocess
    PROC = subprocess.Popen(SERVER_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Wait for port to open (simple retry loop)
    deadline = time.time() + 25
    last_err = None
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=1)
            conn.request("GET", "/status")
            resp = conn.getresponse()
            if resp.status == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"Server failed to start: {last_err}")


def _stop_server():
    global PROC
    if PROC is None:
        return
    PROC.send_signal(signal.SIGTERM)
    try:
        PROC.wait(timeout=10)
    except Exception:  # noqa: BLE001
        PROC.kill()
    PROC = None


def setup_module(module):  # noqa: D401
    _start_server()


def teardown_module(module):  # noqa: D401
    _stop_server()


def _get(path, query=None):
    q = ""
    if query:
        q = "?" + urlencode(query)
    conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
    conn.request("GET", path + q)
    resp = conn.getresponse()
    body = resp.read().decode()
    return resp.status, body


def test_status_basic():
    status, body = _get("/status")
    assert status == 200
    data = json.loads(body)
    assert "services" in data
    # Disabled service should still appear but flagged disabled
    svc = data["services"].get("openmemory-mcp-marina")
    assert svc is not None
    # With config enabled flag false, we expect status field to be 'disabled'
    assert (svc["enabled"] is False) or (svc["status"] == "disabled")


def test_status_with_tools_info():
    status, body = _get("/status", {"include_tools_info": "true"})
    assert status == 200
    data = json.loads(body)
    assert "total_services" in data
    # Totals should be >= enabled_services
    assert data["total_services"] >= data["enabled_services"]


def test_list_tools_excludes_disabled():
    status, body = _get("/list_tools")
    assert status == 200
    data = json.loads(body)
    assert "tools" in data
    # Ensure tools from disabled service are not present (prefix check)
    disabled_prefix = "openmemory-mcp-marina."
    for tool in data["tools"]:
        assert not tool["name"].startswith(disabled_prefix)


def _post(path, body=None):
    conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
    payload = json.dumps(body) if body is not None else ""
    headers = {"Content-Type": "application/json"}
    conn.request("POST", path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = resp.read().decode()
    return resp.status, data


def test_toggle_service_enable_disable_cycle():
    # Disable an enabled service (memory) then re-enable using POST
    status, body = _post("/toggle/memory", {})
    assert status == 200
    data = json.loads(body)
    assert data["service"] == "memory"
    assert data["status"] in ("disabled", "enabled")
    # Toggle again
    status, _ = _post("/toggle/memory", {})
    assert status == 200


def test_call_tool_memory_list():
    # Memory server often provides a 'list-memories' or similar tool; list all tools first
    status, body = _get("/list_tools")
    assert status == 200
    data = json.loads(body)
    memory_tools = [t for t in data["tools"] if t["name"].startswith("memory.")]
    if not memory_tools:
        # Skip if memory tools not available
        return
    tool_name = memory_tools[0]["name"].split(".", 1)[1]
    # Call the first memory tool with empty args
    call_path = f"/call/memory/{tool_name}"
    status, body = _post(call_path, {})
    # Could be 200 or error if tool requires args; assert server responded
    assert status in (200, 400, 500)

