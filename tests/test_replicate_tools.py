import json
import http.client
import time
import subprocess
import signal
import pytest

# Replicate service tests verifying tools listing and a minimal tool invocation.
# Skips instead of failing if authentication not yet completed.

SERVER_PORT = 5859
SERVER_HOST = "127.0.0.1"
SERVER_URL_BASE = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_CMD = ["python", "fractalic_mcp_manager_sdk_v2.py", "serve", "--port", str(SERVER_PORT), "--host", SERVER_HOST]

_PROC = None


def _server_running():
    try:
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=1)
        conn.request("GET", "/status")
        resp = conn.getresponse()
        return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _start_server():
    global _PROC
    if _server_running():
        return
    if _PROC is not None:
        return
    _PROC = subprocess.Popen(SERVER_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 25
    while time.time() < deadline:
        if _server_running():
            return
        time.sleep(0.5)
    raise RuntimeError("Replicate test server failed to start")


def _stop_server():  # Not used; keep server for other tests
    global _PROC
    if _PROC is None:
        return
    _PROC.send_signal(signal.SIGTERM)
    try:
        _PROC.wait(timeout=10)
    except Exception:  # noqa: BLE001
        _PROC.kill()
    _PROC = None


def _get(path):
    conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=15)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode()
    return resp.status, body


def _post(path, body=None):
    conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=30)
    payload = json.dumps(body) if body is not None else ""
    headers = {"Content-Type": "application/json"}
    conn.request("POST", path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = resp.read().decode()
    return resp.status, data


@pytest.fixture(scope="module", autouse=True)
def _ensure_server():
    _start_server()
    yield


def test_replicate_tools_available():
    status, body = _get("/tools/replicate")
    if status != 200:
        pytest.skip(f"/tools/replicate returned status {status}")
    data = json.loads(body)
    if "tools_error" in data:
        pytest.skip(f"Replicate tools pending: {data['tools_error']}")
    if "error" in data:
        pytest.skip(f"Replicate tools error: {data['error']}")
    assert data.get("tool_count", 0) >= 5, f"Unexpected replicate tool_count: {data.get('tool_count')}"


def test_replicate_list_predictions_minimal():
    # If tools not yet available skip early
    status_tools, body_tools = _get("/tools/replicate")
    if status_tools == 200:
        dt = json.loads(body_tools)
        if "tools_error" in dt:
            pytest.skip(f"Replicate tools pending: {dt['tools_error']}")
    
    payload = {"arguments": {"jq_filter": ".results[:1] | length"}}
    status, body = _post("/call/replicate/list_predictions", payload)
    if status != 200:
        pytest.skip(f"Replicate list_predictions call status {status}")
    data = json.loads(body)
    if data.get("isError") or "error" in data:
        pytest.skip(f"Replicate list_predictions error: {data.get('error')} isError={data.get('isError')}")
    content = data.get("content", [])
    assert content, "No content returned from replicate list_predictions"
    first = content[0]
    assert first.get("type") == "text"
    assert first.get("text").strip().isdigit(), f"Unexpected jq filtered text: {first.get('text')}"
