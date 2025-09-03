import json
import http.client
import pytest
import time

# Simple replicate test: locate flux-dev via search_models then fetch its schema (input properties only)
# Skips (not fails) if service not ready or auth not completed.

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5859

try:
    # Reuse server starter from existing replicate tools test if available
    from tests.test_replicate_tools import _start_server  # type: ignore
except Exception:  # pragma: no cover
    _start_server = None  # type: ignore


def _server_running():
    try:
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=2)
        conn.request("GET", "/status")
        return conn.getresponse().status == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module", autouse=True)
def _ensure_server():
    if not _server_running() and _start_server:
        _start_server()
        # small wait for tools cache
        time.sleep(2)
    elif not _server_running():
        pytest.skip("MCP manager not running and no starter available")
    yield


def _post(path, body=None, timeout=30):
    conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=timeout)
    payload = json.dumps(body) if body is not None else ""
    headers = {"Content-Type": "application/json"}
    conn.request("POST", path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = resp.read().decode()
    return resp.status, data


def test_flux_dev_schema():
    # 1. search for flux-dev model slug
    search_payload = {"arguments": {"jq_filter": ".results[] | select(.name==\"flux-dev\") | .owner + \"/\" + .name"}}
    status, body = _post("/call/replicate/search_models", search_payload)
    if status != 200:
        pytest.skip(f"search_models status {status}")
    data = json.loads(body)
    if data.get("isError") or "error" in data:
        pytest.skip(f"search_models error: {data.get('error')}")
    content = data.get("content", [])
    if not content:
        pytest.skip("search_models returned no content")
    found_slug = content[0].get("text", "").strip()
    if not found_slug.endswith("/flux-dev"):
        pytest.skip(f"flux-dev not found in search_models output: {found_slug}")
    owner, name = found_slug.split("/", 1)

    # 2. get model schema (input property names only) to keep token usage low
    jq_filter = "{input_properties: (.latest_version.openapi.components.schemas.Input.properties | keys)}"
    get_payload = {"arguments": {"model_owner": owner, "model_name": name, "jq_filter": jq_filter}}
    status2, body2 = _post("/call/replicate/get_models", get_payload)
    if status2 != 200:
        pytest.skip(f"get_models status {status2}")
    data2 = json.loads(body2)
    if data2.get("isError") or "error" in data2:
        pytest.skip(f"get_models error: {data2.get('error')}")
    content2 = data2.get("content", [])
    assert content2, "No content from get_models"
    schema_text = content2[0].get("text", "").strip()
    # Expect JSON with input_properties array
    try:
        schema_obj = json.loads(schema_text)
    except json.JSONDecodeError:
        pytest.fail(f"Schema text not valid JSON: {schema_text[:120]}")
    assert isinstance(schema_obj, dict) and "input_properties" in schema_obj, "input_properties missing in schema object"
    assert isinstance(schema_obj["input_properties"], list) and schema_obj["input_properties"], "input_properties list empty"
