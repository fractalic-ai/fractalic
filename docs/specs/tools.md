# Fractalic – Tool Plug-in & Auto-Discovery Guide

### Contents

1. Why tools are first-class citizens in Fractalic
2. How the registry builds the master tool list
3. Two ways to publish a tool
4. Contract for Python & Bash scripts (auto-discovery mode)
5. End-to-end life-cycle of a tool call
6. Using Model-Context-Protocol (MCP) servers
7. FAQ & troubleshooting

---

## 1.  Why tools are first-class citizens in Fractalic

Fractalic’s workflows are driven by LLM-calls (`@llm`), which can
**delegate subtasks**—run a shell command, fetch data, resize an image, etc.—by
invoking *tools*.
Instead of hard-coding every helper function in Python, we expose them to the
model through the OpenAI “function-calling” / “tools” interface.
Benefits:

* **Plug-ins, not deployments** – drop a script into `tools/`, restart, done.
* **Language-agnostic** – Bash, Python, Go, even remote micro-services can be
  called the same way.
* **Self-documenting** – the registry produces the JSON schema the LLM needs,
  so prompt-engineering stays declarative.

---

## 2.  How the registry builds the master tool list

```mermaid
graph TD
  A[1. YAML manifests] --> C(Merge)
  B[2. Auto-discover<br>*.py / *.sh] --> C
  D[3. MCP servers] --> C
  C --> E[Master list<br>(schema + callables)]
```

* **Step 1 – YAML manifests**
  Every `*.yaml` file in `tools/` is parsed first. They can override names,
  types, descriptions—think of them as *authoritative declarations*.

* **Step 2 – Auto-discovery**
  Any `*.py` or `*.sh` **without** a companion YAML is inspected:
  the registry runs `script --help`, extracts a one-line summary + flags,
  synthesises a schema, and stores a Python wrapper that knows how to call the
  script.

* **Step 3 – MCP servers** *(optional)*
  If the client was started with

  ```python
  liteclient(..., mcp_servers=["http://tools.mycorp:8900"])
  ```

  the registry fetches `/list_tools` from each server and merges those
  manifests behind local ones.

---

## 3.  Two ways to publish a tool

| Mode                  | When to choose                                                                 | What you write                               |
| --------------------- | ------------------------------------------------------------------------------ | -------------------------------------------- |
| **Explicit manifest** | You want a custom name, rich type hints, or the script can’t produce `--help`. | A short YAML + any script / Python function. |
| **Auto-discover**     | You control the script and are OK to follow the help-text contract.            | **Only the script** – no YAML required.      |

### 3.1  Explicit manifest example

```yaml
# tools/img_resize.yaml
name: resize_image
description: "Resize and save an image to disk"
command: python-cli          # python / python-cli / bash-cli / bash / mcp
entry: tools/resize.py
parameters:
  type: object
  properties:
    infile:  {type: string, description: "Path to source image"}
    width:   {type: integer}
    outfile: {type: string}
  required: [infile, width]
```

### 3.2  Auto-discover example

```python
# tools/weather.py
"""
Simple weather fetcher (first non-blank line becomes the description)
"""
import argparse, json, requests

# Optional: implement get_tool_schema and support --fractalic-dump-schema
# def get_tool_schema():
#     return { ... }

ap = argparse.ArgumentParser(description="Simple weather fetcher")
ap.add_argument("--lat", required=True, help="Latitude of the location (e.g., 48.8566)")
ap.add_argument("--lon", required=True, help="Longitude of the location (e.g., 2.3522)")
ap.add_argument("--fractalic-dump-schema", action="store_true", help=argparse.SUPPRESS)
args = ap.parse_args()

url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={args.lat}&longitude={args.lon}&current=temperature_2m"
)
print(requests.get(url, timeout=5).text)
```

*No YAML file present → registry auto-discovers via `get_tool_schema()` or falls back to `--help`.*

---

## 4.  Contract for auto-discoverable scripts

| Requirement                                                      | Why it matters                                                           | Python CLI (`python-cli`)                                   | Bash CLI (`bash-cli`)                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------- | -------------------------------------------------------- |
| **1. Must exit 0 on `--help`**                                   | Registry interrogates the script; non-zero exit is assumed “not a tool”. | `argparse` does this by default.                             | Add a `show_help` function + `[[ $1 == --help ]]` guard. |
| **2. First non-blank help line is a short description**          | Shown in UI and prompt context.                                          | Provided by `description=` in `ArgumentParser`.             | Put a plain sentence before “Usage: …”.                  |
| **3. Options must be visible as `--flag ARG` in help**           | Regex / argparse walker extracts them into JSON schema.                  | `argparse` prints exactly that.                              | Document each flag in help block (see example).          |
| **4. Every `add_argument` must include a `help` string**         | Ensures parameter descriptions show up in the generated schema.          | Use `help="..."` on all flags.                             |                                                          |
| **5. Implement `get_tool_schema()` and support `--fractalic-dump-schema`** | Allows direct schema JSON output, bypassing fragile help parsing.         | Define `get_tool_schema()` returning full JSON schema object; check for `--fractalic-dump-schema` in `__main__`. |                                                          |
| **6. Script should print JSON on stdout** *(recommended)*        | Allows LLM to keep reasoning on structured data.                         | `print(json.dumps(...))`                                     | `echo '{"key":"value"}'`                                 |

*Violating the contract doesn’t break execution, but the model will see an
empty or incomplete parameter list and may fail to call the tool.*

---

## 5.  Life-cycle of a tool call

1. **Registry produces schema list** and passes it to LiteLLM (`tools=`).
2. **Model replies** with

   ```json
   {
     "tool_calls": [{
        "id":"abc", "function":{"name":"weather","arguments":"{\"lat\":...}"}
   }]}
   ```
3. **lite\_client** looks up `registry["weather"]`, unpacks the JSON arguments,
   runs the wrapper, captures stdout.
4. The result is wrapped as a `"tool"` message and returned to the model.
5. Model integrates the data into the next normal chat turn, or calls another
   tool.

---

## 6.  MCP servers – zero-friction extension

* Local and MCP tools share the **exact same manifest format**.

* The registry simply does:

  ```python
  for srv in mcp_servers:
      for manifest in requests.get(f"{srv}/list_tools").json():
          self._register(manifest, from_mcp=True)
  ```

* Execution is proxied back to the server via `POST /call_tool`.

* Local manifests **override** remote ones if names collide.

---

## 7.  FAQ & troubleshooting

| Symptom                                               | Likely cause                                      | Fix                                                               |
| ----------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| Tool appears in debug list but has empty `properties` | Script prints nothing (or malformed) on `--help`. | Add a proper help block **or** create a YAML manifest.            |
| JSON schema shows a `help` parameter                  | Older manifest or regex matched `--help`.         | Update script; registry now removes it automatically.             |
| Script executed but model sees only `stdout` string   | Script prints plain text.                         | Switch to `echo '{"result": …}'` for structured replies.          |
| Remote MCP tools never appear                         | Network / auth issue.                             | Test `curl http://server/list_tools` manually; check server logs. |

---

## TL;DR for tool authors

> 1. Write a script that **prints helpful `--help` text**, **returns JSON on stdout**, and **implements `get_tool_schema()` with `--fractalic-dump-schema` support**.
> 2. Drop it under `tools/` (no YAML required for autodiscovery).
> 3. Restart Fractalic (or call `registry.rescan()`).
> 4. The LLM can now invoke your script as `tool_name(flag=value, …)`—done!
