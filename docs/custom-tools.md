---
title: Creating Custom Tools
description: Create and integrate custom tools and MCP servers to extend Fractalic capabilities
outline: deep
---

# Creating Custom Tools

Focus: How to add your own tools to Fractalic and how the tool discovery system works.

## What Are Tools?

Tools are external programs that your AI agents can call during `@llm` operations. When you write:

```markdown
@llm
prompt: Find and summarize recent GitHub issues for this project.
tools:
  - github_issues
  - text_summarizer
```

Fractalic automatically discovers and calls these tools, feeding their results back to the AI model.

## Three Ways Tools Get Discovered

Fractalic finds tools from three sources (in this order):

1. **YAML Manifests**: Explicit `*.yaml` files in your `tools/` directory
2. **Auto-Discovery**: Python/shell scripts without YAML files get introspected automatically  
3. **MCP Servers**: Remote tools from Model Context Protocol servers (see [MCP Integration](mcp-integration.md))

If tools have the same name, the first one found wins.

## Your First Tool: Simple JSON Pattern

The easiest way to create a tool is the "simple JSON" pattern. Your script:
1. Receives JSON input as a command line argument
2. Returns JSON output to stdout
3. Must respond to a test ping within 200ms

Here's a minimal example:

**tools/greet.py**:
```python
#!/usr/bin/env python
import json
import sys

def main():
    # REQUIRED: Quick test response (for auto-discovery)
    if len(sys.argv) == 2 and sys.argv[1] == '{"__test__": true}':
        print(json.dumps({"success": True}))
        return
    
    # REQUIRED: Process real input
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Expected JSON input"}))
        sys.exit(1)
    
    try:
        data = json.loads(sys.argv[1])
        name = data.get("name", "World")
        result = {"greeting": f"Hello, {name}!"}
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Save this file and run Fractalic - it will automatically discover `greet` as a tool! 

**Important**: Without additional schema information, the AI model will see this as a generic tool that accepts any JSON parameters. The model can still use it, but won't know what specific parameters are expected.

## Adding Schema Information (Optional)

For better AI integration, you can add a schema dump handler that tells the AI exactly what parameters your tool expects:

```python
def main():
    # Test response (required)
    if len(sys.argv) == 2 and sys.argv[1] == '{"__test__": true}':
        print(json.dumps({"success": True}))
        return
    
    # Schema dump (optional but recommended)
    if len(sys.argv) == 2 and sys.argv[1] == "--fractalic-dump-schema":
        schema = {
            "description": "Generate a personalized greeting",
            "parameters": {
                "type": "object", 
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name to greet"
                    },
                    "style": {
                        "type": "string", 
                        "enum": ["formal", "casual", "enthusiastic"],
                        "default": "casual"
                    }
                },
                "required": ["name"]
            }
        }
        print(json.dumps(schema))
        return
    
    # ... rest of tool logic
```

With this schema dump, the AI model knows exactly what parameters to send and how to use your tool effectively.

**Without schema dump**: Tool works but AI sees it as generic `{"additionalProperties": true}`  
**With schema dump**: AI gets precise parameter definitions and descriptions

## Manual Control with YAML Manifests

For production tools or when you need precise control, create explicit YAML manifests:

**tools/github_issues.yaml**:
```yaml
name: github_issues
description: Fetch issues from a GitHub repository
command: simple-json
entry: tools/github_issues.py
parameters:
  type: object
  properties:
    repo:
      type: string
      description: Repository in format 'owner/name'
    state:
      type: string
      enum: ["open", "closed", "all"]
      default: "open"
    limit:
      type: integer
      minimum: 1
      maximum: 100
      default: 10
  required: [repo]
```

The corresponding Python script follows the same JSON input/output pattern as before.

## Working Directory and Environment

Tools run in their own directory with environment variables from your `settings.toml`:

```toml
[[environment]]
key = "GITHUB_TOKEN"
value = "ghp_your_token_here"
```

Your tool can access this via `os.environ["GITHUB_TOKEN"]`.

## Legacy: Argparse Pattern (Still Supported)

Fractalic can also auto-discover traditional argparse scripts:

```python
#!/usr/bin/env python
"""Fetch repository statistics."""
import argparse
import json

parser = argparse.ArgumentParser(description="Get repo stats")
parser.add_argument("--repo", required=True, help="Repository name")
parser.add_argument("--include-forks", action="store_true", help="Include forks")
args = parser.parse_args()

# Tool logic here...
result = {"stars": 42, "forks": 7}
print(json.dumps(result))
```

But the simple JSON pattern is recommended for new tools.

## Tool Discovery Process

When Fractalic starts, it:

1. Scans your `tools/` directory (default location)
2. Loads any `*.yaml` manifest files first
3. For `*.py` and `*.sh` files without YAML, runs introspection:
   - Sends `{"__test__": true}` and expects JSON response within 200ms
   - If that works, optionally calls `--fractalic-dump-schema` for detailed parameter schema
   - If no schema dump available, creates generic schema with `"additionalProperties": true`
   - Creates auto-generated manifest
4. Connects to MCP servers and fetches their tool lists
5. Makes everything available to AI models as function calls

**Key insight**: Simple JSON tools work immediately without any schema definition - the AI can call them with any parameters and your tool decides how to handle the input. Adding schema dump is optional but helps the AI use your tool more effectively.

## Testing Your Tools

Test tools directly before using them in Fractalic:

```bash
# Test the discovery ping
python tools/greet.py '{"__test__": true}'

# Test actual functionality  
python tools/greet.py '{"name": "Alice", "style": "enthusiastic"}'

# Get schema (if implemented)
python tools/greet.py --fractalic-dump-schema
```

## Checking Tool Registration

See what tools Fractalic found:

```bash
# If UI server is running
curl 'http://localhost:8000/tools_schema/?tools_dir=tools'

# Or check the logs when Fractalic starts
```

## Common Patterns

**File Processing Tool**:
```python
def main():
    # Standard handlers...
    
    data = json.loads(sys.argv[1])
    file_path = data["file_path"]
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Process content...
    result = {"lines": len(content.splitlines()), "chars": len(content)}
    print(json.dumps(result))
```

**API Tool with Error Handling**:
```python
import requests

def main():
    # Standard handlers...
    
    try:
        data = json.loads(sys.argv[1])
        response = requests.get(f"https://api.example.com/{data['endpoint']}")
        response.raise_for_status()
        
        result = {"data": response.json(), "status": "success"}
        print(json.dumps(result))
    except requests.RequestException as e:
        print(json.dumps({"error": f"API error: {str(e)}"}))
        sys.exit(1)
```

## Best Practices

- **Keep output focused**: Return only what the AI needs, not everything available
- **Handle errors gracefully**: Always return JSON, even for errors
- **Add good descriptions**: Help the AI understand when to use your tool
- **Test thoroughly**: Verify both success and error cases
- **Be fast**: Tools that take >30 seconds may cause timeouts
- **Stay secure**: Validate inputs, especially for file/network operations

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Tool not discovered | Check file permissions, ensure `{"__test__": true}` works |
| "JSON parse error" | Tool printed non-JSON to stdout (use stderr for logs) |
| Tool times out | Optimize performance or increase timeout in config |
| Missing from `/tools_schema/` | Check Fractalic logs for discovery errors |
| Wrong parameters | Add explicit YAML manifest or improve schema dump |

## Next Steps

- Try creating a simple tool using the JSON pattern
- Check out [MCP Integration](mcp-integration.md) for connecting external tool ecosystems
- See [Advanced LLM Features](advanced-llm-features.md) for tool usage patterns
- Read [Operations Reference](operations-reference.md) for using tools in workflows

## See Also
- [Operations Reference](operations-reference.md) 
- [Advanced LLM Features](advanced-llm-features.md)
- [MCP Integration](mcp-integration.md)
- [Configuration](configuration.md)
