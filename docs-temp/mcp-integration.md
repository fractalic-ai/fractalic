---
title: MCP Integration
description: Integrate Model Context Protocol servers to extend Fractalic with external tools and capabilities
outline: deep
---

# MCP Integration

## Purpose
MCP Integration extends Fractalic with external tools that your AI agents can use. Instead of being limited to built-in commands, you can add search engines, databases, APIs, and custom business tools. This section explains how tools become available to `@llm` operations and how to add new ones.

## Internal Table of Contents
- [Purpose](#purpose)
- [Internal Table of Contents](#internal-table-of-contents)
- [How MCP Works in Fractalic](#how-mcp-works-in-fractalic)
- [The Manager System](#the-manager-system)
- [Adding Your First Tool](#adding-your-first-tool)
- [Tool Execution Flow](#tool-execution-flow)
- [Managing Services](#managing-services)
- [Common Issues & Solutions](#common-issues--solutions)
- [Performance Tips](#performance-tips)
- [See Also](#see-also)

## How MCP Works in Fractalic
Fractalic uses one MCP Manager that acts as a central hub for all external tool capabilities. The manager is a standalone server that exposes tools, prompts, and resources through a REST API. It handles the complexity of connecting to different types of MCP servers while presenting a unified interface.

The flow works as follows:
1. MCP Manager starts automatically with Fractalic (runs on port 5859)
2. Manager reads `mcp_servers.json` and establishes connections to configured MCP servers
3. Each MCP server can provide tools, prompts, and resources via the MCP protocol
4. Manager caches these capabilities and exposes them through HTTP endpoints
5. When an `@llm` block runs with tools enabled, Fractalic fetches the unified tool catalog
6. AI selects tools based on task requirements; calls are routed through the manager
7. Manager handles the actual tool execution and returns results

Initial startup takes a few seconds as the manager discovers and caches all available capabilities from configured servers. Subsequent operations are much faster due to intelligent caching.

## The Manager System
The MCP Manager provides several key capabilities:

**Multi-Protocol Support**: Handles different MCP server types including stdio (local executables), SSE (server-sent events), and HTTP endpoints. Transport type is auto-detected based on configuration.

**OAuth Integration**: Full OAuth 2.0 support with automatic token refresh for services that require authentication. When enabled, the manager handles the complete OAuth flow and maintains valid tokens.

**Comprehensive Caching**: Intelligent caching system with configurable TTL for tools, prompts, resources, and service status. This dramatically reduces latency after initial discovery.

**Unified API**: Exposes a clean REST API that Fractalic uses:
- `/list_tools` - Flat list of all available tools from all servers
- `/call/{service}/{tool}` - Execute specific tools with parameters  
- `/status/complete` - Full system status with embedded data
- `/toggle/{service}` - Enable/disable services without restart

**UI Integration**: The Fractalic UI can manage MCP servers directly, including adding new servers using standard Claude Desktop JSON configurations that can be copy-pasted.

**Prompts and Resources**: Beyond tools, the manager also exposes prompts and resources from MCP servers, making them available for advanced workflows.

## Server Configuration
MCP servers are configured in `mcp_servers.json`. The manager supports the standard Claude Desktop format, so existing configurations can be directly copied:

```json
{
  "mcpServers": {
    "memory-stdio-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "transport": "stdio",
      "enabled": true
    },
    "web-search": {
      "url": "https://api.example.com/mcp",
      "oauth": true,
      "enabled": true
    }
  }
}
```

**Key fields:**
- `command` + `args` + `env`: For stdio servers (local executables)
- `url`: For HTTP/SSE remote servers
- `transport`: Usually auto-detected (`stdio`, `sse`, `streamable-http`)
- `oauth`: Enable OAuth flow for authenticated services
- `enabled`: Control availability without deleting configuration

The manager auto-detects transport type from URL patterns when not specified.

## Adding Your First Tool
To add a new MCP server, you can use the Fractalic UI or edit the configuration file directly.

**Via Fractalic UI**: The UI provides an "Add Server" interface that accepts standard Claude Desktop JSON configurations. You can copy existing server definitions and paste them directly.

**Via Configuration File**: Edit `mcp_servers.json` in your project root:

```json
{
  "mcpServers": {
    "memory-stdio-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "enabled": true
    }
  }
}
```

After adding a server:
1. Save the configuration
2. Restart the MCP Manager (or it will auto-refresh)
3. Test with an `@llm` block that enables tools

Example usage:
```markdown
@llm
prompt: "Remember that today's weather was sunny and 75°F"
tools:
  - mcp/memory-stdio-server
tools-turns-max: 2
```

## Tool Naming and Selection
Tools are exposed with a `service.tool_name` format for uniqueness. For example, the memory server's `add` tool becomes `memory-stdio-server.add`. 

**Filtering by Service**: You can restrict tool access to specific services using the `mcp/service-name` pattern in your tools list:

```markdown
@llm
prompt: "Store this information in memory"
tools:
  - mcp/memory-stdio-server
tools-turns-max: 2
```

This gives the AI access only to tools from the memory server, not all available tools.

**All Tools (Built-in + All MCP)**: Use `tools: all` (a bare string, not an array) to expose every registered tool. This includes all built-in/local tools plus every tool from every enabled MCP service. Use sparingly—broad exposure increases cost surface and risk of irrelevant tool calls.

**Single Service Shorthand**: Set `tools: mcp/<service-name>` (string form) to include all tools from exactly one MCP service. Example:

```markdown
@llm
prompt: "List current stored memory keys"
tools: mcp/memory-stdio-server
```

**Multiple Services and Specific Tools**: Provide an array mixing service filters (`mcp/service-name`) and explicit tool names. Example:

```markdown
@llm
prompt: "Search the web then store a fact in memory and summarize"
tools:
  - mcp/web-search          # all tools from web-search service
  - mcp/memory-stdio-server # all memory tools
  - summarize_document      # a specific local tool by exact name
```

Resolution logic:
1. Each list entry that starts with `mcp/` expands to all tools from that service.
2. Each plain name is matched exactly to a single tool function.
3. Duplicates are removed.
4. If no matches are found for any requested item, an error is raised with available services/tools context.

**Not Supported**: `tools: [mcp]` or `tools: mcp` (without a `/service`) is NOT a valid pattern—there is no shorthand meaning “all MCP tools only.” Use `tools: all` if you truly need the full set, or enumerate specific services with `mcp/<service>` filters for tighter control.

Practical recommendation: Prefer explicit service scopes (e.g. `mcp/memory-stdio-server`) over `all` for observability, predictability, and token efficiency.

## Tool Execution Flow
When your AI selects a tool:

1. AI generates function call with tool name and JSON parameters
2. Fractalic maps the sanitized name back to service and tool components
3. HTTP POST to manager `/call/{service}/{tool}` with arguments
4. Manager creates short-lived connection to appropriate MCP server
5. Tool executes and returns structured results
6. Results are formatted and added to conversation context as tool message
7. AI continues reasoning with new information until task complete or turn limit reached

The manager uses short-lived connections per call to avoid resource accumulation and ensure clean execution.

## Managing Services
**Enable/Disable**: Set `enabled: false` in `mcp_servers.json` to temporarily disable a service without losing its configuration. The manager will skip disabled services during discovery.

**OAuth Services**: For services requiring authentication, set `oauth: true`. The manager handles the complete OAuth 2.0 flow:
- Initial authorization redirect
- Token exchange and storage in `oauth-cache/` (project directory)
- Automatic token refresh when expired
- Graceful fallback when authentication fails

**Docker Deployment Compatibility**: OAuth tokens are now stored in the project's `oauth-cache/` directory, making them easily portable to Docker containers. The directory is automatically created and configured to use relative paths from the project root.

For production deployments, consider using MCP services with embedded authentication tokens in URLs rather than OAuth flows, or ensure OAuth token persistence through Docker volume mounts if needed.

**Service Status**: The manager provides real-time status through `/status/complete` endpoint, showing which services are connected, tool counts, and any errors.

**Hot Reload**: Configuration changes are detected automatically in most cases. For immediate effect, restart the manager or use the UI toggle functions.

## Logging and Debugging
Enable detailed logging by setting environment variables:
```bash
export MCP_DEBUG=1
export FRACTALIC_TRACE_TOKENS=1
```

This provides insights into:
- Service connection attempts and failures
- Tool discovery and caching behavior
- OAuth token lifecycle events
- Performance metrics and timing

Common log messages help identify configuration issues or service problems.

## Common Issues & Solutions
**No tools appear**: Verify MCP Manager is running on port 5859. Check `settings.toml` points to correct manager URL (`http://127.0.0.1:5859`).

**Service won't start**: Check command syntax and paths in `mcp_servers.json`. For stdio servers, verify the executable is available. Test commands manually first.

**Slow initial response**: First tool discovery takes time as the manager queries all enabled services and builds the cache. Subsequent calls are fast.

**Tool execution fails**: Check tool parameters match expected schema. Enable debug logging to see detailed error messages.

**OAuth authentication issues**: Verify service supports OAuth and redirect URLs are configured correctly. Check token refresh logic in debug logs.

## Performance Optimization
- Only enable servers you actively use to reduce discovery overhead
- Use specific service filtering (`mcp/service-name`) rather than enabling all tools
- Set conservative `tools-turns-max` values (2-3) to prevent runaway loops
- Summarize large tool outputs in follow-up operations to manage context size
- Monitor cache hit rates in debug logs to ensure optimal performance

## See Also
- [Configuration](configuration.md) - MCP manager connection settings
- [Advanced LLM Features](advanced-llm-features.md) - Tool loop patterns and optimization
- [Operations Reference](operations-reference.md) - Complete `@llm` operation syntax
