# MCP Full Feature Integration Plan

## Current State Analysis

### MCP SDK v2 Capabilities (Full Feature Set)
- **Tools**: `list_tools()`, `call_tool()` with metadata, annotations, titles
- **Prompts**: `list_prompts()`, `get_prompt()` for template-based interactions  
- **Resources**: `list_resources()`, `read_resource()` for contextual data
- **Advanced**: Subscriptions, notifications, progress tracking, completion
- **Auth**: Full OAuth2 with refresh tokens, per-service token storage
- **Transports**: STDIO, SSE, StreamableHTTP with auto-detection

### Fractalic Current Integration (Limited)
- **Tools Only**: Simple `/list_tools` and `/call_tool` REST endpoints
- **No Prompts**: Missing prompt discovery and template usage
- **No Resources**: Missing resource discovery and context injection
- **Simple Format**: Basic tool name/description/inputSchema only

## Recommendation: Enhance Fractalic (Not Limit MCP Manager)

The MCP manager should provide **full MCP capabilities**, and Fractalic should be enhanced to use them.

## Implementation Strategy

### Phase 1: MCP Manager Full Feature Exposure
1. **Add Prompts Endpoints**:
   ```
   GET /list_prompts - Discover available prompt templates
   POST /get_prompt/{service}/{prompt} - Retrieve prompt with parameters
   ```

2. **Add Resources Endpoints**:
   ```
   GET /list_resources - Discover available resources
   POST /read_resource/{service} - Read resource content by URI
   ```

3. **Enhanced Tool Metadata**:
   ```json
   {
     "name": "tool_name",
     "title": "Human-readable title",
     "description": "Description",
     "inputSchema": {...},
     "annotations": {...}
   }
   ```

### Phase 2: Fractalic Enhancement
1. **Tool Registry Enhancement**: Support tool titles, annotations, categories
2. **Prompt System Integration**: Add `/prompts` management to tool registry
3. **Resource Context System**: Add resource discovery and auto-injection
4. **MCP Client Enhancement**: Support full MCP feature discovery

### Phase 3: Advanced Features
1. **Dynamic Context**: Automatic resource inclusion based on LLM requests
2. **Prompt Templates**: Slash-command style prompt injection
3. **Subscription Support**: Real-time updates for changing resources/tools
4. **Progressive Enhancement**: Graceful fallback for services with limited capabilities

## Benefits of Full Integration

1. **Future-Proof**: Fractalic becomes compatible with the full MCP ecosystem
2. **Rich Context**: LLMs get access to files, databases, API documentation via resources
3. **Template Reuse**: Common workflows become reusable prompts
4. **Professional Tools**: Better tool metadata improves LLM tool selection
5. **Standard Compliance**: Full adherence to MCP specification

## Implementation Steps

1. ✅ **Current**: MCP Manager SDK v2 with tools-only compatibility
2. 🔄 **Next**: Add prompts and resources endpoints to MCP Manager  
3. ⏳ **Then**: Enhance Fractalic mcp_client.py to use full capabilities
4. ⏳ **Finally**: Integrate prompts/resources into Fractalic tool registry

This approach maximizes the value of the MCP integration while maintaining backward compatibility.
