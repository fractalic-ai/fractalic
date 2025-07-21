# Token Tracking Architecture and Implementation

## Overview

Fractalic's enhanced token tracking system provides pricing-aware analytics for LLM operations with detailed breakdown of input/output tokens and service attribution between MCP and Fractalic tools.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Token Tracking                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   TokenMessage  │  │ TokenStatsQueue │  │ LLM Provider │ │
│  │   Data Model    │  │   Collection    │  │ Integration  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Token Calculation Engine                      │ │
│  │  • Differential calculation (total - base)             │ │
│  │  • Proportional distribution by schema size            │ │
│  │  • MCP vs Fractalic classification                     │ │
│  │  • Real-time and session analytics                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
LLM Request → Tool Filtering → Token Calculation → Classification → Analytics
     ↓              ↓               ↓                ↓            ↓
   Messages    Selected Tools   Base + Total    MCP/Fractalic  Real-time Display
                                    ↓                ↓            ↓
                              Tool Tokens    Service Attribution Session Summary
```

## Core Implementation

### TokenMessage Data Model

```python
@dataclass
class TokenMessage:
    """Complete token usage record with pricing breakdown"""
    # Identifiers
    operation_id: str                    # Unique operation ID
    timestamp: float                     # Unix timestamp
    model: str                          # LLM model used
    
    # Input Token Breakdown (what we send to model)
    input_tokens: int                   # Total input tokens
    input_context_tokens: int           # Messages, text, conversation history
    input_schema_tokens: int            # Tool schemas (MCP + Fractalic)
    input_mcp_schema_tokens: int        # External MCP tool schemas
    input_fractalic_schema_tokens: int  # Internal Fractalic tool schemas
    
    # Output Tokens (what model generates)
    output_tokens: int                  # LLM completion tokens
    
    # Totals and Metadata
    total_tokens: int                   # input_tokens + output_tokens
    tool_calls_count: int               # Number of tool calls made
    tools_available_count: int          # Number of tools in schema
    operation_type: str                 # "llm_turn", "@run", "tool_call"
    source_file: str                    # Source file context
    metadata: Optional[Dict[str, Any]]  # Additional data
```

### Token Calculation Algorithm

**Core Problem**: Accurately attribute tool schema tokens to specific services (MCP vs Fractalic).

**Solution**: Differential calculation with proportional distribution.

```python
def calculate_detailed_tokens_enhanced(messages, tools, calculated_prompt_tokens, 
                                      mcp_tool_tokens, fractalic_tool_tokens):
    """
    Enhanced token calculation algorithm:
    1. Calculate base tokens (messages without tools)
    2. Calculate total tokens (messages with all tools)  
    3. Extract tool tokens as difference
    4. Proportion tool tokens by relative schema sizes
    5. Classify and sum by service type
    """
    
    # Step 1: Base calculation
    base_prompt_tokens = litellm.token_counter(model=model, messages=messages)
    
    # Step 2: Total with tools
    calculation_params = {"model": model, "messages": messages, "tools": tools}
    total_with_tools = litellm.token_counter(**calculation_params)
    
    # Step 3: Extract tool tokens
    total_tool_tokens = total_with_tools - base_prompt_tokens
    
    # Step 4: Proportional distribution by JSON schema size
    tool_sizes = {}
    total_tool_chars = 0
    
    for tool in tools:
        tool_name = tool.get("function", {}).get("name", "unknown")
        tool_json = json.dumps(tool, separators=(',', ':'))
        tool_size = len(tool_json)
        tool_sizes[tool_name] = tool_size
        total_tool_chars += tool_size
    
    # Step 5: Service classification and token attribution
    mcp_tool_tokens = 0
    fractalic_tool_tokens = 0
    
    for tool in tools:
        tool_name = tool.get("function", {}).get("name", "unknown")
        tool_proportion = tool_sizes[tool_name] / total_tool_chars if total_tool_chars > 0 else 0
        tool_tokens = int(total_tool_tokens * tool_proportion)
        
        if is_fractalic_tool(tool_name, registry):
            fractalic_tool_tokens += tool_tokens
        else:
            mcp_tool_tokens += tool_tokens
    
    context_tokens = max(0, calculated_prompt_tokens - total_tool_tokens)
    return context_tokens, mcp_tool_tokens, fractalic_tool_tokens, calculated_prompt_tokens
```

### Tool Classification Logic

```python
def is_fractalic_tool(tool_name, registry):
    """
    Tool classification priority:
    1. Registry manifest service metadata
    2. Naming pattern matching
    3. Default classification
    """
    
    # Priority 1: Registry metadata
    if registry:
        for manifest in registry._manifests:
            if manifest.get("name") == tool_name:
                service = manifest.get("_service", "")
                # No service or "fractalic"/"local" = Fractalic tool
                if not service or service.lower() in ["fractalic", "local"]:
                    return True
                break
    
    # Priority 2: Pattern matching
    fractalic_patterns = ["fractalic_", "add_tools", "edit_tools", "fractalic_opgen"]
    if any(tool_name.startswith(pattern) or tool_name == pattern 
           for pattern in fractalic_patterns):
        return True
    
    # Priority 3: Default to MCP for unrecognized tools
    return False
```

### Tool Filtering System

```python
def apply_tool_filtering(tools_param, available_schema, registry):
    """
    Tool filtering ensures only specified tools contribute to token count.
    
    Supported filters:
    - "all": Include all available tools
    - "none": No tools (zero schema tokens)
    - "tool_name": Single tool by name
    - ["tool1", "tool2"]: Multiple specific tools
    - "mcp/server_name": All tools from specific MCP server
    """
    
    if tools_param == "none":
        return []
    elif tools_param == "all":
        return available_schema.copy()
    elif isinstance(tools_param, str) and tools_param.startswith("mcp/"):
        mcp_server_name = tools_param[4:]
        return filter_by_mcp_server(mcp_server_name, available_schema, registry)
    elif isinstance(tools_param, list):
        return filter_by_tool_list(tools_param, available_schema, registry)
    elif isinstance(tools_param, str):
        return [tool for tool in available_schema 
                if tool["function"]["name"] == tools_param]
    else:
        return []
```

## Integration Points

### 1. LLM Provider Integration

**File**: `core/llm/providers/openai_client.py`

```python
# Token calculation integration
try:
    import litellm
    # Calculate base prompt tokens (conversation without tools)
    base_prompt_tokens = litellm.token_counter(model=self.model, messages=hist[:-1])
    
    # Calculate tool schema tokens breakdown
    mcp_tool_tokens = 0
    fractalic_tool_tokens = 0
    
    if "tools" in params and params["tools"]:
        # Calculate detailed breakdown using enhanced algorithm
        total_with_tools = litellm.token_counter(**calculation_params)
        total_tool_tokens = total_with_tools - base_prompt_tokens
        
        # Proportional distribution and classification
        # ... (implementation as shown above)
    
    calculated_prompt_tokens = litellm.token_counter(**calculation_params)
    
    # Send to tracking system
    token_stats.send_usage(
        operation_id=unique_id,
        model=op.get("model", self.model),
        input_tokens=total_input,
        input_context_tokens=context_tokens,
        input_schema_tokens=mcp_tool_tokens + fractalic_tool_tokens,
        input_mcp_schema_tokens=mcp_tool_tokens,
        input_fractalic_schema_tokens=fractalic_tool_tokens,
        output_tokens=completion_tokens,
        total_tokens=calculated_total,
        tool_calls_count=tool_calls_made,
        tools_available_count=tools_count,
        operation_type="llm_turn",
        source_file=source_file
    )
```

### 2. Session Management

**File**: `core/operations/runner.py`

```python
# Session lifecycle hooks
def setup_session():
    if Config.ENABLE_TOKEN_TRACKING:
        from core.token_stats import token_stats
        token_stats.reset()

def finalize_session():
    if Config.ENABLE_TOKEN_TRACKING:
        from core.token_stats import token_stats
        token_stats.print_session_summary()
```

### 3. Main Application

**File**: `fractalic.py`

```python
# Global configuration
Config.ENABLE_TOKEN_TRACKING = True

# Integration with operation flow
def main():
    setup_session()
    try:
        # Run operations...
        pass
    finally:
        finalize_session()
```

## Constants and Configuration

### System Constants

```python
# Token calculation constants
DEFAULT_CHARS_PER_TOKEN = 4              # Rough estimation ratio
MAX_QUEUE_SIZE = 1000                    # Default message queue limit
SESSION_TIMEOUT = 3600                   # Session timeout in seconds

# Display format constants
TOKEN_DISPLAY_FORMAT_SIMPLE = "simple"
TOKEN_DISPLAY_FORMAT_ENHANCED = "enhanced"

# Tool classification patterns
FRACTALIC_TOOL_PREFIXES = ["fractalic_"]
FRACTALIC_TOOL_NAMES = ["add_tools", "edit_tools", "fractalic_opgen"]
FRACTALIC_SERVICE_TYPES = ["fractalic", "local", ""]

# Operation types
OPERATION_TYPE_LLM_TURN = "llm_turn"
OPERATION_TYPE_TOOL_CALL = "tool_call"
OPERATION_TYPE_RUN = "@run"
OPERATION_TYPE_FINAL = "llm_final"
```

### Configuration Options

```python
# Core configuration in Config class
class Config:
    # Token tracking settings
    ENABLE_TOKEN_TRACKING: bool = True
    TOKEN_QUEUE_SIZE: int = 1000
    TOKEN_DISPLAY_FORMAT: str = "enhanced"
    DEBUG_TOKEN_TRACKING: bool = False
    
    # Performance settings
    TOKEN_CALCULATION_TIMEOUT: float = 5.0
    CLASSIFICATION_CACHE_SIZE: int = 100
    
    # Cost analysis settings
    CONTEXT_TOKEN_RATE: float = 0.001       # Per 1K tokens
    SCHEMA_TOKEN_RATE: float = 0.001        # Per 1K tokens
    OUTPUT_TOKEN_RATE: float = 0.002        # Per 1K tokens
```

### Environment Variables

```bash
# Optional environment overrides
FRACTALIC_ENABLE_TOKEN_TRACKING=true
FRACTALIC_TOKEN_QUEUE_SIZE=1000
FRACTALIC_DEBUG_TOKEN_TRACKING=false
FRACTALIC_TOKEN_DISPLAY_FORMAT=enhanced
```

## Display Formats

### Real-time Token Display

```
[TOKENS] Input: 1378 (Context: 578, MCP: 0, Fractalic: 800), Output: 99, Turn: 1477, Cumulative: 1477, Tools: 1, Turn: 2
```

**Format Components**:
- `Input: 1378` - Total input tokens
- `Context: 578` - Message/conversation tokens
- `MCP: 0` - External MCP tool schema tokens
- `Fractalic: 800` - Internal Fractalic tool schema tokens
- `Output: 99` - LLM completion tokens
- `Turn: 1477` - Total tokens this operation
- `Cumulative: 1477` - Running session total
- `Tools: 1` - Number of tools available
- `Turn: 2` - Operation sequence number

### Session Summary Format

```
=== SESSION TOKEN USAGE SUMMARY ===
Total Input Tokens: 3,796
  - Context Tokens: 2,996
  - Schema Tokens: 800
    • MCP Tools: 0
    • Fractalic Tools: 800
Total Output Tokens: 539
Total Tokens: 4,335
Total Tool Calls: 0
Total Tools Available: 1
Total Operations: 3
Session Duration: 42.74s

Breakdown by source:
  agent-image-analyze.md: 4,335 tokens, 3 operations
    Input: 3,796 (Context: 2,996, MCP: 0, Fractalic: 800), Output: 539
    Tool Calls: 0, Tools Available: 1
```

## Performance Optimization

### Efficient Token Calculation

**Problem**: Multiple LiteLLM API calls are expensive.

**Solution**: Single differential calculation.

```python
# Inefficient approach (avoided)
for tool in tools:
    tool_tokens = litellm.token_counter(model=model, messages=[], tools=[tool])

# Efficient approach (implemented)
base_tokens = litellm.token_counter(model=model, messages=messages)
total_tokens = litellm.token_counter(model=model, messages=messages, tools=tools)
tool_tokens = total_tokens - base_tokens  # Single calculation
```

### Classification Caching

```python
class ToolClassificationCache:
    """Session-level tool classification cache"""
    def __init__(self, max_size=100):
        self._cache = {}
        self.max_size = max_size
    
    def classify_tool(self, tool_name, registry):
        if tool_name not in self._cache:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[tool_name] = self._perform_classification(tool_name, registry)
        return self._cache[tool_name]
```

### Memory Management

```python
class TokenStatsQueue:
    """Bounded queue with automatic cleanup"""
    def __init__(self, max_messages=1000):
        self.max_messages = max_messages
        self.messages = []
    
    def send_usage(self, **kwargs):
        # Add new message
        self.messages.append(TokenMessage(**kwargs))
        
        # Maintain size limit
        if len(self.messages) > self.max_messages:
            # Remove oldest 10% when limit exceeded
            remove_count = max(1, self.max_messages // 10)
            self.messages = self.messages[remove_count:]
```

## Error Handling and Fallbacks

### Graceful Degradation

```python
def calculate_tokens_with_fallback(messages, tools, model):
    """Multi-level fallback for token calculation"""
    try:
        # Primary: LiteLLM accurate calculation
        return litellm.token_counter(model=model, messages=messages, tools=tools)
    except ImportError:
        # Fallback 1: Character-based estimation
        return estimate_tokens_from_content(messages, tools)
    except Exception as e:
        # Fallback 2: Safe minimum
        logger.warning(f"Token calculation failed: {e}")
        return 50  # Safe default
```

### Safe Tool Classification

```python
def safe_classify_tool(tool_name, registry):
    """Error-safe tool classification"""
    try:
        return classify_tool_via_registry(tool_name, registry)
    except RegistryError:
        return classify_tool_via_patterns(tool_name)
    except Exception:
        logger.debug(f"Classification failed for {tool_name}, defaulting to 'unknown'")
        return "unknown"
```

## Usage Examples

### Basic Integration

```python
from core.llm.llm_client import LLMClient
from core.token_stats import token_stats

# Initialize client
llm_client = LLMClient(model='gpt-4o', provider='openai')

# Make tracked call
response = llm_client.llm_call(
    prompt_text="Analyze this data",
    operation_params={
        'tools': 'all',  # Use all available tools
        '_source_file': 'analysis.py'
    }
)

# Get session analytics
summary = token_stats.get_session_summary()
print(f"Total cost: {summary['total_tokens']} tokens")
```

### Cost Analysis

```python
# Analyze service costs
summary = token_stats.get_session_summary()

fractalic_cost = summary['total_fractalic_schema_tokens'] * SCHEMA_TOKEN_RATE
mcp_cost = summary['total_mcp_schema_tokens'] * SCHEMA_TOKEN_RATE
context_cost = summary['total_context_tokens'] * CONTEXT_TOKEN_RATE
output_cost = summary['total_output_tokens'] * OUTPUT_TOKEN_RATE

print(f"Service breakdown:")
print(f"  Fractalic tools: ${fractalic_cost:.4f}")
print(f"  MCP tools: ${mcp_cost:.4f}")
print(f"  Context: ${context_cost:.4f}")
print(f"  Output: ${output_cost:.4f}")
```

### Tool Optimization

```python
# Compare tool configurations
def analyze_tool_efficiency():
    # Test with all tools
    stats_all = run_operation(tools='all')
    
    # Test with specific tool
    stats_specific = run_operation(tools='fractalic_run')
    
    # Test with no tools
    stats_none = run_operation(tools='none')
    
    print(f"Schema token comparison:")
    print(f"  All tools: {stats_all['schema_tokens']}")
    print(f"  Specific: {stats_specific['schema_tokens']}")  
    print(f"  None: {stats_none['schema_tokens']}")
    
    efficiency = stats_specific['tool_calls'] / max(1, stats_specific['tools_available'])
    print(f"Tool efficiency: {efficiency:.2%}")
```

## Testing and Validation

### Test Categories

```python
# Unit tests for core algorithms
def test_token_calculation():
    assert calculate_tool_tokens(messages, tools) == expected_tokens

def test_tool_classification():
    assert classify_tool("fractalic_run", registry) == "fractalic"
    assert classify_tool("mcp_tool", registry) == "mcp"

# Integration tests for complete workflow
def test_complete_tracking():
    result = llm_client.llm_call(messages, tools="all")
    assert token_stats.get_session_summary()["total_tokens"] > 0

# Performance tests for efficiency
def test_calculation_performance():
    start_time = time.time()
    calculate_detailed_tokens_enhanced(large_dataset)
    duration = time.time() - start_time
    assert duration < MAX_CALCULATION_TIME
```

---

## Summary

This enhanced token tracking system provides:

- **Accurate token attribution** through differential calculation
- **Service cost breakdown** between MCP and Fractalic tools
- **Real-time monitoring** with cumulative session tracking
- **Performance optimization** through efficient algorithms
- **Error resilience** with multiple fallback mechanisms
- **Flexible configuration** for different use cases

The implementation is production-ready and provides comprehensive pricing analytics for Fractalic LLM operations.
