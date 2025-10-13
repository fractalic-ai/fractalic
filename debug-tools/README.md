# Event Debugging Tool

Tool for debugging and analyzing events emitted by Fractalic during workflow file execution.

## Usage

### 1. Start the server

```bash
./test_chat.sh
```

Or manually:
```bash
python core/ui_server/server.py
```

### 2. Run the debug script

```bash
python debug-tools/debug_events_api.py path/to/your/workflow.md
```

Example:
```bash
python debug-tools/debug_events_api.py tutorials/newchat-test/test-nested-execution.md
```

### 3. Analyze events

Events are output to console in real-time and saved to a JSON file:
- `events_log_YYYYMMDD_HHMMSS.json` - on successful completion
- `events_log_error_YYYYMMDD_HHMMSS.json` - on error
- `events_log_interrupted_YYYYMMDD_HHMMSS.json` - on interruption (Ctrl+C)

## Event Types

### Lifecycle Events

- **execution_start** - Workflow execution started
  - `target`: file name

- **execution_complete** - Execution completed successfully
  - `target`: file name

- **execution_error** - Execution error
  - `target`: file name

### Workflow Events (nested executions)

- **workflow_start** - Nested execution started (@run)
  - `target`: child file name
  - `parent_execution_id`: parent execution ID
  - `nested_execution_id`: nested execution ID
  - `node_id`: node ID in call tree
  - `block_id`: operation block ID

- **workflow_complete** - Nested execution completed
  - `return_content`: returned content (if @return present)
  - `explicit_return`: explicit return flag

- **workflow_error** - Error in nested execution
  - `error_message`: error text

### AST Events

- **ast_update** - AST tree updated
  - `operation`: operation type (parse, llm, shell, import, return)
  - `blocks`: array of AST blocks with their structure

### Processing Events

- **block_processing** - Started processing operation block
  - `block_id`: ID of block being processed
  - `operation`: operation type (llm, shell, run, return)

### Terminal Output Events

- **terminal_output** - Terminal output chunk
  - `data`: output text
  - `is_stderr`: whether from stderr
  - `seq`: sequence number

### Token Usage Events

- **token_usage_call** - LLM token usage for a single call
  - `model`: model name
  - `input_tokens`: input token count
  - `output_tokens`: output token count
  - `response_cost`: request cost
  - `block_id`: operation block ID

- **token_usage_summary** - Aggregated token usage summary
  - Same fields as token_usage_call, but aggregated for entire file/operation

### Chat Events

- **chat_message** - Chat message
  - `role`: role (user, assistant)
  - `content`: message text

## Event Structure

All events contain common fields:

```json
{
  "type": "event_type",
  "execution_id": "uuid",
  "session_id": "uuid",
  "timestamp": 1234567890.123,
  "root_execution_id": "uuid",
  "seq": 1,
  ...event-specific fields
}
```

### Nested execution tracing

- **root_execution_id** - Root execution ID (same for all nested)
- **execution_id** - Current level execution ID
- **parent_execution_id** - Parent execution ID (for nested)
- **nested_execution_id** - Nested execution ID (format: `parent_id_suffix`)

## Examples

### View events in real-time

```bash
python debug-tools/debug_events_api.py tutorials/newchat-test/test-nested-execution.md
```

### Analyze saved events

```bash
# Pretty-print JSON
cat events_log_20251012_162745.json | python -m json.tool

# Count events by type
cat events_log_20251012_162745.json | python -m json.tool | grep '"type"' | sort | uniq -c

# Filter workflow events
cat events_log_20251012_162745.json | python -m json.tool | grep -A5 '"type": "workflow_'
```

### Python analysis

```python
import json

with open('events_log_20251012_162745.json') as f:
    events = json.load(f)

# Find all nested executions
nested = [e for e in events if e.get('nested_execution_id')]
print(f"Nested executions: {len(nested)}")

# Build call graph
for event in events:
    if event['type'] == 'workflow_start':
        print(f"{event['parent_execution_id']} -> {event['nested_execution_id']}")
        print(f"  Target: {event['target']}")
```

## Troubleshooting

### session_root not set

If you see `"session_root is not set"` error, make sure:
1. You're using `debug_events_api.py` (not direct `run()` call)
2. Server is running correctly

### Events not arriving

1. Check server is running: `curl http://localhost:8000/health`
2. Check `FRACTALIC_SERVER_URL` environment variable
3. Look at server logs in `test_chat.sh` output

### Duplicate events

This is normal - some events (e.g., `token_usage`) may be emitted twice:
- Once for the current block
- Second time as summary for entire file

### Terminal output duplication

**Fixed in recent version**: Terminal output now uses event-based streaming with sequence numbers, ensuring each chunk is emitted exactly once.

## Architecture

```
fractalic.py (process)
    ↓ emit_event()
    ↓ HTTP POST
/api/events/receive (server.py)
    ↓ events_queue (unified queue)
    ↓
/api/chat/stream (server.py)
    ↓ NDJSON stream
debug_events_api.py
    ↓ JSON file
events_log_*.json
```

## Key Implementation Details

### Event-based Terminal Streaming

Terminal output is now streamed through the unified event queue with:
- **Sequence numbers**: Each chunk has a unique `seq` number
- **No duplication**: Each chunk emitted exactly once
- **Execution ID routing**: Terminal output keyed by `execution_id` for correct bubble association

### Return Content Handling

The `@return` operation correctly references blocks using exact IDs:
- ✅ Correct: `@return block: llm-response-block/*`
- ❌ Incorrect: `@return block: llm-response/*` (matches wrong blocks)

## See Also

- [core/events/types.py](../core/events/types.py) - Event type definitions
- [core/event_emitters.py](../core/event_emitters.py) - Event emission functions
- [core/ui_server/server.py](../core/ui_server/server.py) - API endpoints for events
