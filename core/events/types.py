"""Event types for Fractalic event system.

All events are sent via HTTP using emit_event() from core.event_emitters.
The emit_event() function is type-agnostic and accepts any payload via **kwargs.

Event structure:
    {
        "type": str,              # Event type from EventType enum
        "execution_id": str,      # Auto-injected from FRACTALIC_EXECUTION_ID env
        "session_id": str,        # Auto-injected, groups related executions
        "timestamp": float,       # Auto-injected Unix timestamp
        **payload                 # Event-specific fields (see below)
    }
"""

from enum import Enum


class EventType(str, Enum):
    """Event types with their expected payload fields.

    Usage: emit_event(EventType.CHAT_MESSAGE, role="user", content="Hello")

    All event types use unique identifiers - no phase/status fields needed.
    """

    # ===== Chat messages =====
    # Payload: role (str), content (str)
    CHAT_MESSAGE = "chat_message"

    # ===== Execution lifecycle (main process) =====
    # Payload: target (Optional[str])
    EXECUTION_START = "execution_start"
    EXECUTION_COMPLETE = "execution_complete"
    EXECUTION_ERROR = "execution_error"

    # ===== Workflow lifecycle (@run operation / agent execution) =====
    # Payload: target (str), parent_execution_id (Optional[str]),
    #          nested_execution_id (str), node_id (Optional[str]),
    #          block_id (Optional[str]), return_content (Optional[str]),
    #          explicit_return (Optional[bool]), error_message (Optional[str])
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"

    # ===== Error events =====
    # Payload: message (str), details (Optional[str])
    ERROR = "error"

    # ===== AST structure updates =====
    # Payload: operation (str), blocks (List[Dict])
    AST_UPDATE = "ast_update"

    # ===== Tool interactions =====
    # Payload: tool_name (str), args (Dict)
    TOOL_CALL = "tool_call"
    # Payload: tool_name (str), result (Any)
    TOOL_RESULT = "tool_result"

    # ===== LLM streaming =====
    # Payload: content (str), block_id (Optional[str])
    LLM_CHUNK = "llm_chunk"

    # ===== Token usage tracking =====
    # TOKEN_USAGE_CALL: Individual LLM call token usage
    # Payload: input_tokens (int), output_tokens (int), model (str),
    #          block_id (Optional[str]), response_cost (Optional[float]),
    #          input_cost (Optional[float]), output_cost (Optional[float]),
    #          tool_usage_cost (Optional[float])
    TOKEN_USAGE_CALL = "token_usage_call"

    # TOKEN_USAGE_SUMMARY: Aggregated token usage for a file/operation
    # Payload: input_tokens (int), output_tokens (int), model (str),
    #          block_id (Optional[str]), source_file (str),
    #          total_input (int), total_output (int),
    #          response_cost (Optional[float])
    TOKEN_USAGE_SUMMARY = "token_usage_summary"

    # ===== Block processing =====
    # Payload: block_id (str), operation (str), status (str)
    BLOCK_PROCESSING = "block_processing"

    # ===== Terminal output =====
    # Payload: data (str), is_stderr (bool)
    TERMINAL_OUTPUT = "terminal_output"
