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
    # Payload: target (Optional[str]), return_content (Optional[str])
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
    # Payload: input_tokens (int), output_tokens (int), model (str),
    #          block_id (Optional[str]), file_path (Optional[str])
    TOKEN_USAGE = "token_usage"

    # ===== Block processing =====
    # Payload: block_id (str), operation (str), status (str)
    BLOCK_PROCESSING = "block_processing"
