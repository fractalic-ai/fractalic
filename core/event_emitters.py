# Event emission functions for UI communication

import os
import time
from core.events.types import EventType

try:
    import requests
except Exception:
    requests = None


def _session_id():
    """Get or generate session_id for grouping related executions."""
    # Use execution_id as session_id for now (one session = one execution)
    # In future, could use FRACTALIC_SESSION_ID for multi-execution sessions
    return os.getenv('FRACTALIC_EXECUTION_ID')


def emit_event(event_type: EventType, nested_execution_id: str = None, **data):
    """Universal event emission function.

    Args:
        event_type: Type of event (EventType enum value)
        nested_execution_id: Optional nested execution ID for routing events to child bubbles (explicitly passed, not from env)
        **data: Event-specific data fields
    """
    execution_id = os.getenv('FRACTALIC_EXECUTION_ID')
    if not execution_id:
        # Events without execution_id are local-only (not sent to server)
        return

    # Use explicitly passed nested_execution_id instead of reading from environment
    # This prevents race conditions with finally blocks that cleanup env vars
    if nested_execution_id:
        event = {
            'type': event_type.value if isinstance(event_type, EventType) else event_type,
            'execution_id': nested_execution_id,  # Route to nested bubble
            'parent_execution_id': execution_id,   # Keep reference to parent
            'session_id': _session_id(),
            'timestamp': time.time(),
            **data
        }
    else:
        event = {
            'type': event_type.value if isinstance(event_type, EventType) else event_type,
            'execution_id': execution_id,
            'session_id': _session_id(),
            'timestamp': time.time(),
            **data
        }

    _stream_event_to_server(event)


def emit_ast_snapshot(ast, operation_type: str = "unknown", changed_blocks: list = None, nested_execution_id: str = None):
    """Send AST structure snapshot to UI for visualization.

    Args:
        ast: AST object to extract blocks from
        operation_type: Type of operation that modified AST (llm, shell, import, return, parse)
        changed_blocks: List of block IDs that were added/modified
        nested_execution_id: Optional nested execution ID for routing to child bubbles
    """
    execution_id = os.getenv('FRACTALIC_EXECUTION_ID')
    if not execution_id:
        print(f"[DEBUG emit_ast_snapshot] No execution_id, skipping AST snapshot")
        return

    from core.ast_md.node import NodeType

    blocks = []
    current_node = ast.first()
    print(f"[DEBUG emit_ast_snapshot] Starting AST snapshot for operation: {operation_type}")

    try:
        while current_node:
            # Extract block information
            node_type = current_node.type
            if hasattr(node_type, 'value'):
                type_str = str(node_type.value)
            else:
                type_str = str(node_type)

            block_info = {
                'id': current_node.key or current_node.hash,
                'type': type_str,
                'header': None,
                'content': None,  # Full content - frontend will truncate for preview
                'created_by': getattr(current_node, 'created_by', None),
                'parent_id': None,  # Could be extracted from parent relationships if needed
                'is_new': False,
                'is_modified': False
            }

            # Add header for HEADING nodes
            if type_str == 'heading':
                # Extract just the header line (first line) without the markdown #
                content_lines = current_node.content.split('\n')
                if content_lines:
                    header_line = content_lines[0].strip()
                    # Remove markdown # symbols
                    header_line = header_line.lstrip('#').strip()
                    block_info['header'] = header_line[:100] if header_line else None

            # Add content (skip header for heading nodes)
            if current_node.content:
                if type_str == 'heading':
                    # For heading nodes, skip the first line (header) and show the rest
                    content_lines = current_node.content.split('\n', 1)
                    content_text = content_lines[1] if len(content_lines) > 1 else ''
                else:
                    # For other nodes, show full content
                    content_text = current_node.content

                # Send full content - frontend will create preview
                block_info['content'] = content_text.strip() if content_text else None

            # Mark as new/modified if in changed_blocks list
            if changed_blocks and block_info['id'] in changed_blocks:
                block_info['is_new'] = True

            blocks.append(block_info)
            current_node = current_node.next
    except Exception as e:
        print(f"[ERROR emit_ast_snapshot] Error processing AST: {e}")
        import traceback
        traceback.print_exc()
        return  # Don't send partial data

    print(f"[DEBUG emit_ast_snapshot] Emitting {len(blocks)} AST blocks for operation '{operation_type}'")
    emit_event(EventType.AST_UPDATE, nested_execution_id=nested_execution_id, operation=operation_type, blocks=blocks)


def _stream_event_to_server(event):
    """Stream event to server.py via HTTP POST."""
    if not requests:
        return

    try:
        server_url = os.getenv('FRACTALIC_SERVER_URL', 'http://localhost:8000')
        response = requests.post(
            f"{server_url}/api/events/receive",
            json=event,
            timeout=1.0,
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code != 200:
            print(f"[Event Stream Warning] Server returned {response.status_code}")
    except Exception as e:
        # Ignore streaming errors - fractalic should work even without server
        pass
