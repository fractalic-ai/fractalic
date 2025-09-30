# Event emission functions for UI communication

import os
import time

try:
    import requests
except Exception:
    requests = None


def _session_id():
    """Get or generate session_id for grouping related executions."""
    # Use execution_id as session_id for now (one session = one execution)
    # In future, could use FRACTALIC_SESSION_ID for multi-execution sessions
    return os.getenv('FRACTALIC_EXECUTION_ID')


def emit_ast_snapshot(ast, operation_type: str = "unknown", changed_blocks: list = None):
    """Send AST structure snapshot to UI for visualization.

    Args:
        ast: AST object to extract blocks from
        operation_type: Type of operation that modified AST (llm, shell, import, return, parse)
        changed_blocks: List of block IDs that were added/modified
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
                'content_preview': None,
                'created_by': getattr(current_node, 'created_by', None),
                'parent_id': None,  # Could be extracted from parent relationships if needed
                'is_new': False,
                'is_modified': False
            }

            # Add header for HEADING nodes
            if type_str == 'heading':
                block_info['header'] = current_node.content[:100] if current_node.content else None

            # Add content preview (first 100 chars)
            if current_node.content:
                preview = current_node.content.strip()[:100]
                if len(current_node.content.strip()) > 100:
                    preview += '...'
                block_info['content_preview'] = preview

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

    event = {
        'type': 'ast_update',
        'execution_id': execution_id,
        'session_id': _session_id(),
        'timestamp': time.time(),
        'operation': operation_type,
        'blocks': blocks
    }

    print(f"[DEBUG emit_ast_snapshot] Emitting {len(blocks)} AST blocks for operation '{operation_type}'")
    _stream_event_to_server(event)


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
