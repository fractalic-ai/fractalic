# emit_op.py
from typing import Optional
from core.ast_md.ast import AST
from core.ast_md.node import Node
from core.event_emitters import emit_event
from rich.console import Console

def process_emit(ast: AST, current_node: Node, nested_execution_id: str = None) -> Optional[Node]:
    """Process @emit operation - send custom event to frontend

    Args:
        ast: Current AST
        current_node: Current operation node
        nested_execution_id: Optional nested execution ID for routing

    Returns:
        Next node to process
    """
    console = Console(force_terminal=True, color_system="auto")
    params = current_node.params
    if not params:
        raise ValueError("No parameters found for @emit operation")

    # Get event type (required)
    event_type = params.get('event')
    if not event_type:
        raise ValueError("@emit operation requires 'event' parameter")

    # Get optional data and prompt
    event_data = params.get('data', {})
    prompt_text = params.get('prompt')

    # Add prompt to data if provided
    if prompt_text:
        event_data['message'] = prompt_text

    # Inject workspace_path for proper image resolution
    # os.getcwd() returns workspace because fractalic.py does os.chdir(workspace_dir) at line 205
    import os
    workspace_dir = os.getcwd()
    if workspace_dir:
        event_data['_workspace_path'] = str(workspace_dir)

    # Emit the custom event using the event emitters infrastructure
    # The event will be routed through the HTTP event system to the frontend
    emit_event(
        event_type,
        nested_execution_id=nested_execution_id,
        **event_data
    )

    # Log operation for debugging
    console.print(f"[light_green]→[/light_green] @emit event: '[cyan]{event_type}[/cyan]'")
    if event_data:
        data_preview = str(event_data)[:100]
        if len(str(event_data)) > 100:
            data_preview += '...'
        console.print(f"  Data: {data_preview}")

    # Continue to next node
    return current_node.next
