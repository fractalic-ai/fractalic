# goto_op.py
from typing import Optional
from core.ast_md.node import Node
from core.ast_md.ast import AST
from core.errors import BlockNotFoundError
from core.config import GOTO_LIMIT
from rich.console import Console

def process_goto(ast: AST, current_node: Node, goto_count: dict) -> Optional[Node]:
    console = Console(force_terminal=True, color_system="auto")
    """Process @goto operation with schema validation support"""
    # Get parameters
    params = current_node.params or {}
    block_params = params.get('block', {})
    
    # Get target block path
    block_uri = block_params.get('block_uri')
    if not block_uri:
        raise ValueError("@goto operation requires 'block' parameter")
        
    # Find target node by searching through AST nodes
    target_node = None
    for node in ast.parser.nodes.values():
        if (node.id and node.id.lower() == block_uri.lower()) or (node.key and node.key.lower() == block_uri.lower()):
            target_node = node
            break
            
    if target_node is None:
        raise BlockNotFoundError(f"Block with path '{block_uri}' not found")
        
    # Print operation for console
    console.print(f"[light_green]→[/light_green] @goto block: '{block_uri}'")

    # Track goto count for infinite loop prevention
    target_key = target_node.key
    goto_count[target_key] = goto_count.get(target_key, 0) + 1
    
    if goto_count[target_key] > GOTO_LIMIT:
        raise RuntimeError(f"@goto operation limit exceeded for block '{block_uri}'")
    
    return target_node