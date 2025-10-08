# Render AST
# - render_ast_to_markdown

import os
import json
from core.ast_md.ast import AST
from core.ast_md.node import Node, NodeType

class NodeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, NodeType):
            return obj.value
        
        # Handle OpenAI/LiteLLM objects that might be in the AST
        if hasattr(obj, '__class__') and 'ChatCompletion' in obj.__class__.__name__:
            try:
                # Try to convert to dict if possible
                if hasattr(obj, 'model_dump'):
                    return obj.model_dump()
                elif hasattr(obj, 'dict'):
                    return obj.dict()
                elif hasattr(obj, '__dict__'):
                    return {key: self._safe_serialize(value) for key, value in obj.__dict__.items()}
                else:
                    return str(obj)
            except Exception:
                return str(obj)
        
        # Handle other pydantic/dataclass objects
        if hasattr(obj, 'model_dump'):
            try:
                return obj.model_dump()
            except Exception:
                return str(obj)
        elif hasattr(obj, 'dict'):
            try:
                return obj.dict()
            except Exception:
                return str(obj)
        
        # Handle datetime objects
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
            
        # Default fallback - convert to string
        try:
            return str(obj)
        except Exception:
            return f"<unserializable {obj.__class__.__name__}>"
    
    def _safe_serialize(self, obj):
        """Safely serialize an object, falling back to string representation"""
        try:
            # Try normal JSON serialization first
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If that fails, use our custom serialization
            return self.default(obj)

# it soesnt grab header while using content

def render_ast_to_markdown_string(ast: AST) -> str:
    """Render AST to markdown string (for storage API)."""
    lines = []
    current = ast.first()
    while current:
        lines.append(''.join(f"{line}\n" for line in current.content.splitlines())+"\n")
        current = current.next
    return ''.join(lines)

def render_ast_to_markdown(ast: AST, output_file: str = "out.ctx") -> None:
    """Render AST to markdown file (legacy compatibility)."""
    content = render_ast_to_markdown_string(ast)
    with open(output_file, 'w') as f:
        f.write(content)

def render_ast_to_trace_string(ast: AST) -> str:
    """Render AST to trace JSON string (for storage API)."""
    nodes = []
    current_node = ast.first()

    while current_node:
        node_data = current_node.__dict__.copy()
        node_data['prev'] = current_node.prev.key if current_node.prev else None
        node_data['next'] = current_node.next.key if current_node.next else None
        nodes.append(node_data)
        current_node = current_node.next

    return json.dumps(nodes, ensure_ascii=False, indent=4, cls=NodeEncoder)

def render_ast_to_trace(ast: AST, output_file: str) -> None:
    """Render AST to trace JSON file (legacy compatibility)."""
    content = render_ast_to_trace_string(ast)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

# Example usage
# ast = AST("...")  # Assuming you have an AST object
# render_ast_to_trace(ast, "output_trace.json")

