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
        return json.JSONEncoder.default(self, obj)

# it soesnt grab header while using content

def render_ast_to_markdown(ast: AST, output_file: str = "out.ctx") -> None:
    with open(output_file, 'w') as f:
        current = ast.first()
        while current:
            f.write(''.join(f"{line}\n" for line in current.content.splitlines())+"\n")
            current = current.next

def render_ast_to_trace(ast: AST, output_file: str) -> None:
    """
    Traverse through AST blocks and write a JSON file with full set of fields of the node and their values.
    """
    nodes = []
    current_node = ast.first()
    
    while current_node:
        node_data = current_node.__dict__.copy()
        node_data['prev'] = current_node.prev.key if current_node.prev else None
        node_data['next'] = current_node.next.key if current_node.next else None
        nodes.append(node_data)
        current_node = current_node.next
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, ensure_ascii=False, indent=4, cls=NodeEncoder)

# Example usage
# ast = AST("...")  # Assuming you have an AST object
# render_ast_to_trace(ast, "output_trace.json")

