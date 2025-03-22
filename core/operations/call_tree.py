# call_tree.py

import json

class CallTreeNode:
    def __init__(self, operation: str, operation_src: str, filename: str, md_commit_hash: str = None, ctx_commit_hash: str = None,  ctx_file: str = None, trc_commit_hash: str = None, trc_file: str = None, parent=None):
            self.operation = operation  # The operation performed, e.g., @run
            self.operation_src   =   operation_src,
            self.filename = filename  # The file being processed
            self.md_commit_hash = md_commit_hash  # Commit hash for .md file
            self.ctx_commit_hash = ctx_commit_hash  # Commit hash for .ctx file
            self.ctx_file = ctx_file  # The .ctx file being processed
            self.trc_commit_hash = trc_commit_hash  # Commit hash for .trc file
            self.trc_file = trc_file  # The .trc file being processed
            self.children = []  # Children nodes
            self.parent = parent  # The parent node

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        return {
            
            "operation": self.operation,
            "operation_src": self.operation_src,
            "filename": self.filename,
            "ctx_file": self.ctx_file,
            "md_commit_hash": self.md_commit_hash,
            "ctx_commit_hash": self.ctx_commit_hash,
            "trc_file": self.trc_file,
            "trc_commit_hash": self.trc_commit_hash,
            "children": [child.to_dict() for child in self.children],
            "node_python" : str(self),
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
