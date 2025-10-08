# call_tree.py

import json
import uuid
import os
import hashlib
from typing import Optional, List, Dict, Any


class CallTreeNode:
    """
    Node in the call tree representing a single execution of a file.

    This class has been extended to support the new storage architecture while
    maintaining backward compatibility with the old Git-based approach.

    New fields:
        node_id: Unique identifier for this node (solves multiple runs of same file)
        execution_id: Session identifier this node belongs to
        original_file_path: Original file path (relative to source_root)
        workspace_file_path: Workspace file path (absolute)
        workspace_cwd: Working directory when file was executed
        artifacts_dir: Directory where this node's artifacts are stored
        tools_resolved: List of tools directories available during execution

    Legacy fields (for backward compatibility):
        operation: Operation type (e.g., @run)
        operation_src: Source operation that triggered this
        filename: File being processed (legacy, use original_file_path)
        md_commit_hash: Commit hash for .md file (now content hash)
        ctx_commit_hash: Commit hash for .ctx file (now content hash)
        ctx_file: Path to .ctx file (legacy, use artifacts_dir)
        trc_commit_hash: Commit hash for .trc file (now content hash)
        trc_file: Path to .trc file (legacy, use artifacts_dir)
    """

    def __init__(self,
                 operation: str,
                 operation_src: Optional[str],
                 filename: str,
                 md_commit_hash: Optional[str] = None,
                 ctx_commit_hash: Optional[str] = None,
                 ctx_file: Optional[str] = None,
                 trc_commit_hash: Optional[str] = None,
                 trc_file: Optional[str] = None,
                 parent: Optional['CallTreeNode'] = None,
                 # New parameters
                 node_id: Optional[str] = None,
                 execution_id: Optional[str] = None,
                 original_file_path: Optional[str] = None,
                 workspace_file_path: Optional[str] = None,
                 workspace_cwd: Optional[str] = None,
                 tools_resolved: Optional[List[str]] = None):

        # Legacy fields (maintained for backward compatibility)
        self.operation = operation  # The operation performed, e.g., @run
        self.operation_src = operation_src
        self.filename = filename  # The file being processed (legacy)
        self.md_commit_hash = md_commit_hash  # Commit hash for .md file
        self.ctx_commit_hash = ctx_commit_hash  # Commit hash for .ctx file
        self.ctx_file = ctx_file  # The .ctx file being processed
        self.trc_commit_hash = trc_commit_hash  # Commit hash for .trc file
        self.trc_file = trc_file  # The .trc file being processed
        self.children = []  # Children nodes
        self.parent = parent  # The parent node

        # New fields for storage architecture
        self.node_id = node_id or str(uuid.uuid4())[:12]  # Generate if not provided
        self.execution_id = execution_id or os.getenv('FRACTALIC_EXECUTION_ID')

        # Path tracking
        self.original_file_path = original_file_path or filename  # Original path
        self.workspace_file_path = workspace_file_path  # Workspace path (set during execution)
        self.workspace_cwd = workspace_cwd  # CWD during execution

        # Artifacts directory (node-based storage)
        self.artifacts_dir = f"artifacts/nodes/{self.node_id}"

        # Tools context
        self.tools_resolved = tools_resolved or []

        # Timestamp
        from datetime import datetime
        self.timestamp = datetime.now().isoformat()

    def add_child(self, child_node: 'CallTreeNode'):
        """Add a child node to this node."""
        self.children.append(child_node)

    def get_artifact_path(self, artifact_type: str) -> str:
        """
        Get the path for a specific artifact type.

        Args:
            artifact_type: Type of artifact (ctx, trc, source_snapshot, metadata)

        Returns:
            str: Path to artifact relative to session root

        Example:
            node.get_artifact_path('ctx')
            → 'artifacts/nodes/abc123def456/result.ctx'
        """
        artifact_filenames = {
            'ctx': 'result.ctx',
            'trc': 'result.trc',
            'source_snapshot': 'source_snapshot.md',
            'metadata': 'metadata.json'
        }

        filename = artifact_filenames.get(artifact_type, f"{artifact_type}.dat")
        return f"{self.artifacts_dir}/{filename}"

    def compute_content_hash(self, content: str) -> str:
        """
        Compute content hash for Git API compatibility.

        This creates a hash that can be used as a "commit hash" in the legacy API.
        The hash is computed as: SHA256(node_id + ":" + content)

        Args:
            content: Content to hash

        Returns:
            str: SHA256 hash (40 characters, compatible with Git)
        """
        data = f"{self.node_id}:{content}".encode('utf-8')
        return hashlib.sha256(data).hexdigest()[:40]

    def set_artifact_paths_from_storage(self, ctx_path: str, trc_path: str):
        """
        Update legacy artifact paths from storage paths.

        This maintains backward compatibility by setting ctx_file and trc_file
        to the node-based paths.

        Args:
            ctx_path: Path to .ctx artifact
            trc_path: Path to .trc artifact
        """
        self.ctx_file = ctx_path
        self.trc_file = trc_path

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Includes both legacy and new fields for maximum compatibility.
        """
        return {
            # Legacy fields
            "operation": self.operation,
            "operation_src": self.operation_src,
            "filename": self.filename,
            "ctx_file": self.ctx_file,
            "md_commit_hash": self.md_commit_hash,
            "ctx_commit_hash": self.ctx_commit_hash,
            "trc_file": self.trc_file,
            "trc_commit_hash": self.trc_commit_hash,

            # New fields
            "node_id": self.node_id,
            "execution_id": self.execution_id,
            "original_file_path": self.original_file_path,
            "workspace_file_path": self.workspace_file_path,
            "workspace_cwd": self.workspace_cwd,
            "artifacts_dir": self.artifacts_dir,
            "tools_resolved": self.tools_resolved,
            "timestamp": self.timestamp,

            # Children (recursive)
            "children": [child.to_dict() for child in self.children],

            # Debug info
            "node_python": str(self),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=4)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CallTreeNode':
        """
        Create CallTreeNode from dictionary (deserialization).

        Supports both legacy and new formats for backward compatibility.

        Args:
            data: Dictionary with node data

        Returns:
            CallTreeNode: Reconstructed node
        """
        # Create node with both legacy and new parameters
        node = CallTreeNode(
            operation=data.get('operation', '@run'),
            operation_src=data.get('operation_src'),
            filename=data.get('filename', ''),
            md_commit_hash=data.get('md_commit_hash'),
            ctx_commit_hash=data.get('ctx_commit_hash'),
            ctx_file=data.get('ctx_file'),
            trc_commit_hash=data.get('trc_commit_hash'),
            trc_file=data.get('trc_file'),
            parent=None,  # Parent set later if needed
            # New fields (with fallbacks for legacy data)
            node_id=data.get('node_id'),
            execution_id=data.get('execution_id'),
            original_file_path=data.get('original_file_path'),
            workspace_file_path=data.get('workspace_file_path'),
            workspace_cwd=data.get('workspace_cwd'),
            tools_resolved=data.get('tools_resolved', [])
        )

        # Restore timestamp if available
        if 'timestamp' in data:
            node.timestamp = data['timestamp']

        # Recursively create children
        for child_data in data.get('children', []):
            child = CallTreeNode.from_dict(child_data)
            child.parent = node
            node.add_child(child)

        return node

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"CallTreeNode(node_id={self.node_id}, file={self.original_file_path})"
