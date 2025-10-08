"""
Session context and metadata classes.

These classes represent the structure and metadata of Fractalic execution sessions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class SessionContext:
    """
    Context for an active Fractalic session.

    Attributes:
        execution_id: Unique identifier for this session
        workspace_dir: Directory where files are executed (isolated copy)
        artifacts_dir: Directory where execution artifacts are stored
        source_dir: Original source directory (read-only reference)
        metadata: Session metadata object
    """
    execution_id: str
    workspace_dir: Path
    artifacts_dir: Path
    source_dir: Path
    metadata: Optional['SessionMetadata'] = None

    def get_workspace_path(self, relative_path: str) -> Path:
        """
        Get workspace path for a file.

        Args:
            relative_path: Path relative to source_dir

        Returns:
            Path: Absolute path in workspace

        Example:
            ctx.get_workspace_path("agents/helper.md")
            → /path/to/.fractalic/sessions/{exec_id}/workspace/agents/helper.md
        """
        return self.workspace_dir / relative_path

    def get_artifact_path(self, node_id: str, artifact_type: str, filename: Optional[str] = None) -> Path:
        """
        Get path for a node artifact.

        Args:
            node_id: Unique identifier for the call tree node
            artifact_type: Type of artifact (ctx, trc, source_snapshot, metadata)
            filename: Optional custom filename (if not provided, uses defaults)

        Returns:
            Path: Absolute path to artifact

        Example:
            ctx.get_artifact_path("abc123", "ctx", "test.ctx")
            → /path/to/.fractalic/sessions/{exec_id}/artifacts/nodes/abc123/test.ctx
        """
        node_dir = self.artifacts_dir / "nodes" / node_id

        # If filename provided, use it
        if filename:
            return node_dir / filename

        # Otherwise use defaults
        artifact_filenames = {
            'source_snapshot': 'source_snapshot.md',
            'metadata': 'metadata.json'
        }

        default_filename = artifact_filenames.get(artifact_type, f"{artifact_type}.dat")
        return node_dir / default_filename


@dataclass
class SessionMetadata:
    """
    Metadata for a Fractalic session.

    Attributes:
        execution_id: Unique identifier for this session
        source_root: Original source directory
        workspace_root: Workspace directory
        created_at: ISO timestamp when session was created
        completed_at: ISO timestamp when session completed (None if running)
        status: Session status (running, completed, failed)
        initial_file: Initial .md file that started the session
        settings: Copy of settings used for this session
        extra: Additional metadata fields
    """
    execution_id: str
    source_root: str
    workspace_root: str
    created_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    initial_file: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'execution_id': self.execution_id,
            'source_root': self.source_root,
            'workspace_root': self.workspace_root,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'status': self.status,
            'initial_file': self.initial_file,
            'settings': self.settings,
            'extra': self.extra,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SessionMetadata':
        """Create from dictionary (deserialization)."""
        return SessionMetadata(
            execution_id=data['execution_id'],
            source_root=data['source_root'],
            workspace_root=data['workspace_root'],
            created_at=data['created_at'],
            completed_at=data.get('completed_at'),
            status=data.get('status', 'running'),
            initial_file=data.get('initial_file'),
            settings=data.get('settings'),
            extra=data.get('extra', {}),
        )


@dataclass
class SessionInfo:
    """
    Brief information about a session (for listing).

    Attributes:
        execution_id: Unique identifier for this session
        created_at: ISO timestamp when created
        status: Session status
        initial_file: Initial file that started the session
        source_root: Original source directory
    """
    execution_id: str
    created_at: str
    status: str
    initial_file: Optional[str] = None
    source_root: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_id': self.execution_id,
            'created_at': self.created_at,
            'status': self.status,
            'initial_file': self.initial_file,
            'source_root': self.source_root,
        }


@dataclass
class NodeMetadata:
    """
    Metadata for a single node in the call tree.

    Attributes:
        node_id: Unique identifier for this node
        execution_id: Session this node belongs to
        original_path: Original file path (relative to source_root)
        workspace_path: Workspace file path (absolute)
        workspace_cwd: Working directory when file was executed
        tools_resolved: List of tools directories available
        timestamp: When this node was executed
        parent_node_id: Parent node in call tree (None for root)
        extra: Additional metadata
    """
    node_id: str
    execution_id: str
    original_path: str
    workspace_path: str
    workspace_cwd: str
    tools_resolved: list = field(default_factory=list)
    timestamp: Optional[str] = None
    parent_node_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'node_id': self.node_id,
            'execution_id': self.execution_id,
            'original_path': self.original_path,
            'workspace_path': self.workspace_path,
            'workspace_cwd': self.workspace_cwd,
            'tools_resolved': self.tools_resolved,
            'timestamp': self.timestamp or datetime.now().isoformat(),
            'parent_node_id': self.parent_node_id,
            'extra': self.extra,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'NodeMetadata':
        """Create from dictionary."""
        return NodeMetadata(
            node_id=data['node_id'],
            execution_id=data['execution_id'],
            original_path=data['original_path'],
            workspace_path=data['workspace_path'],
            workspace_cwd=data['workspace_cwd'],
            tools_resolved=data.get('tools_resolved', []),
            timestamp=data.get('timestamp'),
            parent_node_id=data.get('parent_node_id'),
            extra=data.get('extra', {}),
        )
