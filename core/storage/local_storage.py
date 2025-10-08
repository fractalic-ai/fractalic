"""
Local filesystem-based session storage implementation.

This module implements SessionStorage using the local filesystem with
the following structure:

.fractalic/
  sessions/
    {execution_id}/
      workspace/              # Isolated copy of source directory
      artifacts/nodes/        # Node-based artifacts
      metadata/              # Session and call tree metadata
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base import SessionStorage
from .session_context import SessionContext, SessionMetadata, SessionInfo, NodeMetadata


class LocalFileStorage(SessionStorage):
    """
    Local filesystem storage implementation.

    Stores sessions in .fractalic/sessions/ directory with full workspace isolation.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize local file storage.

        Args:
            base_dir: Base directory for sessions (defaults to .fractalic in current session_root)
        """
        if base_dir is None:
            # Default to .fractalic in session_root (where script was run from)
            from core.paths import get_session_root
            session_root = get_session_root()
            base_dir = Path(session_root) / ".fractalic"

        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def get_sessions_dir(self) -> Path:
        """Get the root directory where sessions are stored."""
        return self.sessions_dir

    def create_session(self, execution_id: str, source_dir: Path,
                      initial_file: Optional[str] = None,
                      settings: Optional[Dict[str, Any]] = None) -> SessionContext:
        """
        Create a new isolated session.

        Implementation:
        1. Creates session directory structure
        2. Copies source_dir to workspace preserving structure
        3. Creates artifacts and metadata directories
        4. Saves session metadata
        """
        source_dir = Path(source_dir).resolve()

        if not source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        session_dir = self.sessions_dir / execution_id

        if session_dir.exists():
            raise ValueError(f"Session already exists: {execution_id}")

        # Create directory structure
        workspace_dir = session_dir / "workspace"
        artifacts_dir = session_dir / "artifacts"
        metadata_dir = session_dir / "metadata"

        workspace_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "nodes").mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Copy source directory to workspace
        # Use shutil.copytree with dirs_exist_ok for robustness
        try:
            # Copy all contents from source_dir to workspace_dir
            for item in source_dir.iterdir():
                src_path = source_dir / item.name
                dst_path = workspace_dir / item.name

                if src_path.is_dir():
                    # Skip .git and .fractalic directories
                    if item.name in ['.git', '.fractalic']:
                        continue
                    shutil.copytree(src_path, dst_path, symlinks=False,
                                  ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
                else:
                    shutil.copy2(src_path, dst_path)

        except Exception as e:
            # Clean up on failure
            if session_dir.exists():
                shutil.rmtree(session_dir)
            raise IOError(f"Failed to copy workspace: {e}")

        # Create session metadata
        metadata = SessionMetadata(
            execution_id=execution_id,
            source_root=str(source_dir),
            workspace_root=str(workspace_dir),
            created_at=datetime.now().isoformat(),
            status="running",
            initial_file=initial_file,
            settings=settings
        )

        # Save metadata
        metadata_path = metadata_dir / "session.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Create path mapping for reference
        path_mapping = {
            'source_root': str(source_dir),
            'workspace_root': str(workspace_dir),
            'created_at': metadata.created_at
        }

        mapping_path = metadata_dir / "path_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(path_mapping, f, indent=2)

        return SessionContext(
            execution_id=execution_id,
            workspace_dir=workspace_dir,
            artifacts_dir=artifacts_dir,
            source_dir=source_dir,
            metadata=metadata
        )

    def get_session_context(self, execution_id: str) -> SessionContext:
        """Get context for an existing session."""
        session_dir = self.sessions_dir / execution_id

        if not session_dir.exists():
            raise ValueError(f"Session does not exist: {execution_id}")

        workspace_dir = session_dir / "workspace"
        artifacts_dir = session_dir / "artifacts"
        metadata_dir = session_dir / "metadata"

        # Load metadata
        metadata_path = metadata_dir / "session.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                metadata = SessionMetadata.from_dict(metadata_dict)
        else:
            metadata = None

        source_dir = Path(metadata.source_root) if metadata else workspace_dir

        return SessionContext(
            execution_id=execution_id,
            workspace_dir=workspace_dir,
            artifacts_dir=artifacts_dir,
            source_dir=source_dir,
            metadata=metadata
        )

    def save_node_artifact(self, execution_id: str, node_id: str,
                          artifact_type: str, content: str,
                          metadata: Optional[NodeMetadata] = None,
                          filename: Optional[str] = None) -> str:
        """
        Save an artifact for a call tree node.

        Creates directory: artifacts/nodes/{node_id}/
        Saves artifact with provided filename or default naming
        """
        session_ctx = self.get_session_context(execution_id)
        node_dir = session_ctx.artifacts_dir / "nodes" / node_id
        node_dir.mkdir(parents=True, exist_ok=True)

        # Get artifact filename (use custom filename if provided)
        artifact_path = session_ctx.get_artifact_path(node_id, artifact_type, filename)

        # Save artifact content
        with open(artifact_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Save node metadata if provided
        if metadata:
            metadata_path = session_ctx.get_artifact_path(node_id, 'metadata')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2)

        # Return relative path from session root
        return str(artifact_path.relative_to(self.sessions_dir / execution_id))

    def get_node_artifact(self, execution_id: str, node_id: str,
                         artifact_type: str) -> str:
        """Get an artifact for a call tree node."""
        session_ctx = self.get_session_context(execution_id)
        artifact_path = session_ctx.get_artifact_path(node_id, artifact_type)

        if not artifact_path.exists():
            raise ValueError(f"Artifact not found: {node_id}/{artifact_type}")

        with open(artifact_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_artifact_by_content_hash(self, execution_id: str,
                                    content_hash: str,
                                    file_path: str) -> str:
        """
        Get an artifact by its content hash.

        This is used for Git API compatibility. We search through all nodes
        to find one with matching content hash.
        """
        session_ctx = self.get_session_context(execution_id)
        nodes_dir = session_ctx.artifacts_dir / "nodes"

        if not nodes_dir.exists():
            raise ValueError(f"No artifacts found for session: {execution_id}")

        # Search through all nodes
        for node_dir in nodes_dir.iterdir():
            if not node_dir.is_dir():
                continue

            node_id = node_dir.name

            # Try to get the requested artifact type from file_path
            # e.g., "result.ctx" → "ctx", "result.trc" → "trc"
            artifact_type = self._infer_artifact_type(file_path)

            try:
                content = self.get_node_artifact(execution_id, node_id, artifact_type)

                # Compute content hash
                computed_hash = self._compute_content_hash(node_id, content)

                if computed_hash == content_hash or computed_hash.startswith(content_hash):
                    return content

            except ValueError:
                # Artifact doesn't exist for this node, continue searching
                continue

        raise ValueError(f"Artifact not found for content hash: {content_hash}")

    def save_call_tree(self, execution_id: str, call_tree: Dict[str, Any]):
        """Save the call tree for a session."""
        session_ctx = self.get_session_context(execution_id)
        call_tree_path = session_ctx.metadata.workspace_root
        metadata_dir = Path(call_tree_path).parent / "metadata"

        call_tree_file = metadata_dir / "call_tree.json"

        with open(call_tree_file, 'w', encoding='utf-8') as f:
            json.dump(call_tree, f, indent=2)

    def get_call_tree(self, execution_id: str) -> Dict[str, Any]:
        """Get the call tree for a session."""
        session_ctx = self.get_session_context(execution_id)
        call_tree_path = session_ctx.metadata.workspace_root
        metadata_dir = Path(call_tree_path).parent / "metadata"

        call_tree_file = metadata_dir / "call_tree.json"

        if not call_tree_file.exists():
            raise ValueError(f"Call tree not found for session: {execution_id}")

        with open(call_tree_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_sessions(self, source_dir: Optional[Path] = None) -> List[SessionInfo]:
        """List all sessions, optionally filtered by source directory."""
        sessions = []

        if not self.sessions_dir.exists():
            return sessions

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            execution_id = session_dir.name
            metadata_file = session_dir / "metadata" / "session.json"

            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                    metadata = SessionMetadata.from_dict(metadata_dict)

                # Filter by source_dir if provided
                if source_dir:
                    if Path(metadata.source_root).resolve() != Path(source_dir).resolve():
                        continue

                session_info = SessionInfo(
                    execution_id=metadata.execution_id,
                    created_at=metadata.created_at,
                    status=metadata.status,
                    initial_file=metadata.initial_file,
                    source_root=metadata.source_root
                )

                sessions.append(session_info)

            except (json.JSONDecodeError, KeyError) as e:
                # Skip invalid metadata files
                print(f"[WARNING] Failed to load session metadata for {execution_id}: {e}")
                continue

        # Sort by created_at (newest first)
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        return sessions

    def update_session_status(self, execution_id: str, status: str,
                             completed_at: Optional[str] = None):
        """Update session status."""
        session_ctx = self.get_session_context(execution_id)

        if session_ctx.metadata is None:
            raise ValueError(f"No metadata found for session: {execution_id}")

        # Update metadata
        session_ctx.metadata.status = status
        if completed_at:
            session_ctx.metadata.completed_at = completed_at
        elif status in ['completed', 'failed']:
            session_ctx.metadata.completed_at = datetime.now().isoformat()

        # Save updated metadata
        metadata_dir = Path(session_ctx.workspace_dir).parent / "metadata"
        metadata_path = metadata_dir / "session.json"

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(session_ctx.metadata.to_dict(), f, indent=2)

    def cleanup_session(self, execution_id: str):
        """Delete a session and all its data."""
        session_dir = self.sessions_dir / execution_id

        if not session_dir.exists():
            raise ValueError(f"Session does not exist: {execution_id}")

        shutil.rmtree(session_dir)

    # Private helper methods

    def _infer_artifact_type(self, file_path: str) -> str:
        """
        Infer artifact type from file path.

        Examples:
            "result.ctx" → "ctx"
            "agents/helper.ctx" → "ctx"
            "result.trc" → "trc"
        """
        if '.ctx' in file_path:
            return 'ctx'
        elif '.trc' in file_path:
            return 'trc'
        elif '.md' in file_path:
            return 'source_snapshot'
        else:
            return 'ctx'  # Default

    def _compute_content_hash(self, node_id: str, content: str) -> str:
        """
        Compute content hash for Git API compatibility.

        Hash is computed as: SHA256(node_id + ":" + content)
        """
        data = f"{node_id}:{content}".encode('utf-8')
        return hashlib.sha256(data).hexdigest()
