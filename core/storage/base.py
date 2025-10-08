"""
Abstract base class for session storage.

This module defines the interface that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
from .session_context import SessionContext, SessionInfo, NodeMetadata


class SessionStorage(ABC):
    """
    Abstract base class for Fractalic session storage.

    This class defines the interface for storing and retrieving session data,
    artifacts, and metadata. Implementations can use different backends:
    - Local filesystem
    - S3/object storage
    - Database
    - Hybrid approaches
    """

    @abstractmethod
    def create_session(self, execution_id: str, source_dir: Path,
                      initial_file: Optional[str] = None,
                      settings: Optional[Dict[str, Any]] = None) -> SessionContext:
        """
        Create a new isolated session.

        This method should:
        1. Create session directory structure
        2. Copy source_dir to workspace (preserving structure)
        3. Create artifacts and metadata directories
        4. Save session metadata

        Args:
            execution_id: Unique identifier for this session
            source_dir: Original source directory to copy
            initial_file: Initial .md file being executed
            settings: Settings to save with session

        Returns:
            SessionContext: Context object for the created session

        Raises:
            ValueError: If session already exists
            IOError: If filesystem operations fail
        """
        pass

    @abstractmethod
    def get_session_context(self, execution_id: str) -> SessionContext:
        """
        Get context for an existing session.

        Args:
            execution_id: Session identifier

        Returns:
            SessionContext: Session context

        Raises:
            ValueError: If session doesn't exist
        """
        pass

    @abstractmethod
    def save_node_artifact(self, execution_id: str, node_id: str,
                          artifact_type: str, content: str,
                          metadata: Optional[NodeMetadata] = None,
                          filename: Optional[str] = None) -> str:
        """
        Save an artifact for a call tree node.

        Args:
            execution_id: Session identifier
            node_id: Unique node identifier
            artifact_type: Type of artifact (ctx, trc, source_snapshot)
            content: Artifact content
            metadata: Node metadata to save
            filename: Optional custom filename (e.g., "test.ctx")

        Returns:
            str: Path to saved artifact (relative to session root)

        Raises:
            ValueError: If session doesn't exist
            IOError: If save fails
        """
        pass

    @abstractmethod
    def get_node_artifact(self, execution_id: str, node_id: str,
                         artifact_type: str) -> str:
        """
        Get an artifact for a call tree node.

        Args:
            execution_id: Session identifier
            node_id: Node identifier
            artifact_type: Type of artifact (ctx, trc, source_snapshot)

        Returns:
            str: Artifact content

        Raises:
            ValueError: If session or artifact doesn't exist
        """
        pass

    @abstractmethod
    def get_artifact_by_content_hash(self, execution_id: str,
                                    content_hash: str,
                                    file_path: str) -> str:
        """
        Get an artifact by its content hash.

        This is used for Git API compatibility - the "commit hash"
        in the old system is actually a content hash.

        Args:
            execution_id: Session identifier
            content_hash: SHA hash of content
            file_path: Path to artifact (for disambiguation)

        Returns:
            str: Artifact content

        Raises:
            ValueError: If not found
        """
        pass

    @abstractmethod
    def save_call_tree(self, execution_id: str, call_tree: Dict[str, Any]):
        """
        Save the call tree for a session.

        Args:
            execution_id: Session identifier
            call_tree: Call tree dictionary (from CallTreeNode.to_dict())

        Raises:
            ValueError: If session doesn't exist
        """
        pass

    @abstractmethod
    def get_call_tree(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the call tree for a session.

        Args:
            execution_id: Session identifier

        Returns:
            dict: Call tree dictionary

        Raises:
            ValueError: If session or call tree doesn't exist
        """
        pass

    @abstractmethod
    def list_sessions(self, source_dir: Optional[Path] = None) -> List[SessionInfo]:
        """
        List all sessions, optionally filtered by source directory.

        Args:
            source_dir: If provided, only return sessions from this source

        Returns:
            List[SessionInfo]: List of session information objects
        """
        pass

    @abstractmethod
    def update_session_status(self, execution_id: str, status: str,
                             completed_at: Optional[str] = None):
        """
        Update session status (running, completed, failed).

        Args:
            execution_id: Session identifier
            status: New status
            completed_at: ISO timestamp when completed (optional)

        Raises:
            ValueError: If session doesn't exist
        """
        pass

    @abstractmethod
    def cleanup_session(self, execution_id: str):
        """
        Delete a session and all its data.

        Args:
            execution_id: Session identifier

        Raises:
            ValueError: If session doesn't exist
        """
        pass

    @abstractmethod
    def get_sessions_dir(self) -> Path:
        """
        Get the root directory where sessions are stored.

        Returns:
            Path: Sessions directory path
        """
        pass

    # Helper methods (implemented by base class, can be overridden)

    def node_exists(self, execution_id: str, node_id: str) -> bool:
        """
        Check if a node exists in a session.

        Args:
            execution_id: Session identifier
            node_id: Node identifier

        Returns:
            bool: True if node exists
        """
        try:
            # Try to get node metadata
            session_ctx = self.get_session_context(execution_id)
            node_dir = session_ctx.artifacts_dir / "nodes" / node_id
            return node_dir.exists()
        except (ValueError, FileNotFoundError):
            return False

    def get_workspace_path(self, execution_id: str, relative_path: str) -> Path:
        """
        Get workspace path for a file.

        Args:
            execution_id: Session identifier
            relative_path: Path relative to source_dir

        Returns:
            Path: Absolute path in workspace
        """
        session_ctx = self.get_session_context(execution_id)
        return session_ctx.get_workspace_path(relative_path)
