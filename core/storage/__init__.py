"""
Storage layer for Fractalic sessions.

This module provides an abstraction layer for storing and retrieving Fractalic
session data, artifacts, and metadata. It replaces the previous Git-based storage
with a more flexible system that supports:

- Isolated session workspaces
- Node-based artifact storage (solving multiple runs of same file)
- Parallel execution without conflicts
- Multiple storage backends (local files, S3, database)

Public API:
    get_session_storage() - Get the configured storage instance
    SessionStorage - Abstract base class for storage implementations
    SessionContext - Context object for a session
    SessionMetadata - Metadata for a session
    LocalFileStorage - File-based storage implementation
"""

from .base import SessionStorage
from .session_context import SessionContext, SessionMetadata, SessionInfo, NodeMetadata
from .local_storage import LocalFileStorage

# Global storage instance
_storage_instance = None


def get_session_storage() -> SessionStorage:
    """
    Get the global storage instance.

    Returns:
        SessionStorage: The configured storage instance
    """
    global _storage_instance

    if _storage_instance is None:
        # Default to LocalFileStorage
        _storage_instance = LocalFileStorage()

    return _storage_instance


def set_session_storage(storage: SessionStorage):
    """
    Set the global storage instance.

    Args:
        storage: The storage instance to use
    """
    global _storage_instance
    _storage_instance = storage


def get_sessions_dir(session_root=None):
    """
    Get the sessions directory path.

    Returns proper path: {session_root}/.fractalic/sessions/

    Args:
        session_root: Optional session root path. If not provided, uses get_session_root()

    Returns:
        Path: Sessions directory
    """
    from pathlib import Path
    from core.paths import get_session_root as _get_session_root

    if session_root is None:
        session_root = _get_session_root()

    return Path(session_root) / '.fractalic' / 'sessions'


__all__ = [
    'SessionStorage',
    'SessionContext',
    'SessionMetadata',
    'SessionInfo',
    'NodeMetadata',
    'LocalFileStorage',
    'get_session_storage',
    'set_session_storage',
    'get_sessions_dir',
]
