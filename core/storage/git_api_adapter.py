"""
Git API Adapter for Storage Backend

This module provides a compatibility layer that emulates Git operations
for the storage backend, allowing React frontend to work seamlessly
with both legacy Git mode and new storage mode.

Key concepts:
- In storage mode, "commit_hash" is actually a content_hash or artifact path
- repo.git.show() is emulated by reading from storage artifacts
- Supports both modes: storage (execution_id set) and legacy Git
"""

import os
from pathlib import Path
from typing import Optional
from git import Repo, InvalidGitRepositoryError

from core.storage import get_session_storage
from core.storage.base import SessionStorage


class GitApiAdapter:
    """
    Adapter that emulates Git API for storage backend.

    Provides backward compatibility for React frontend which expects
    repo.git.show() interface but works with storage artifacts.
    """

    def __init__(self, repo_path: str, execution_id: Optional[str] = None):
        """
        Initialize adapter.

        Args:
            repo_path: Path to repository (used in legacy Git mode)
            execution_id: Execution ID for storage mode (if None, uses Git mode)
        """
        self.repo_path = Path(repo_path)
        self.execution_id = execution_id
        self.storage = get_session_storage() if execution_id else None
        self._git_repo = None

    @property
    def git_repo(self) -> Optional[Repo]:
        """Lazy-load Git repo (only in legacy mode)."""
        if self._git_repo is None and not self.execution_id:
            try:
                self._git_repo = Repo(self.repo_path)
            except InvalidGitRepositoryError:
                pass
        return self._git_repo

    def show(self, ref: str) -> str:
        """
        Emulate `repo.git.show(ref)` command.

        Args:
            ref: Git reference in format "commit_hash:filepath"
                 In storage mode: "content_hash:filepath" or "artifact_path"

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If artifact not found
            ValueError: If ref format invalid
        """
        if ':' not in ref:
            raise ValueError(f"Invalid ref format: {ref}. Expected 'hash:filepath'")

        hash_or_path, filepath = ref.split(':', 1)

        if self.execution_id:
            # Storage mode: Read from artifacts
            return self._read_from_storage(hash_or_path, filepath)
        else:
            # Legacy Git mode: Use actual Git
            return self._read_from_git(ref)

    def _read_from_storage(self, hash_or_path: str, filepath: str) -> str:
        """
        Read artifact from storage.

        Strategy:
        1. If hash_or_path looks like artifact path (artifacts/nodes/...), read directly
        2. If hash_or_path is content hash, use get_artifact_by_content_hash()
        3. Fallback to searching in all nodes

        Args:
            hash_or_path: Content hash or artifact path
            filepath: Original file path (for hash lookup)

        Returns:
            Artifact content
        """
        # Case 1: Direct artifact path (e.g., "artifacts/nodes/abc123/result.ctx")
        if hash_or_path.startswith('artifacts/'):
            return self._read_artifact_by_path(hash_or_path)

        # Case 2: Content hash lookup
        try:
            content = self.storage.get_artifact_by_content_hash(
                execution_id=self.execution_id,
                content_hash=hash_or_path,
                file_path=filepath
            )
            if content is not None:
                return content
        except Exception as e:
            print(f"[GitApiAdapter] Hash lookup failed: {e}")

        # Case 3: Fallback - search by filename in all nodes
        return self._search_artifact_by_filename(filepath)

    def _read_artifact_by_path(self, artifact_path: str) -> str:
        """Read artifact directly by path."""
        from core.storage import get_sessions_dir
        sessions_dir = get_sessions_dir()
        session_dir = sessions_dir / self.execution_id
        full_path = session_dir / artifact_path

        if not full_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _search_artifact_by_filename(self, filename: str) -> str:
        """
        Search for artifact by filename across all nodes.

        This is a fallback when content hash lookup fails.
        Returns the first matching artifact found.
        """
        from core.storage import get_sessions_dir
        sessions_dir = get_sessions_dir()
        session_dir = sessions_dir / self.execution_id
        artifacts_dir = session_dir / 'artifacts' / 'nodes'

        if not artifacts_dir.exists():
            raise FileNotFoundError(f"No artifacts found for session {self.execution_id}")

        # Determine artifact type from filename extension
        file_ext = Path(filename).suffix
        if file_ext == '.ctx':
            artifact_name = 'result.ctx'
        elif file_ext == '.trc':
            artifact_name = 'result.trc'
        elif file_ext == '.md':
            # Original markdown - check workspace
            workspace_file = session_dir / 'workspace' / filename
            if workspace_file.exists():
                with open(workspace_file, 'r', encoding='utf-8') as f:
                    return f.read()
            artifact_name = filename  # Try as artifact name
        else:
            artifact_name = filename

        # Search all node directories
        for node_dir in artifacts_dir.iterdir():
            if not node_dir.is_dir():
                continue

            artifact_file = node_dir / artifact_name
            if artifact_file.exists():
                with open(artifact_file, 'r', encoding='utf-8') as f:
                    return f.read()

        raise FileNotFoundError(
            f"Artifact '{filename}' not found in any node for session {self.execution_id}"
        )

    def _read_from_git(self, ref: str) -> str:
        """Read content using actual Git (legacy mode)."""
        if not self.git_repo:
            raise FileNotFoundError(f"No Git repository at {self.repo_path}")

        try:
            return self.git_repo.git.show(ref)
        except Exception as e:
            raise FileNotFoundError(f"Git show failed for '{ref}': {e}")


def get_repo_adapter(repo_path: str, execution_id: Optional[str] = None):
    """
    Factory function to create Git adapter.

    This replaces the original `get_repo()` function in server.py
    when working with storage mode.

    Args:
        repo_path: Path to repository
        execution_id: Optional execution ID for storage mode

    Returns:
        GitApiAdapter instance with .git.show() interface
    """
    adapter = GitApiAdapter(repo_path, execution_id)

    # Create a proxy object that mimics GitPython's Repo.git interface
    class GitProxy:
        def __init__(self, adapter):
            self.adapter = adapter

        def show(self, ref: str) -> str:
            return self.adapter.show(ref)

    class RepoProxy:
        def __init__(self, adapter):
            self.git = GitProxy(adapter)
            self.adapter = adapter

    return RepoProxy(adapter)
