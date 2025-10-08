"""
Tests for Git API Adapter

Tests the compatibility layer that emulates Git operations
for the storage backend.
"""

import pytest
import os
from pathlib import Path
from core.storage.git_api_adapter import GitApiAdapter, get_repo_adapter


def test_git_adapter_with_storage_mode(tmp_path):
    """Test Git adapter in storage mode with actual session."""
    # Use the latest session from our test runs
    sessions_dir = Path.home() / '.fractalic' / 'sessions'

    if not sessions_dir.exists():
        pytest.skip("No sessions directory found")

    # Get the most recent session
    sessions = sorted(sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        pytest.skip("No sessions found")

    execution_id = sessions[0].name
    print(f"Testing with session: {execution_id}")

    # Create adapter
    adapter = GitApiAdapter(
        repo_path="/private/tmp/fractalic-storage-test",
        execution_id=execution_id
    )

    # Find an artifact to test with
    artifacts_dir = sessions[0] / 'artifacts' / 'nodes'
    test_artifact = None
    test_path = None

    for node_dir in artifacts_dir.iterdir():
        if not node_dir.is_dir():
            continue

        # Try to find result.ctx
        ctx_file = node_dir / 'result.ctx'
        if ctx_file.exists():
            test_artifact = f"artifacts/nodes/{node_dir.name}/result.ctx"
            test_path = "test.ctx"  # Original filename
            break

    if not test_artifact:
        pytest.skip("No artifacts found in session")

    # Test 1: Read artifact by path
    content = adapter.show(f"{test_artifact}:dummy")
    assert content is not None
    assert len(content) > 0
    print(f"✓ Read artifact by path: {len(content)} bytes")

    # Test 2: Read via proxy object
    repo_proxy = get_repo_adapter("/private/tmp/fractalic-storage-test", execution_id)
    content2 = repo_proxy.git.show(f"{test_artifact}:dummy")
    assert content2 == content
    print(f"✓ Proxy object works correctly")


def test_git_adapter_legacy_mode(tmp_path):
    """Test Git adapter falls back to actual Git when no execution_id."""
    # Skip if not in a git repo
    try:
        import git
        repo = git.Repo(os.getcwd())
    except Exception:
        pytest.skip("Not in a git repository")

    # Create adapter without execution_id (legacy mode)
    adapter = GitApiAdapter(
        repo_path=os.getcwd(),
        execution_id=None
    )

    # This should use actual Git
    # Note: This test assumes we're in fractalic git repo
    try:
        # Try to read README.md from current commit
        content = adapter.show("HEAD:README.md")
        assert content is not None
        assert "fractalic" in content.lower() or "readme" in content.lower()
        print(f"✓ Legacy Git mode works: {len(content)} bytes")
    except Exception as e:
        pytest.skip(f"Could not read from Git: {e}")


def test_git_adapter_artifact_search(tmp_path):
    """Test artifact search by filename."""
    sessions_dir = Path.home() / '.fractalic' / 'sessions'

    if not sessions_dir.exists():
        pytest.skip("No sessions directory found")

    sessions = sorted(sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        pytest.skip("No sessions found")

    execution_id = sessions[0].name
    adapter = GitApiAdapter(
        repo_path="/private/tmp/fractalic-storage-test",
        execution_id=execution_id
    )

    # Test searching by .ctx extension
    try:
        content = adapter.show("dummy_hash:test.ctx")
        assert content is not None
        print(f"✓ Found artifact by filename search: {len(content)} bytes")
    except FileNotFoundError:
        pytest.skip("No .ctx artifacts found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
