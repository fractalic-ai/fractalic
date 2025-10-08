"""
Tests for LocalFileStorage implementation.
"""

import pytest
import tempfile
import shutil
import uuid
from pathlib import Path

from core.storage import LocalFileStorage, SessionContext, SessionMetadata, NodeMetadata


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_source_dir():
    """Create a temporary source directory with sample files."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create sample structure
    (temp_dir / "workflow.md").write_text("# Main workflow\n@llm\nprompt: test")
    (temp_dir / "agents").mkdir()
    (temp_dir / "agents" / "helper.md").write_text("# Helper agent\n@llm\nprompt: help")
    (temp_dir / "tools").mkdir()
    (temp_dir / "tools" / "tool.py").write_text("def tool(): pass")
    (temp_dir / "agents" / "tools").mkdir()
    (temp_dir / "agents" / "tools" / "local_tool.py").write_text("def local_tool(): pass")

    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_create_session(temp_base_dir, temp_source_dir):
    """Test creating a new session."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())
    initial_file = "workflow.md"
    settings = {"model": "gpt-4"}

    # Create session
    session_ctx = storage.create_session(
        execution_id=execution_id,
        source_dir=temp_source_dir,
        initial_file=initial_file,
        settings=settings
    )

    # Verify session context
    assert session_ctx.execution_id == execution_id
    assert session_ctx.workspace_dir.exists()
    assert session_ctx.artifacts_dir.exists()
    assert session_ctx.source_dir.resolve() == temp_source_dir.resolve()

    # Verify workspace structure
    assert (session_ctx.workspace_dir / "workflow.md").exists()
    assert (session_ctx.workspace_dir / "agents" / "helper.md").exists()
    assert (session_ctx.workspace_dir / "tools" / "tool.py").exists()
    assert (session_ctx.workspace_dir / "agents" / "tools" / "local_tool.py").exists()

    # Verify metadata
    assert session_ctx.metadata.execution_id == execution_id
    assert session_ctx.metadata.initial_file == initial_file
    assert session_ctx.metadata.settings == settings
    assert session_ctx.metadata.status == "running"


def test_create_duplicate_session(temp_base_dir, temp_source_dir):
    """Test that creating duplicate session raises error."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())

    # Create first session
    storage.create_session(execution_id, temp_source_dir)

    # Try to create duplicate
    with pytest.raises(ValueError, match="Session already exists"):
        storage.create_session(execution_id, temp_source_dir)


def test_get_session_context(temp_base_dir, temp_source_dir):
    """Test retrieving session context."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())

    # Create session
    original_ctx = storage.create_session(execution_id, temp_source_dir)

    # Get session context
    retrieved_ctx = storage.get_session_context(execution_id)

    assert retrieved_ctx.execution_id == original_ctx.execution_id
    assert retrieved_ctx.workspace_dir == original_ctx.workspace_dir
    assert retrieved_ctx.artifacts_dir == original_ctx.artifacts_dir


def test_save_and_get_node_artifact(temp_base_dir, temp_source_dir):
    """Test saving and retrieving node artifacts."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())
    storage.create_session(execution_id, temp_source_dir)

    node_id = str(uuid.uuid4())[:12]
    artifact_content = "# Result context\nThis is the processed output."

    metadata = NodeMetadata(
        node_id=node_id,
        execution_id=execution_id,
        original_path="workflow.md",
        workspace_path="/workspace/workflow.md",
        workspace_cwd="/workspace/",
        tools_resolved=["/workspace/tools/"]
    )

    # Save artifact
    artifact_path = storage.save_node_artifact(
        execution_id=execution_id,
        node_id=node_id,
        artifact_type='ctx',
        content=artifact_content,
        metadata=metadata
    )

    assert artifact_path is not None

    # Retrieve artifact
    retrieved_content = storage.get_node_artifact(execution_id, node_id, 'ctx')
    assert retrieved_content == artifact_content


def test_multiple_runs_same_file(temp_base_dir, temp_source_dir):
    """Test that multiple runs of same file create separate node artifacts."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())
    storage.create_session(execution_id, temp_source_dir)

    # First run
    node_id_1 = str(uuid.uuid4())[:12]
    content_1 = "First run result"
    storage.save_node_artifact(execution_id, node_id_1, 'ctx', content_1)

    # Second run (same file, different node_id)
    node_id_2 = str(uuid.uuid4())[:12]
    content_2 = "Second run result"
    storage.save_node_artifact(execution_id, node_id_2, 'ctx', content_2)

    # Verify both artifacts exist independently
    retrieved_1 = storage.get_node_artifact(execution_id, node_id_1, 'ctx')
    retrieved_2 = storage.get_node_artifact(execution_id, node_id_2, 'ctx')

    assert retrieved_1 == content_1
    assert retrieved_2 == content_2
    assert retrieved_1 != retrieved_2


def test_save_and_get_call_tree(temp_base_dir, temp_source_dir):
    """Test saving and retrieving call tree."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())
    storage.create_session(execution_id, temp_source_dir)

    call_tree = {
        "node_id": "root123",
        "filename": "workflow.md",
        "children": [
            {
                "node_id": "child456",
                "filename": "agents/helper.md",
                "children": []
            }
        ]
    }

    # Save call tree
    storage.save_call_tree(execution_id, call_tree)

    # Retrieve call tree
    retrieved_tree = storage.get_call_tree(execution_id)

    assert retrieved_tree == call_tree


def test_list_sessions(temp_base_dir, temp_source_dir):
    """Test listing sessions."""
    storage = LocalFileStorage(temp_base_dir)

    # Create multiple sessions
    exec_id_1 = str(uuid.uuid4())
    exec_id_2 = str(uuid.uuid4())

    storage.create_session(exec_id_1, temp_source_dir, initial_file="workflow1.md")
    storage.create_session(exec_id_2, temp_source_dir, initial_file="workflow2.md")

    # List all sessions
    sessions = storage.list_sessions()

    assert len(sessions) == 2
    assert any(s.execution_id == exec_id_1 for s in sessions)
    assert any(s.execution_id == exec_id_2 for s in sessions)


def test_update_session_status(temp_base_dir, temp_source_dir):
    """Test updating session status."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())
    storage.create_session(execution_id, temp_source_dir)

    # Update to completed
    storage.update_session_status(execution_id, "completed")

    # Retrieve and verify
    session_ctx = storage.get_session_context(execution_id)
    assert session_ctx.metadata.status == "completed"
    assert session_ctx.metadata.completed_at is not None


def test_cleanup_session(temp_base_dir, temp_source_dir):
    """Test session cleanup."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())
    session_ctx = storage.create_session(execution_id, temp_source_dir)

    # Verify session exists
    assert session_ctx.workspace_dir.exists()

    # Cleanup
    storage.cleanup_session(execution_id)

    # Verify session is deleted
    session_dir = temp_base_dir / "sessions" / execution_id
    assert not session_dir.exists()

    # Try to get cleaned up session should raise error
    with pytest.raises(ValueError, match="Session does not exist"):
        storage.get_session_context(execution_id)


def test_workspace_isolation(temp_base_dir, temp_source_dir):
    """Test that workspace changes don't affect source directory."""
    storage = LocalFileStorage(temp_base_dir)

    execution_id = str(uuid.uuid4())
    session_ctx = storage.create_session(execution_id, temp_source_dir)

    # Modify workspace file
    workspace_file = session_ctx.workspace_dir / "workflow.md"
    workspace_file.write_text("# Modified in workspace")

    # Verify source is unchanged
    source_file = temp_source_dir / "workflow.md"
    assert "# Main workflow" in source_file.read_text()
    assert "Modified in workspace" not in source_file.read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
