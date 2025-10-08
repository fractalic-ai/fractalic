"""
Tests for extended CallTreeNode with storage support.
"""

import pytest
import os
import json
from core.operations.call_tree import CallTreeNode


def test_legacy_constructor():
    """Test that legacy constructor still works (backward compatibility)."""
    node = CallTreeNode(
        operation='@run',
        operation_src=None,
        filename='workflow.md',
        md_commit_hash='abc123',
        ctx_commit_hash='def456',
        ctx_file='workflow.ctx',
        trc_file='workflow.trc'
    )

    # Legacy fields should work
    assert node.operation == '@run'
    assert node.filename == 'workflow.md'
    assert node.md_commit_hash == 'abc123'
    assert node.ctx_commit_hash == 'def456'
    assert node.ctx_file == 'workflow.ctx'
    assert node.trc_file == 'workflow.trc'

    # New fields should be auto-generated
    assert node.node_id is not None
    assert len(node.node_id) == 12  # UUID truncated to 12 chars
    assert node.artifacts_dir == f"artifacts/nodes/{node.node_id}"


def test_extended_constructor():
    """Test new constructor with extended parameters."""
    node = CallTreeNode(
        operation='@run',
        operation_src=None,
        filename='agents/helper.md',
        node_id='abc123def456',
        execution_id='exec-uuid-here',
        original_file_path='agents/helper.md',
        workspace_file_path='/workspace/agents/helper.md',
        workspace_cwd='/workspace/agents/',
        tools_resolved=['/workspace/agents/tools/', '/workspace/tools/']
    )

    # New fields should be set
    assert node.node_id == 'abc123def456'
    assert node.execution_id == 'exec-uuid-here'
    assert node.original_file_path == 'agents/helper.md'
    assert node.workspace_file_path == '/workspace/agents/helper.md'
    assert node.workspace_cwd == '/workspace/agents/'
    assert node.tools_resolved == ['/workspace/agents/tools/', '/workspace/tools/']
    assert node.artifacts_dir == 'artifacts/nodes/abc123def456'


def test_node_id_autogeneration():
    """Test that node_id is auto-generated if not provided."""
    node1 = CallTreeNode(operation='@run', operation_src=None, filename='test1.md')
    node2 = CallTreeNode(operation='@run', operation_src=None, filename='test2.md')

    # Each node should have unique ID
    assert node1.node_id != node2.node_id
    assert len(node1.node_id) == 12
    assert len(node2.node_id) == 12


def test_execution_id_from_env():
    """Test that execution_id is read from environment if not provided."""
    # Set environment variable
    test_exec_id = 'test-exec-id-123'
    os.environ['FRACTALIC_EXECUTION_ID'] = test_exec_id

    try:
        node = CallTreeNode(operation='@run', operation_src=None, filename='test.md')
        assert node.execution_id == test_exec_id
    finally:
        # Cleanup
        if 'FRACTALIC_EXECUTION_ID' in os.environ:
            del os.environ['FRACTALIC_EXECUTION_ID']


def test_get_artifact_path():
    """Test get_artifact_path helper method."""
    node = CallTreeNode(
        operation='@run',
        operation_src=None,
        filename='test.md',
        node_id='abc123'
    )

    assert node.get_artifact_path('ctx') == 'artifacts/nodes/abc123/result.ctx'
    assert node.get_artifact_path('trc') == 'artifacts/nodes/abc123/result.trc'
    assert node.get_artifact_path('source_snapshot') == 'artifacts/nodes/abc123/source_snapshot.md'
    assert node.get_artifact_path('metadata') == 'artifacts/nodes/abc123/metadata.json'


def test_compute_content_hash():
    """Test content hash computation for Git API compatibility."""
    node = CallTreeNode(
        operation='@run',
        operation_src=None,
        filename='test.md',
        node_id='abc123'
    )

    content = "# Test content\nSome data here"
    hash1 = node.compute_content_hash(content)

    # Hash should be 40 characters (Git-compatible SHA256)
    assert len(hash1) == 40

    # Same content should produce same hash
    hash2 = node.compute_content_hash(content)
    assert hash1 == hash2

    # Different content should produce different hash
    hash3 = node.compute_content_hash("Different content")
    assert hash1 != hash3


def test_set_artifact_paths_from_storage():
    """Test updating legacy paths from storage."""
    node = CallTreeNode(operation='@run', operation_src=None, filename='test.md')

    ctx_path = 'artifacts/nodes/abc123/result.ctx'
    trc_path = 'artifacts/nodes/abc123/result.trc'

    node.set_artifact_paths_from_storage(ctx_path, trc_path)

    assert node.ctx_file == ctx_path
    assert node.trc_file == trc_path


def test_add_child():
    """Test adding children to the tree."""
    parent = CallTreeNode(operation='@run', operation_src=None, filename='parent.md')
    child1 = CallTreeNode(operation='@run', operation_src='@run', filename='child1.md', parent=parent)
    child2 = CallTreeNode(operation='@run', operation_src='@run', filename='child2.md', parent=parent)

    parent.add_child(child1)
    parent.add_child(child2)

    assert len(parent.children) == 2
    assert parent.children[0] == child1
    assert parent.children[1] == child2
    assert child1.parent == parent
    assert child2.parent == parent


def test_to_dict_serialization():
    """Test dictionary serialization."""
    node = CallTreeNode(
        operation='@run',
        operation_src=None,
        filename='test.md',
        node_id='abc123',
        execution_id='exec-123',
        original_file_path='agents/test.md',
        workspace_file_path='/workspace/agents/test.md',
        workspace_cwd='/workspace/agents/',
        md_commit_hash='md-hash',
        ctx_commit_hash='ctx-hash',
        ctx_file='test.ctx',
        trc_file='test.trc'
    )

    data = node.to_dict()

    # Legacy fields
    assert data['operation'] == '@run'
    assert data['filename'] == 'test.md'
    assert data['md_commit_hash'] == 'md-hash'
    assert data['ctx_commit_hash'] == 'ctx-hash'
    assert data['ctx_file'] == 'test.ctx'
    assert data['trc_file'] == 'test.trc'

    # New fields
    assert data['node_id'] == 'abc123'
    assert data['execution_id'] == 'exec-123'
    assert data['original_file_path'] == 'agents/test.md'
    assert data['workspace_file_path'] == '/workspace/agents/test.md'
    assert data['workspace_cwd'] == '/workspace/agents/'
    assert data['artifacts_dir'] == 'artifacts/nodes/abc123'


def test_to_json_serialization():
    """Test JSON serialization."""
    node = CallTreeNode(
        operation='@run',
        operation_src=None,
        filename='test.md',
        node_id='abc123'
    )

    json_str = node.to_json()

    # Should be valid JSON
    data = json.loads(json_str)
    assert data['node_id'] == 'abc123'
    assert data['operation'] == '@run'


def test_from_dict_deserialization():
    """Test creating node from dictionary."""
    data = {
        'operation': '@run',
        'operation_src': None,
        'filename': 'test.md',
        'node_id': 'abc123',
        'execution_id': 'exec-123',
        'original_file_path': 'agents/test.md',
        'workspace_file_path': '/workspace/agents/test.md',
        'workspace_cwd': '/workspace/agents/',
        'md_commit_hash': 'md-hash',
        'ctx_commit_hash': 'ctx-hash',
        'ctx_file': 'test.ctx',
        'trc_file': 'test.trc',
        'artifacts_dir': 'artifacts/nodes/abc123',
        'tools_resolved': ['/workspace/tools/'],
        'timestamp': '2025-10-08T00:00:00',
        'children': []
    }

    node = CallTreeNode.from_dict(data)

    # Verify all fields
    assert node.operation == '@run'
    assert node.filename == 'test.md'
    assert node.node_id == 'abc123'
    assert node.execution_id == 'exec-123'
    assert node.original_file_path == 'agents/test.md'
    assert node.workspace_file_path == '/workspace/agents/test.md'
    assert node.timestamp == '2025-10-08T00:00:00'


def test_from_dict_with_children():
    """Test deserialization with nested children."""
    data = {
        'operation': '@run',
        'operation_src': None,
        'filename': 'parent.md',
        'node_id': 'parent123',
        'children': [
            {
                'operation': '@run',
                'operation_src': '@run',
                'filename': 'child1.md',
                'node_id': 'child456',
                'children': []
            },
            {
                'operation': '@run',
                'operation_src': '@run',
                'filename': 'child2.md',
                'node_id': 'child789',
                'children': []
            }
        ]
    }

    parent = CallTreeNode.from_dict(data)

    assert len(parent.children) == 2
    assert parent.children[0].node_id == 'child456'
    assert parent.children[1].node_id == 'child789'
    assert parent.children[0].parent == parent
    assert parent.children[1].parent == parent


def test_legacy_data_compatibility():
    """Test that old format data can be loaded (no node_id, execution_id)."""
    # Old format (legacy)
    legacy_data = {
        'operation': '@run',
        'operation_src': None,
        'filename': 'old_workflow.md',
        'md_commit_hash': 'abc123',
        'ctx_commit_hash': 'def456',
        'ctx_file': 'old_workflow.ctx',
        'trc_file': 'old_workflow.trc',
        'children': []
    }

    node = CallTreeNode.from_dict(legacy_data)

    # Legacy fields should work
    assert node.filename == 'old_workflow.md'
    assert node.md_commit_hash == 'abc123'

    # New fields should be auto-generated or None
    assert node.node_id is not None  # Auto-generated
    assert node.execution_id is None or node.execution_id  # May be from env or None


def test_roundtrip_serialization():
    """Test that node can be serialized and deserialized without loss."""
    original = CallTreeNode(
        operation='@run',
        operation_src='fractalic_run',
        filename='agents/helper.md',
        node_id='abc123',
        execution_id='exec-456',
        original_file_path='agents/helper.md',
        workspace_file_path='/workspace/agents/helper.md',
        md_commit_hash='md-hash',
        ctx_commit_hash='ctx-hash'
    )

    # Add a child
    child = CallTreeNode(
        operation='@llm',
        operation_src='llm_call',
        filename='agents/helper.md',
        node_id='child789',
        parent=original
    )
    original.add_child(child)

    # Serialize
    data = original.to_dict()

    # Deserialize
    restored = CallTreeNode.from_dict(data)

    # Compare
    assert restored.node_id == original.node_id
    assert restored.execution_id == original.execution_id
    assert restored.original_file_path == original.original_file_path
    assert restored.workspace_file_path == original.workspace_file_path
    assert len(restored.children) == 1
    assert restored.children[0].node_id == 'child789'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
