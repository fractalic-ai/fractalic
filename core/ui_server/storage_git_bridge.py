"""
Storage-Git Bridge

Transparent Git API emulation for storage sessions.
Frontend thinks it's talking to Git, server automatically routes to storage.
"""

from pathlib import Path
import json
from typing import Optional, Dict, Any, List


def is_storage_session(branch_or_execution_id: str) -> bool:
    """
    Check if a "branch name" is actually a storage session.

    Args:
        branch_or_execution_id: Could be Git branch name or execution_id

    Returns:
        True if it's a storage session, False if it's a Git branch
    """
    from core.storage import get_sessions_dir
    sessions_dir = get_sessions_dir()
    session_path = sessions_dir / branch_or_execution_id
    return session_path.exists() and session_path.is_dir()


def list_storage_sessions_as_branches(repo_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List storage sessions formatted as Git branches.

    Frontend sees execution_id as "branch name".

    Args:
        repo_path: Optional filter - only show sessions from this source directory
    """
    from pathlib import Path
    from core.storage import get_sessions_dir

    # Get sessions directory - if repo_path provided, look there, otherwise use current session_root
    if repo_path:
        sessions_dir = Path(repo_path) / '.fractalic' / 'sessions'
    else:
        sessions_dir = get_sessions_dir()

    if not sessions_dir.exists():
        return []

    # Normalize repo_path for comparison (to filter by source_dir)
    filter_path = Path(repo_path).resolve() if repo_path else None

    branches_data = []

    for session_dir in sorted(sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not session_dir.is_dir():
            continue

        execution_id = session_dir.name

        # Read session metadata
        metadata_file = session_dir / 'metadata' / 'session.json'
        session_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    session_metadata = json.load(f)
            except:
                pass

        # Filter by source_root if repo_path provided
        if filter_path:
            source_root = session_metadata.get('source_root')
            if source_root:
                try:
                    session_source = Path(source_root).resolve()
                    if session_source != filter_path:
                        continue  # Skip sessions from different directories
                except:
                    continue  # Skip if path resolution fails

        # Read call_tree from metadata
        call_tree_file = session_dir / 'metadata' / 'call_tree.json'
        call_tree = None

        if call_tree_file.exists():
            try:
                with open(call_tree_file, 'r') as f:
                    call_tree = json.load(f)
            except:
                pass

        if not call_tree:
            continue

        # Build session node (frontend thinks it's a branch)
        created_at = session_metadata.get('created_at', 'unknown')
        source_root_full = session_metadata.get('source_root', '')

        # Extract just the folder name (not full path) for display
        project_folder_name = Path(source_root_full).name if source_root_full else 'unknown'

        # Session displays like git branch: execution_id (timestamp)
        branch_node = {
            'id': execution_id,  # Frontend uses this as "branch name"
            'text': f"{execution_id} ({created_at})",
            'state': {'opened': True},
            'children': []
        }

        # Build tree from call_tree
        def build_tree(node):
            node_id = f"{node.get('ctx_file', 'unknown')}_{node.get('ctx_commit_hash', 'unknown')}"

            # File displays with project folder prefix for context
            original_path = node.get('original_file_path') or node.get('filename', 'unknown')
            display_text = f"{project_folder_name}/{original_path}"

            tree_node = {
                'id': node_id,
                'text': display_text,  # Display: "project-folder/path/to/file.md"
                'ctx_file': node.get('ctx_file'),
                'filename': node.get('filename'),
                'md_file': node.get('filename'),
                'md_commit_hash': node.get('md_commit_hash'),
                'ctx_commit_hash': node.get('ctx_commit_hash'),
                'trc_file': node.get('trc_file', ''),
                'trc_commit_hash': node.get('trc_commit_hash', ''),
                'branch': execution_id,  # This is the "fake branch name"
                'children': []
            }
            for child in node.get('children', []):
                child_node = build_tree(child)
                tree_node['children'].append(child_node)
            return tree_node

        root_node = build_tree(call_tree)
        branch_node['children'].append(root_node)
        branches_data.append(branch_node)

    return branches_data


def load_call_tree_from_storage(execution_id: str) -> Optional[Dict[str, Any]]:
    """
    Load call tree from storage session.

    Args:
        execution_id: Storage session ID

    Returns:
        Call tree dict or None if not found
    """
    from core.storage import get_sessions_dir
    sessions_dir = get_sessions_dir()
    session_dir = sessions_dir / execution_id
    artifacts_dir = session_dir / 'artifacts' / 'nodes'

    if not artifacts_dir.exists():
        return None

    # Find call_tree.dat in any node directory
    for node_dir in artifacts_dir.iterdir():
        if not node_dir.is_dir():
            continue

        call_tree_file = node_dir / 'call_tree.dat'
        if call_tree_file.exists():
            try:
                with open(call_tree_file, 'r') as f:
                    return json.load(f)
            except:
                pass

    return None


def get_artifact_from_storage(execution_id: str, commit_hash: str, file_path: str, repo_path: str) -> Optional[str]:
    """
    Get artifact content from storage directly.

    Args:
        execution_id: Storage session ID
        commit_hash: Content hash or artifact path (can be artifact path like "artifacts/nodes/xxx/result.ctx")
        file_path: Original file path or artifact path
        repo_path: Repository root path to locate sessions

    Returns:
        Artifact content or None
    """
    from pathlib import Path

    try:
        # Build path to session
        sessions_dir = Path(repo_path) / '.fractalic' / 'sessions'
        session_dir = sessions_dir / execution_id

        if not session_dir.exists():
            return None

        # If file_path starts with 'artifacts/', it's an artifact - read from session root
        if file_path.startswith('artifacts/'):
            artifact_path = session_dir / file_path
            if artifact_path.exists():
                with open(artifact_path, 'r', encoding='utf-8') as f:
                    return f.read()
        # Otherwise it's a source file - read from workspace snapshot
        else:
            workspace_path = session_dir / 'workspace' / file_path
            if workspace_path.exists():
                with open(workspace_path, 'r', encoding='utf-8') as f:
                    return f.read()

        # Fallback: try commit_hash as artifact path
        if commit_hash and commit_hash.startswith('artifacts/'):
            artifact_path = session_dir / commit_hash
            if artifact_path.exists():
                with open(artifact_path, 'r', encoding='utf-8') as f:
                    return f.read()

        return None
    except Exception as e:
        print(f"[Storage Bridge] Error reading artifact: {e}")
        import traceback
        traceback.print_exc()
        return None
