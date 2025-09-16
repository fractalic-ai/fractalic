#!/usr/bin/env python3
"""
Test script to demonstrate the new path resolution architecture and safety checks.
"""

from core.paths import (
    get_fractalic_root, 
    set_session_root, 
    set_session_cwd,
    get_settings_path,
    get_tools_directory,
    get_mcp_servers_config_path,
    get_context_summary,
    reset_session_context,
    validate_session_safety,
    get_git_repository_root,
    ensure_git_in_session_root
)

def test_path_resolution():
    """Test the new path resolution system."""
    
    print("=== Fractalic Path Resolution Test ===\n")
    
    # Show initial state (no session)
    print("1. Initial state (no session):")
    print(f"   Fractalic root: {get_fractalic_root()}")
    print(f"   Settings path: {get_settings_path()}")
    print(f"   Tools directory: {get_tools_directory()}")
    print(f"   MCP config: {get_mcp_servers_config_path()}")
    
    # Test safety validation without session
    print("\n2. Testing safety validation (no session):")
    try:
        validate_session_safety()
        print("   ✗ Unexpected: validation passed without session")
    except RuntimeError as e:
        print(f"   ✓ Expected error: {e}")
    
    # Test unsafe session (fractalic_root == session_root)
    print("\n3. Testing unsafe session (fractalic_root == session_root):")
    try:
        set_session_root(get_fractalic_root())
        print("   ✗ Unexpected: unsafe session was allowed")
    except RuntimeError as e:
        print(f"   ✓ Expected safety error: {e}")
    
    # Reset and test safe session
    reset_session_context()
    print("\n4. Setting safe session context:")
    session_root = get_fractalic_root() / "tutorials" / "hello-world"
    session_cwd = session_root / "agents"
    
    print(f"   Session root: {session_root}")
    print(f"   Session CWD: {session_cwd}")
    
    try:
        set_session_root(session_root)
        set_session_cwd(session_cwd)
        print("   ✓ Safe session established")
    except RuntimeError as e:
        print(f"   ✗ Unexpected error: {e}")
        return
    
    print("\n5. After setting safe session context:")
    print(f"   Settings path: {get_settings_path()}")
    print(f"   Tools directory: {get_tools_directory()}")
    print(f"   MCP config: {get_mcp_servers_config_path()}")
    print(f"   Git repository root: {get_git_repository_root()}")
    
    print("\n6. Git directory management:")
    git_dir = ensure_git_in_session_root()
    print(f"   Git directory location: {git_dir}")
    print(f"   Git is in session_root: {git_dir.parent == session_root}")
    
    print("\n7. Full context summary:")
    context = get_context_summary()
    for key, value in context.items():
        print(f"   {key}: {value}")
    
    # Reset for clean state
    reset_session_context()
    print("\n8. After reset:")
    print(f"   Settings path: {get_settings_path()}")
    print(f"   Tools directory: {get_tools_directory()}")

if __name__ == "__main__":
    test_path_resolution()
