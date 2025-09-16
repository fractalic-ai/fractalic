"""
Centralized path management for Fractalic.

This module provides robust path resolution that works with proper separation of concerns:

- fractalic_root: Where fractalic.py is located (for libraries and core functionality)
- session_root: Where the first .md file being executed is located  
- session_cwd: Where the current .md file being executed is located (can change during execution)

Path resolution rules:
- tools/: Check session_cwd first, fallback to session_root
- settings.toml: Check session_root first, fallback to fractalic_root
- Core files: Always relative to fractalic_root

ARCHITECTURE SUMMARY
===================

This module implements a three-tier path resolution system for Fractalic:

1. **fractalic_root** (/path/to/fractalic/)
   - Where fractalic.py and core libraries are installed
   - Used for: Core functionality, fallback configs, MCP manager, logs
   - Fixed at import time based on this file's location

2. **session_root** (/path/to/user/project/) 
   - Where the first .md file being executed is located
   - Used for: Primary settings.toml, MCP configs, git repository
   - Set when Fractalic execution begins

3. **session_cwd** (/path/to/user/project/subdir/)
   - Where the currently executing .md file is located
   - Used for: Tools discovery (first priority)
   - Changes as execution moves through different directories

SAFETY RULES
============

- NEVER allow session_root == fractalic_root (prevents running files from installation)
- Git repositories (.git) are ONLY created in session_root
- Settings resolution: session_root → fractalic_root
- Tools resolution: session_cwd → session_root
- MCP config resolution: session_root → fractalic_root

USAGE PATTERN
=============

```python
from core.paths import set_session_root, set_session_cwd, validate_session_safety

# When starting Fractalic execution
input_file = "/some/project/workflow.md"
session_root = Path(input_file).parent
set_session_root(session_root)  # Auto-validates safety
set_session_cwd(session_root)

# When changing to a different file
current_file = "/some/project/agents/helper.md" 
set_session_cwd(Path(current_file).parent)

# Before any git operations
validate_session_safety()
git_root = get_git_repository_root()  # Always session_root
```
"""

import os
from pathlib import Path
from typing import Optional

# Determine the fractalic root directory based on this file's location
# This file is in core/paths.py, so we need to go up two levels to reach fractalic_root
_FRACTALIC_ROOT = Path(__file__).resolve().parent.parent

# Global variables to track session context (set by the runner)
_SESSION_ROOT: Optional[Path] = None
_SESSION_CWD: Optional[Path] = None

def get_fractalic_root() -> Path:
    """Get the absolute path to the fractalic installation root.
    
    This is where fractalic.py, core libraries, and fallback configurations are located.
    
    Returns:
        Path: The absolute path to the fractalic root directory
    """
    return _FRACTALIC_ROOT

def set_session_root(session_root: Path) -> None:
    """Set the session root directory (where the first .md file is located).
    
    Args:
        session_root: Path to the directory containing the initial .md file
        
    Raises:
        RuntimeError: If session_root equals fractalic_root (safety violation)
    """
    global _SESSION_ROOT
    _SESSION_ROOT = Path(session_root).resolve()
    
    # Immediate safety check
    validate_session_safety()

def set_session_cwd(session_cwd: Path) -> None:
    """Set the current session working directory (where the current .md file is located).
    
    Args:
        session_cwd: Path to the directory containing the currently executing .md file
    """
    global _SESSION_CWD
    _SESSION_CWD = Path(session_cwd).resolve()

def get_session_root() -> Optional[Path]:
    """Get the session root directory.
    
    Returns:
        Path: The session root directory, or None if not set
    """
    return _SESSION_ROOT

def get_session_cwd() -> Optional[Path]:
    """Get the current session working directory.
    
    Returns:
        Path: The current session working directory, or None if not set
    """
    return _SESSION_CWD

def get_settings_path() -> Path:
    """Get the path to settings.toml file.
    
    Resolution order:
    1. session_root/settings.toml
    2. fractalic_root/settings.toml
    
    Returns:
        Path: The path to the settings.toml file to use
    """
    # Try session_root first
    if _SESSION_ROOT:
        session_settings = _SESSION_ROOT / "settings.toml"
        if session_settings.exists():
            return session_settings
    
    # Fallback to fractalic_root
    return _FRACTALIC_ROOT / "settings.toml"

def get_tools_directory() -> Path:
    """Get the path to the tools directory.
    
    Resolution order:
    1. session_cwd/tools/
    2. session_root/tools/
    
    Returns:
        Path: The path to the tools directory to use
    """
    # Try session_cwd first
    if _SESSION_CWD:
        session_cwd_tools = _SESSION_CWD / "tools"
        if session_cwd_tools.exists():
            return session_cwd_tools
    
    # Fallback to session_root
    if _SESSION_ROOT:
        session_root_tools = _SESSION_ROOT / "tools"
        if session_root_tools.exists():
            return session_root_tools
        # Return session_root/tools even if it doesn't exist (for potential creation)
        return session_root_tools
    
    # Final fallback: return session_cwd/tools or current directory/tools
    if _SESSION_CWD:
        return _SESSION_CWD / "tools"
    
    return Path.cwd() / "tools"

def get_oauth_cache_directory() -> Path:
    """Get the absolute path to the oauth-cache directory.
    
    This is always in fractalic_root as OAuth cache is shared across all sessions.
    
    Returns:
        Path: The absolute path to the oauth-cache directory
    """
    return _FRACTALIC_ROOT / "oauth-cache"

def get_logs_directory() -> Path:
    """Get the absolute path to the logs directory.
    
    This is always in fractalic_root as logs are shared across all sessions.
    
    Returns:
        Path: The absolute path to the logs directory
    """
    return _FRACTALIC_ROOT / "logs"

def get_mcp_servers_config_path() -> Path:
    """Get the path to the mcp_servers.json file.
    
    Resolution order:
    1. session_root/mcp_servers.json
    2. fractalic_root/mcp_servers.json
    
    Returns:
        Path: The path to the mcp_servers.json file to use
    """
    # Try session_root first
    if _SESSION_ROOT:
        session_mcp_config = _SESSION_ROOT / "mcp_servers.json"
        if session_mcp_config.exists():
            return session_mcp_config
    
    # Fallback to fractalic_root
    return _FRACTALIC_ROOT / "mcp_servers.json"

def resolve_file_path(file_path: str, base_dir: Optional[str] = None) -> Path:
    """Resolve a file path to an absolute path.
    
    Args:
        file_path: The file path to resolve (can be relative or absolute)
        base_dir: Optional base directory for relative paths (defaults to current working directory)
    
    Returns:
        Path: The resolved absolute path
    """
    path = Path(file_path)
    
    if path.is_absolute():
        return path.resolve()
    
    if base_dir:
        return (Path(base_dir) / path).resolve()
    
    return (Path.cwd() / path).resolve()

def get_relative_to_session_root(file_path: str) -> str:
    """Get a path relative to the session root.
    
    Args:
        file_path: The file path (can be relative or absolute)
    
    Returns:
        str: The path relative to the session root, or relative to fractalic_root if no session
    """
    abs_path = resolve_file_path(file_path)
    
    # Try relative to session_root first
    if _SESSION_ROOT:
        try:
            return str(abs_path.relative_to(_SESSION_ROOT))
        except ValueError:
            pass  # Path is outside session root
    
    # Fallback to relative to fractalic_root
    try:
        return str(abs_path.relative_to(_FRACTALIC_ROOT))
    except ValueError:
        # Path is outside both roots, return as-is
        return str(abs_path)

def ensure_directory(directory_path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: The directory path to ensure
    
    Returns:
        Path: The ensured directory path
    """
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path

def get_fractalic_script_path() -> Path:
    """Get the absolute path to the main fractalic.py script.
    
    Returns:
        Path: The absolute path to fractalic.py
    """
    return _FRACTALIC_ROOT / "fractalic.py"

def get_mcp_manager_script_path() -> Path:
    """Get the absolute path to the fractalic_mcp_manager.py script.
    
    Returns:
        Path: The absolute path to fractalic_mcp_manager.py
    """
    return _FRACTALIC_ROOT / "fractalic_mcp_manager.py"

def validate_session_safety():
    """Validate that the session is safe to run.
    
    Raises:
        RuntimeError: If session_root equals fractalic_root (unsafe to run files from fractalic root)
        RuntimeError: If session context is not properly set
    """
    if not _SESSION_ROOT:
        raise RuntimeError(
            "Session root not set. Call set_session_root() before running Fractalic operations."
        )
    
    # Critical safety check: never allow running files from fractalic_root
    if _SESSION_ROOT.resolve() == _FRACTALIC_ROOT.resolve():
        raise RuntimeError(
            f"SAFETY ERROR: Cannot run Fractalic files from the fractalic installation directory.\n"
            f"Fractalic root: {_FRACTALIC_ROOT}\n"
            f"Session root: {_SESSION_ROOT}\n\n"
            f"Please run your .md files from a separate project directory, not from the Fractalic installation."
        )

def get_git_repository_root() -> Path:
    """Get the root directory where .git should be created/located.
    
    Git repositories are always created in session_root, never in fractalic_root.
    
    Returns:
        Path: The session_root directory where .git should be located
        
    Raises:
        RuntimeError: If session_root is not set
    """
    if not _SESSION_ROOT:
        raise RuntimeError(
            "Cannot determine git repository root: session_root not set. "
            "Call set_session_root() first."
        )
    
    return _SESSION_ROOT

def ensure_git_in_session_root():
    """Ensure .git directory exists in session_root only.
    
    This function enforces the rule that git repositories are created
    only in session_root, never in fractalic_root.
    
    Returns:
        Path: Path to the .git directory in session_root
        
    Raises:
        RuntimeError: If session safety validation fails
    """
    validate_session_safety()
    
    git_dir = _SESSION_ROOT / ".git"
    return git_dir

def reset_session_context():
    """Reset session context (useful for testing or new sessions)."""
    global _SESSION_ROOT, _SESSION_CWD
    _SESSION_ROOT = None
    _SESSION_CWD = None

def is_session_active() -> bool:
    """Check if we're currently in an active session.
    
    Returns:
        bool: True if session_root is set, False otherwise
    """
    return _SESSION_ROOT is not None

def resolve_session_file_path(file_path: str) -> Path:
    """Resolve a file path in the context of the current session.
    
    For relative paths, resolution order:
    1. Relative to session_cwd (if set)
    2. Relative to session_root (if set)
    3. Relative to current working directory
    
    Args:
        file_path: The file path to resolve
    
    Returns:
        Path: The resolved absolute path
    """
    path = Path(file_path)
    
    if path.is_absolute():
        return path.resolve()
    
    # Try session_cwd first
    if _SESSION_CWD:
        resolved = (_SESSION_CWD / path).resolve()
        if resolved.exists():
            return resolved
    
    # Try session_root next
    if _SESSION_ROOT:
        resolved = (_SESSION_ROOT / path).resolve()
        if resolved.exists():
            return resolved
        # Return session_root-based path even if it doesn't exist
        return resolved
    
    # Fallback to current working directory
    return (Path.cwd() / path).resolve()

def get_context_summary() -> dict:
    """Get a summary of the current path context (useful for debugging).
    
    Returns:
        dict: Summary of current path context
    """
    context = {
        "fractalic_root": str(_FRACTALIC_ROOT),
        "session_root": str(_SESSION_ROOT) if _SESSION_ROOT else None,
        "session_cwd": str(_SESSION_CWD) if _SESSION_CWD else None,
        "current_working_directory": str(Path.cwd()),
        "settings_path": str(get_settings_path()),
        "tools_directory": str(get_tools_directory()),
        "mcp_config_path": str(get_mcp_servers_config_path())
    }
    
    # Add safety and git information
    try:
        validate_session_safety()
        context["session_safe"] = True
        context["git_repository_root"] = str(get_git_repository_root())
    except RuntimeError as e:
        context["session_safe"] = False
        context["safety_error"] = str(e)
        context["git_repository_root"] = None
    
    return context

# Validate that we can find expected files in the fractalic root
def _validate_fractalic_root():
    """Validate that the detected fractalic root contains expected files."""
    # Check if this is a development environment or an installed package
    # In development: we should have pyproject.toml 
    # In installed package: we won't have pyproject.toml, but should have fractalic.py
    
    # Always check for fractalic.py as it's the main entry point
    fractalic_py = _FRACTALIC_ROOT / "fractalic.py" 
    if not fractalic_py.exists():
        raise RuntimeError(
            f"Fractalic root validation failed: fractalic.py not found in {_FRACTALIC_ROOT}. "
            f"This might indicate an incorrect project structure or corrupted installation."
        )
    
    # Check if this looks like a development environment (has pyproject.toml)
    pyproject_toml = _FRACTALIC_ROOT / "pyproject.toml"
    if pyproject_toml.exists():
        # Development environment - also check for requirements.txt
        requirements_txt = _FRACTALIC_ROOT / "requirements.txt"
        if not requirements_txt.exists():
            raise RuntimeError(
                f"Fractalic root validation failed: requirements.txt not found in {_FRACTALIC_ROOT}. "
                f"This might indicate an incorrect project structure."
            )
    # If no pyproject.toml, assume this is an installed package and skip additional checks

# Validate on import
_validate_fractalic_root()
