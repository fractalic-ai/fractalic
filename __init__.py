"""
Fractalic - The Agentic Development Environment

Turn simple documents into powerful, AI-native applications.
"""

try:
    from importlib.metadata import version
    __version__ = version("fractalic")
except ImportError:
    # Fallback for older Python versions
    __version__ = "unknown"

# Expose main functions without importing dependencies yet
def main():
    """CLI entry point for Fractalic."""
    from .fractalic import main as _main
    return _main()

def run_fractalic(*args, **kwargs):
    """Programmatic entry point for Fractalic."""
    from .fractalic import run_fractalic as _run_fractalic
    return _run_fractalic(*args, **kwargs)

def run(filename, prompt=None, **kwargs):
    """
    Simple interface to run a Fractalic workflow file.
    
    Args:
        filename: Path to the Fractalic markdown file
        prompt: Optional text prompt to inject at the beginning of context
        **kwargs: Additional parameters (model, api_key, etc.)
    
    Returns:
        dict: Execution result with keys:
            - success (bool): Whether execution succeeded
            - output (str): Execution output message
            - explicit_return (bool): Whether workflow used @return operation
            - return_content (str|None): Content returned by @return operation
            - branch_name (str|None): Git branch name created for execution
            - ctx_file (str|None): Path to context file (.ctx)
            - ctx_hash (str|None): Git commit hash for context
            - provider_info (dict|None): Model and API key information
            - error (str): Error message (only if success=False)
    """
    from .fractalic import run_fractalic as _run_fractalic
    
    if prompt:
        # Create a simple object with content attribute
        # This will be wrapped in "# Input Parameters" header by the runner
        class SimpleNode:
            def __init__(self, content):
                self.content = content
        
        param_node = SimpleNode(prompt)
        # Pass the param_node directly - no task_file needed
        return _run_fractalic(input_file=filename, param_node=param_node, **kwargs)
    else:
        return _run_fractalic(input_file=filename, **kwargs)

def run_content(content, prompt=None, **kwargs):
    """
    Run Fractalic workflow from markdown content string.
    
    Args:
        content: Markdown content as string
        prompt: Optional text prompt to inject at the beginning of context
        **kwargs: Additional parameters (model, api_key, etc.)
    
    Returns:
        dict: Execution result with keys:
            - success (bool): Whether execution succeeded
            - output (str): Execution output message
            - explicit_return (bool): Whether workflow used @return operation
            - return_content (str|None): Content returned by @return operation
            - branch_name (str|None): Git branch name created for execution
            - ctx_file (str|None): Path to context file (.ctx)
            - ctx_hash (str|None): Git commit hash for context
            - provider_info (dict|None): Model and API key information
            - error (str): Error message (only if success=False)
    """
    import tempfile
    import os
    from .fractalic import run_fractalic as _run_fractalic
    
    # Create temporary file for content
    fd, temp_path = tempfile.mkstemp(suffix='.md', text=True)
    try:
        # Write content to temp file
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if prompt:
            # Create a simple object with content attribute
            class SimpleNode:
                def __init__(self, content):
                    self.content = content
            
            param_node = SimpleNode(prompt)
            return _run_fractalic(input_file=temp_path, param_node=param_node, **kwargs)
        else:
            return _run_fractalic(input_file=temp_path, **kwargs)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

__all__ = ["main", "run_fractalic", "run", "run_content", "__version__"]
