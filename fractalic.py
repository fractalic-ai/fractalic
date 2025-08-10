# filepath: /Users/marina/llexem-jan-25-deploy/llexem_deploy_2025/fractalic/fractalic.py
import warnings
# TODO: #5 remove warning supression when pydantic v2 is stable
warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed in V2:",  # match only the first line
    category=UserWarning,
    module="pydantic._internal._config",
)

import os
import sys
import io
import builtins
import argparse
import traceback
import toml
from pathlib import Path

from core.git import commit_changes, ensure_git_repo
from core.ast_md.parser import print_parsed_structure
from core.utils import parse_file, load_settings
from core.config import Config
from core.ast_md.ast import AST
from core.utils import read_file
from core.operations.runner import run
from core.operations.call_tree import CallTreeNode
from core.errors import BlockNotFoundError, UnknownOperationError
from core.render.render_ast import render_ast_to_markdown

from rich.console import Console
from rich.panel import Panel

# Set the encoding for standard output, input, and error streams to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

original_open = open


def run_fractalic(input_file, task_file=None, param_input_user_request=None, capture_output=False, 
                 model=None, api_key=None, operation=None, show_operations=False, context_render_mode=None):
    """
    Run a Fractalic script programmatically - the core execution function.
    
    Args:
        input_file: Path to the Fractalic script to execute
        task_file: Optional path to task file for parameter injection
        param_input_user_request: Optional parameter path for injection
        capture_output: Whether to capture output (not used in current implementation)
        model: LLM model to use (overrides settings.defaultProvider)
        api_key: LLM API key (overrides settings)
        operation: Default operation to perform
        show_operations: Make operations visible to LLM
        context_render_mode: Context rendering mode - "direct" (default) or "json"
    
    Returns:
        dict: Execution result with success status, output, return information, and provider details
    """
    original_cwd = os.getcwd()
    
    try:
        # Load settings
        settings = load_settings()
        
        # Update show-operations setting if explicitly requested
        if show_operations:
            if 'settings' not in settings:
                settings['settings'] = {}
            settings['settings']['enableOperationsVisibility'] = True
        
        # Setup provider configuration
        raw_model = model or settings.get("defaultProvider")
        if not raw_model:
            return {
                'success': False,
                'error': 'No model specified and no defaultProvider in settings.toml',
                'output': '',
                'explicit_return': False,
                'return_content': None,
                'branch_name': None,
                'provider_info': None
            }

        all_models = settings.get("settings", {})
        model_key = None

        # 1) direct match on the table key
        if raw_model in all_models:
            model_key = raw_model
        else:
            # 2) match against each record's "model" field (and common variants)
            for key, conf in all_models.items():
                name = conf.get("model", key)
                if raw_model == name \
                   or raw_model == name.replace(".", "-") \
                   or raw_model == name.replace(".", "_"):
                    model_key = key
                    break
            # 3) fallback: sanitize CLI string to table keys
            if model_key is None:
                for alt in (raw_model.replace(".", "-"), raw_model.replace(".", "_")):
                    if alt in all_models:
                        model_key = alt
                        break

        if model_key is None:
            return {
                'success': False,
                'error': f'model "{raw_model}" not found under [settings]. Available models: {", ".join(all_models.keys())}',
                'output': '',
                'explicit_return': False,
                'return_content': None,
                'branch_name': None,
                'provider_info': None
            }

        # use the section name (anthropic/openrouter/openai) as provider
        provider = model_key
        provider_settings = all_models[model_key]
        # ensure downstream sees the actual model name (with dots if present)
        provider_settings = {
            **provider_settings,
            "model": provider_settings.get("model", model_key),
        }

        # Get API key (prefer parameter, then provider settings)
        final_api_key = api_key or provider_settings.get("apiKey")
        if not final_api_key:
            return {
                'success': False,
                'error': f'No API key found for provider {provider}',
                'output': '',
                'explicit_return': False,
                'return_content': None,
                'branch_name': None,
                'provider_info': None
            }
        
        # Configure globals
        Config.TOML_SETTINGS = settings
        Config.LLM_PROVIDER = provider
        Config.API_KEY = final_api_key
        Config.DEFAULT_OPERATION = operation or settings.get('defaultOperation', 'append')
        Config.CONTEXT_RENDER_MODE = context_render_mode or settings.get('contextRenderMode', 'direct')
        
        # Set environment variable for API key
        os.environ[f"{provider.upper()}_API_KEY"] = final_api_key
        
        # Change working directory to the input file's directory for proper git tracking
        input_file_dir = os.path.dirname(os.path.abspath(input_file))
        
        # Change to the input file's directory so git operations happen in the right place
        os.chdir(input_file_dir)
        print(f"Changed working directory to: {input_file_dir}")
        
        # Reset token stats for this new session
        
        # Validate input file exists
        input_file_basename = os.path.basename(input_file)
        if not os.path.exists(input_file_basename):
            return {
                'success': False,
                'error': f"Input file not found: {input_file}",
                'output': '',
                'explicit_return': False,
                'return_content': None,
                'branch_name': None,
                'provider_info': None
            }
        
        # Handle parameter injection if provided
        param_node = None
        if task_file and param_input_user_request:
            if not os.path.exists(task_file):
                return {
                    'success': False,
                    'error': f"Task file not found: {task_file}",
                    'output': '',
                    'explicit_return': False,
                    'return_content': None,
                    'branch_name': None,
                    'provider_info': None
                }
            temp_ast = parse_file(task_file)
            param_node = temp_ast.get_part_by_path(param_input_user_request, True)
        
        # Initialize variables for exception handling
        result_nodes = None
        call_tree_root = None
        ctx_file = None
        ctx_hash = None
        trc_file = None
        trc_hash = None
        branch_name = None
        explicit_return = False
        return_content = None
        
        try:
            # Run the Fractalic script (use basename since we're in the correct directory)
            result_nodes, call_tree_root, ctx_file, ctx_hash, trc_file, trc_hash, branch_name, explicit_return = run(
                input_file_basename,
                param_node,
                p_call_tree_node=None
            )
            
            # Extract return content if there was an explicit return
            try:
                if explicit_return and result_nodes:
                    # Use the proper render_ast_to_markdown function instead of manual extraction
                    import tempfile
                    
                    try:
                        # Create temporary file to render content
                        with tempfile.NamedTemporaryFile(mode='w+', suffix='.ctx', delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # Render the AST to markdown using the proper function
                        render_ast_to_markdown(result_nodes, temp_path)
                        
                        # Read back the properly rendered content
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            return_content = f.read()
                        
                        # Clean up temporary file
                        os.unlink(temp_path)
                        
                    except Exception as e:
                        # print(f"DEBUG: Error in AST rendering: {e}")
                        # Fallback: try to read from the ctx_file if it exists
                        if ctx_file and os.path.exists(ctx_file):
                            try:
                                with open(ctx_file, 'r', encoding='utf-8') as f:
                                    return_content = f.read()
                                # print(f"DEBUG: Successfully read content from ctx_file: {ctx_file}")
                            except Exception as ctx_e:
                                # print(f"DEBUG: Failed to read ctx_file {ctx_file}: {ctx_e}")
                                pass
                                
            except Exception as e:
                # print(f"DEBUG: Exception in return content extraction: {e}")
                pass
                
        except (BlockNotFoundError, UnknownOperationError, FileNotFoundError, ValueError) as e:
            print(f"[ERROR] Known exception during execution: {str(e)}")
            # These are handled exceptions that don't return useful data
            # Continue to save whatever state we have (which will be None values)
        except Exception as e:
            print(f"[ERROR] Unexpected exception during execution: {str(e)}")
            import traceback
            traceback.print_exc()
            # For unexpected exceptions, the runner should have handled it and returned data
            # But if we get here, the runner couldn't handle it, so variables remain None
        
        # Save call tree regardless of success or failure - this captures the actual execution state
        def save_call_tree_state():
            call_tree_path = os.path.join('.', 'call_tree.json')
            files_to_commit = [call_tree_path]
            
            try:
                if call_tree_root is not None:
                    # Save actual call tree state
                    with open(call_tree_path, 'w', encoding='utf-8') as json_file:
                        call_tree_root.ctx_file = ctx_file
                        call_tree_root.ctx_hash = ctx_hash
                        call_tree_root.trc_file = trc_file  
                        call_tree_root.trc_hash = trc_hash  
                        json_file.write(call_tree_root.to_json())
                else:
                    # Create minimal call tree if no execution happened
                    from core.operations.call_tree import CallTreeNode
                    minimal_call_tree = CallTreeNode(
                        operation='@run',
                        operation_src=None,
                        filename=input_file_basename,
                        ctx_file=ctx_file,
                        trc_file=trc_file
                    )
                    with open(call_tree_path, 'w', encoding='utf-8') as json_file:
                        json_file.write(minimal_call_tree.to_json())
                
                # Check for any uncommitted ctx/trc files and include them in commit
                ctx_file_path = None
                trc_file_path = None
                if ctx_file:
                    ctx_file_path = os.path.join('.', os.path.basename(ctx_file))
                    if os.path.exists(ctx_file_path):
                        files_to_commit.append(ctx_file_path)
                
                if trc_file:
                    trc_file_path = os.path.join('.', os.path.basename(trc_file))
                    if os.path.exists(trc_file_path):
                        files_to_commit.append(trc_file_path)
                
                # Also check for ctx/trc files based on input filename
                base_name = os.path.splitext(input_file_basename)[0]
                potential_ctx = f"{base_name}.ctx"
                potential_trc = f"{base_name}.trc"
                
                if os.path.exists(potential_ctx) and potential_ctx not in files_to_commit:
                    files_to_commit.append(potential_ctx)
                    
                if os.path.exists(potential_trc) and potential_trc not in files_to_commit:
                    files_to_commit.append(potential_trc)
                        
                # Commit all relevant files
                try:
                    md_commit_hash = commit_changes(
                        '.',  # Current directory (which is the input file's directory)
                        "Saving call_tree.json with execution state and any pending files",
                        files_to_commit,
                        None,
                        None
                    )
                    print(f"[INFO] Call tree and files saved and committed: {', '.join(files_to_commit)}")
                except Exception as commit_e:
                    print(f"[WARNING] Files saved but commit failed: {commit_e}")
                    
            except Exception as save_e:
                print(f"[ERROR] Failed to save call tree: {save_e}")
                import traceback
                traceback.print_exc()
        
        # Always save call tree state
        save_call_tree_state()
        
        # Determine if execution was successful
        execution_successful = (result_nodes is not None and call_tree_root is not None)
        
        if execution_successful:
            # Build success output
            output = f"Execution completed. Branch: {branch_name}, Context: {ctx_hash}"
            
            return {
                'success': True,
                'output': output,
                'explicit_return': explicit_return,
                'return_content': return_content,
                'branch_name': branch_name,
                'ctx_file': ctx_file,
                'ctx_hash': ctx_hash,
                # Include provider information for console display
                'provider_info': {
                    'model_key': model_key,
                    'model_name': provider_settings.get('model'),
                    'api_key': final_api_key,
                    'api_key_source': 'CLI argument' if api_key else 'settings.toml'
                }
            }
        else:
            # Build failure output but still include partial state
            error_msg = "Execution failed but call tree state was preserved"
            return {
                'success': False,
                'error': error_msg,
                'output': '',
                'explicit_return': explicit_return,
                'return_content': return_content,
                'branch_name': branch_name,
                'ctx_file': ctx_file,
                'ctx_hash': ctx_hash,
                'provider_info': {
                    'model_key': model_key,
                    'model_name': provider_settings.get('model'),
                    'api_key': final_api_key,
                    'api_key_source': 'CLI argument' if api_key else 'settings.toml'
                }
            }
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (10) + key[-4:]


def enable_rich_terminal_features():
    """Comprehensive test function to check Rich terminal capabilities for xterm."""
    # Force environment variables for maximum xterm compatibility
    os.environ['TERM'] = 'xterm-256color'
    os.environ['COLORTERM'] = 'truecolor'
    os.environ['FORCE_COLOR'] = '1'
    if 'NO_COLOR' in os.environ:
        del os.environ['NO_COLOR']


def main():
    """Main function - thin wrapper that handles argument parsing and calls run_fractalic"""
    # Test Rich terminal capabilities first
    enable_rich_terminal_features()
    
    parser = argparse.ArgumentParser(description="Process and run operations on a markdown file.")
    parser.add_argument('input_file', type=str, help='Path to the input markdown file.')
    parser.add_argument('--task_file', type=str, help='Path to the task markdown file.')
    parser.add_argument('--api_key', type=str, help='LLM API key', default=None)
    parser.add_argument(
        "--model",
        help="LLM model to use (overrides settings.defaultProvider)"
    )
    parser.add_argument('--operation', type=str, help='Default operation to perform',
                       default='append')
    parser.add_argument('--param_input_user_request', type=str,
                       help='Part path for ParamInput-UserRequest', default=None)
    parser.add_argument('-v', '--show-operations', action='store_true',
                       help='Make operations visible to LLM (overrides TOML setting)')
    parser.add_argument('--context-render-mode', choices=['direct', 'json'], default=None,
                       help='Context rendering mode: "direct" (replace JSON with markers, render markdown) or "json" (preserve JSON values, no direct rendering)')

    args = parser.parse_args()

    try:
        # Call the core execution function
        result = run_fractalic(
            input_file=args.input_file,
            task_file=args.task_file,
            param_input_user_request=args.param_input_user_request,
            model=args.model,
            api_key=args.api_key,
            operation=args.operation,
            show_operations=args.show_operations,
            context_render_mode=args.context_render_mode
        )
        
        if not result['success']:
            print(f"[ERROR fractalic.py] {result['error']}")
            sys.exit(1)
        
        # Use same force settings as test function for consistency
        console = Console(
            force_terminal=True, 
            force_interactive=True,
            color_system="truecolor",
            legacy_windows=False
        )

        # Display provider information if available
        if result.get('provider_info'):
            provider_info = result['provider_info']
            masked_key = _mask_key(provider_info['api_key'])
            
            console.print(f"[bright_green]✓[/bright_green] Using provider [bold]{provider_info['model_key']}[/bold], model [bold]{provider_info['model_name']}[/bold]")
            console.print(f"[bright_green]✓[/bright_green] API key [bold]{masked_key}[/bold] (from {provider_info['api_key_source']})")

        # Send message to UI for branch information
        print(f"[EventMessage: Root-Context-Saved] ID: {result['branch_name']}, {result['ctx_hash']}")
        
        # Log information about how the workflow completed
        if result['explicit_return']:
            print(f"[EventMessage: Execution-Mode] Explicit @return operation")
            
            # Print the content of the returned AST
            print("\n[EventMessage: Return-Content-Start]")
            if result['return_content']:
                print(result['return_content'])
            print("[EventMessage: Return-Content-End]\n")
            
            # Print token usage summary at end of session
            from core.simple_token_tracker import token_tracker
            token_tracker.print_session_summary()
        else:
            print(f"[EventMessage: Execution-Mode] Natural workflow completion")
        
        # Print token usage summary at end of session
        from core.simple_token_tracker import token_tracker
        token_tracker.print_session_summary()


    except (BlockNotFoundError, UnknownOperationError, FileNotFoundError, ValueError) as e:
        print(f"[ERROR fractalic.py] {str(e)}")
        sys.exit(1)
    except Exception as e:
        # Check if this is a linting error and try to get context information
        if e.__class__.__name__ == 'FractalicLintError':
            print(f"[ERROR fractalic.py] Linting failed: {str(e)}")
            
            # Try to extract any context information that might have been generated
            # Look for recently created branch and context files
            try:
                from core.git import get_current_git_branch, get_latest_commit_hash
                import glob
                import os
                
                # Get current branch name (should be the test branch created for this run)
                current_branch = get_current_git_branch()
                
                # Look for context files that were just created
                ctx_files = glob.glob("*.ctx")
                if ctx_files:
                    # Get the most recent context file
                    latest_ctx = max(ctx_files, key=os.path.getctime)
                    ctx_hash = get_latest_commit_hash()
                    
                    print(f"[EventMessage: Root-Context-Saved] ID: {current_branch}, {ctx_hash}")
                    print(f"[EventMessage: Execution-Mode] Linting validation failed")
                    
                    # Check if context file contains linting details
                    if os.path.exists(latest_ctx):
                        with open(latest_ctx, 'r', encoding='utf-8') as f:
                            ctx_content = f.read()
                            if "Linting Error Details" in ctx_content:
                                print(f"[EventMessage: Linting-Errors] Details saved to {latest_ctx}")
                
            except Exception as ctx_e:
                # If we can't get context info, just continue with basic error reporting
                print(f"[DEBUG] Could not extract context information: {ctx_e}")
            
            sys.exit(1)
        
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        filename, line_no, func_name, text = tb[-1]  # Get the last frame (where error originated)
        print(f"[ERROR][Unexpected] {exc_type.__name__} in module {filename}, line {line_no}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
