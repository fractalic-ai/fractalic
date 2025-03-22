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

from rich.console import Console
from rich.panel import Panel

# Set the encoding for standard output, input, and error streams to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

original_open = open

def setup_provider_config(args, settings):
    """Setup provider configuration with proper error handling."""
    # Define provider to environment variable mapping
    PROVIDER_API_KEYS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'groq': 'GROQ_API_KEY'
    }
    
    provider = args.provider.lower()
    
    # Validate provider
    if provider not in PROVIDER_API_KEYS:
        raise ValueError(f"Unsupported provider: {provider}")
        
    api_key_env_var = PROVIDER_API_KEYS[provider]
    provider_settings = settings.get('settings', {}).get(provider, {})
    
    # API key precedence: 1) command line arg, 2) settings.toml, 3) environment variable
    api_key = (args.api_key or 
              provider_settings.get('apiKey') or 
              os.getenv(api_key_env_var))
    
    # Log API key sources (without showing the actual keys)
    console = Console(force_terminal=True, color_system="auto")
    
    def mask_key(key):
        if key and len(key) > 7:
            return f"{key[:5]}........{key[-2:]}"
        return key if key else None

    api_key_masked = mask_key(api_key)
    console.print(f"\n[bold] Provider: [light_green]{provider}[/light_green][/bold]")
    
    if args.api_key:
        console.print(f"[light_green]✓[/light_green] API key from arguments: [light_green]{api_key_masked}[/light_green]")
    else:
        console.print(f"[dim][red]✗[/red] API key from arguments[/dim]")
        
    if provider_settings.get('apiKey'):
        console.print(f"[light_green]✓[/light_green] API key from settings.toml: [light_green]{api_key_masked}[/light_green]")
    else:
        console.print(f"[dim][red]✗[/red] API key from settings.toml[/dim]")
        
    if os.getenv(api_key_env_var):
        console.print(f"[light_green]✓[/light_green] API key from environment variable {api_key_env_var}: [light_green]{api_key_masked}[/light_green]")
    else:
        console.print(f"[dim][red]✗[/red] API key from environment variable {api_key_env_var}[/dim]")

    if not api_key:
        raise ValueError(f"API key for {provider} must be provided either as an argument, in settings.toml, or through the {api_key_env_var} environment variable.")
    
    return provider, api_key, provider_settings

def main():
    settings = load_settings()  # Load settings.toml once
    
    default_provider = settings.get('defaultProvider', 'openai')
    default_operation = settings.get('defaultOperation', 'append')

    parser = argparse.ArgumentParser(description="Process and run operations on a markdown file.")
    parser.add_argument('input_file', type=str, help='Path to the input markdown file.')
    parser.add_argument('--task_file', type=str, help='Path to the task markdown file.')
    parser.add_argument('--api_key', type=str, help='LLM API key', default=None)
    parser.add_argument('--provider', type=str, help='LLM provider (e.g., openai, anthropic, groq)',
                       default=default_provider)
    parser.add_argument('--operation', type=str, help='Default operation to perform',
                       default=default_operation)
    parser.add_argument('--param_input_user_request', type=str,
                       help='Part path for ParamInput-UserRequest', default=None)

    args = parser.parse_args()

    try:
        provider, api_key, provider_settings = setup_provider_config(args, settings)

        Config.TOML_SETTINGS = settings
        Config.LLM_PROVIDER = provider
        Config.API_KEY = api_key
        Config.DEFAULT_OPERATION = args.operation

        os.environ[f"{provider.upper()}_API_KEY"] = api_key

        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

        if args.task_file and args.param_input_user_request:
            if not os.path.exists(args.task_file):
                raise FileNotFoundError(f"Task file not found: {args.task_file}")
                
            temp_ast = parse_file(args.task_file)
            param_node = temp_ast.get_part_by_path(args.param_input_user_request, True)
            result_nodes, call_tree_root, ctx_file, ctx_hash, trc_file, trc_hash, branch_name = run(
                args.input_file,
                param_node,
                p_call_tree_node=None
            )
        else:
            result_nodes, call_tree_root, ctx_file, ctx_hash, trc_file, trc_hash, branch_name = run(
                args.input_file,
                p_call_tree_node=None
            )

        # Save call tree
        abs_path = os.path.abspath(args.input_file)
        file_dir = os.path.dirname(abs_path)
        call_tree_path = os.path.join(file_dir, 'call_tree.json')

        with open(call_tree_path, 'w', encoding='utf-8') as json_file:
            call_tree_root.ctx_file = ctx_file
            call_tree_root.ctx_hash = ctx_hash
            call_tree_root.trc_file = trc_file  
            call_tree_root.trc_hash = trc_hash  
            json_file.write(call_tree_root.to_json())

        md_commit_hash = commit_changes(
            file_dir,
            "Saving call_tree.json",
            [call_tree_path],
            None,
            None
        )

        # Send message to UI for branch information
        print(f"[EventMessage: Root-Context-Saved] ID: {branch_name}, {ctx_hash}")

    except (BlockNotFoundError, UnknownOperationError, FileNotFoundError, ValueError) as e:
        print(f"[ERROR fractalic.py] {str(e)}")
        sys.exit(1)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        filename, line_no, func_name, text = tb[-1]  # Get the last frame (where error originated)
        print(f"[ERROR][Unexpected] {exc_type.__name__} in module {filename}, line {line_no}: {str(e)}")
        traceback.print_exc()

        sys.exit(1)

if __name__ == "__main__":
    main()