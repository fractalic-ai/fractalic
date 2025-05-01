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

def setup_provider_config(args, settings):
    raw_model = args.model or settings.get("defaultProvider")
    if not raw_model:
        raise ValueError("No model specified via --model or defaultProvider in settings.toml")

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
        raise KeyError(
            f'model "{raw_model}" not found under [settings]. '
            f"Available models: {', '.join(all_models.keys())}"
        )

    # use the section name (anthropic/openrouter/openai) as provider
    provider = model_key

    provider_settings = all_models[model_key]
    # ensure downstream sees the actual model name (with dots if present)
    provider_settings = {
        **provider_settings,
        "model": provider_settings.get("model", model_key),
    }

    api_key = (
        args.api_key
        or provider_settings.get("apiKey")
       # or os.getenv(PROVIDER_API_KEYS[provider])
    )
    return provider, api_key, provider_settings

def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (10) + key[-4:]

def main():
    settings = load_settings()  # Load settings.toml once
    
    default_provider = settings.get('defaultProvider', 'openai')
    default_operation = settings.get('defaultOperation', 'append')

    parser = argparse.ArgumentParser(description="Process and run operations on a markdown file.")
    parser.add_argument('input_file', type=str, help='Path to the input markdown file.')
    parser.add_argument('--task_file', type=str, help='Path to the task markdown file.')
    parser.add_argument('--api_key', type=str, help='LLM API key', default=None)
    parser.add_argument(
        "--model",
        help="LLM model to use (overrides settings.defaultProvider)"
    )
    parser.add_argument('--operation', type=str, help='Default operation to perform',
                       default=default_operation)
    parser.add_argument('--param_input_user_request', type=str,
                       help='Part path for ParamInput-UserRequest', default=None)
    parser.add_argument('-v', '--show-operations', action='store_true',
                       help='Make operations visible to LLM (overrides TOML setting)')

    args = parser.parse_args()

    try:
        provider, api_key, provider_settings = setup_provider_config(args, settings)
        
        # Update TOML settings if show-operations flag is explicitly set
        if args.show_operations:
            if 'settings' not in settings:
                settings['settings'] = {}
            settings['settings']['enableOperationsVisibility'] = True
        
        Config.TOML_SETTINGS = settings
        Config.LLM_PROVIDER = provider
        Config.API_KEY = api_key
        Config.DEFAULT_OPERATION = args.operation

        console = Console()  # rich output manager

        # show masked key and its source with icons
        masked = _mask_key(api_key)
        source = "CLI argument" if args.api_key else "settings.toml"
        console.print(f"[bright_green]✓[/bright_green] Using provider [bold]{provider}[/bold], model [bold]{provider_settings.get('model')}[/bold]")
        console.print(f"[bright_green]✓[/bright_green] API key [bold]{masked}[/bold] (from {source})")

        os.environ[f"{provider.upper()}_API_KEY"] = api_key

        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

        if args.task_file and args.param_input_user_request:
            if not os.path.exists(args.task_file):
                raise FileNotFoundError(f"Task file not found: {args.task_file}")
                
            temp_ast = parse_file(args.task_file)
            param_node = temp_ast.get_part_by_path(args.param_input_user_request, True)
            result_nodes, call_tree_root, ctx_file, ctx_hash, trc_file, trc_hash, branch_name, explicit_return = run(
                args.input_file,
                param_node,
                p_call_tree_node=None
            )
        else:
            result_nodes, call_tree_root, ctx_file, ctx_hash, trc_file, trc_hash, branch_name, explicit_return = run(
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
        
        # Log information about how the workflow completed
        if explicit_return:
            print(f"[EventMessage: Execution-Mode] Explicit @return operation")
            print(f"[EventMessage: Return-Nodes-Count] {len(result_nodes.parser.nodes)}")
            
            # Print the content of the returned AST
            print("\n[EventMessage: Return-Content-Start]")
            # Print each node's content in sequence
            current_node = result_nodes.first()
            while current_node:
                print(current_node.content)
                current_node = current_node.next
            print("[EventMessage: Return-Content-End]\n")
        else:
            print(f"[EventMessage: Execution-Mode] Natural workflow completion")
            # No need to print the full AST for natural completion as it's already in the .ctx file

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