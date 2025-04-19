# shell_op.py
from sys import stderr, stdout
import subprocess
import time
from typing import Optional
from core.ast_md.node import Node, NodeType, OperationType
from core.ast_md.ast import AST, perform_ast_operation
from core.errors import BlockNotFoundError
from core.config import Config
import os
from core.utils import load_settings
from rich.console import Console
from rich.spinner import Spinner
from rich import print
from rich.status import Status

def clean_shell_command(command: str) -> str:
    """Clean shell command string from YAML escaping"""
    # Remove outer quotes if present
    command = command.strip()
    if (command.startswith('"') and command.endswith('"')) or \
       (command.startswith("'") and command.endswith("'")):
        command = command[1:-1]
    return command

def execute_shell_command(command: str) -> str:
    """Execute shell command and return any output with real-time display."""
    console = Console(force_terminal=True)
    captured_output = []
    
    try:
        env = os.environ.copy()
        if Config.TOML_SETTINGS and 'environment' in Config.TOML_SETTINGS:
            for env_var in Config.TOML_SETTINGS['environment']:
                if 'key' in env_var and 'value' in env_var:
                    env[env_var['key']] = env_var['value']

        start_time = time.time()
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            shell=True,
            env=env,
            bufsize=0,
            universal_newlines=True,
            text=True,
        )

        with console.status("[cyan]@shell[/cyan] processing...") as status:
            import select
            outputs = [process.stdout, process.stderr]
            
            while outputs:
                readable, _, _ = select.select(outputs, [], [])
                
                for output in readable:
                    line = output.readline()
                    if not line:
                        outputs.remove(output)
                        continue
                        
                    if output == process.stderr:
                        if "ERROR" in line or "Traceback" in line:
                            console.print(f"[red]{line.rstrip()}[/red]")
                        else:
                            console.print(line.rstrip())
                        captured_output.append(line)
                    else:
                        console.print(line.rstrip())
                        captured_output.append(line)

        process.wait()
        # If the command returned a non-zero exit code and no output was captured,
        # add a fallback error message.
        if process.returncode != 0:
            fallback = f"Command exited with non-zero code: {process.returncode}"
            if not captured_output or all(not l.strip() for l in captured_output):
                console.print(f"[red]{fallback}[/red]")
                captured_output.append(fallback + "\n")
        
        duration = time.time() - start_time
        mins, secs = divmod(int(duration), 60)
        duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        console.print(f"[light_green]✓[/light_green][cyan] @shell[/cyan] completed ({duration_str})")
        return "".join(captured_output)
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed: {str(e)}[/bold red]")
        return str(e)

def process_shell(ast: AST, current_node: Node) -> Optional[Node]:
    """Process @shell operation with updated schema support"""
    # Get parameters
    params = current_node.params or {}
    prompt = params.get('prompt')
    
    # Validate required parameters
    if not prompt:
        return current_node.next
    
    # Clean command string
    command = clean_shell_command(prompt)
    
    # Get target parameters
    to_params = params.get('to', {})
    target_block_uri = to_params.get('block_uri') if to_params else None
    target_nested = to_params.get('nested_flag', False) if to_params else False
    
    # Execute command
    response = execute_shell_command(command)
    # Store the command output in the current node
    current_node.response_content = response
    
    # Handle header
    header = ""
    use_header = params.get('use-header')
    if use_header is not None:
        if use_header.lower() != "none":
            header = f"{use_header}\n"
    else:
        header = "# Shell response block\n"
        
    response_ast = AST(f"{header}{response}\n")
    # Shell_op.py process_shell logic created_by_file setup
    # After creating the AST from the shell command's response, iterate through all nodes in the AST
    # to set the 'created_by_file' attribute. This ensures that each node knows its origin.
    # The value should be the file path of the current file being processed.
    for node_key, node in response_ast.parser.nodes.items():
        node.role = "user"
        node.created_by = current_node.key  # Store the ID of the operation node that triggered this response
        node.created_by_file = current_node.created_by_file # set the file path

    # Handle targ nodes respons insertion
    operation_type = OperationType(params.get('mode', Config.DEFAULT_OPERATION))
    target_key = current_node.key
    
    if target_block_uri:
        try:
            target_node = ast.get_node_by_path(target_block_uri)
            target_key = target_node.key
        except BlockNotFoundError:
            pass
            
    perform_ast_operation(
        src_ast=response_ast,
        src_path="",
        src_hierarchy=False,
        dest_ast=ast,
        dest_path=target_key,
        dest_hierarchy=target_nested,
        operation=operation_type
    )
    
    return current_node.next