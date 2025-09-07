# runner.py

import os
import uuid
from typing import Optional, Tuple, Union
from pathlib import Path

from core.ast_md.ast import AST, get_ast_part_by_id, perform_ast_operation, get_ast_part_by_path
from core.ast_md.node import Node, NodeType, OperationType
from core.errors import BlockNotFoundError, UnknownOperationError
from core.config import Config
from core.utils import parse_file, get_content_without_header
from core.render.render_ast import render_ast_to_markdown, render_ast_to_trace
from core.operations.import_op import process_import
from core.operations.llm_op import process_llm
from core.operations.goto_op import process_goto
from core.operations.shell_op import process_shell
from core.operations.return_op import process_return
from core.operations.call_tree import CallTreeNode
from core.git import ensure_git_repo, create_session_branch, commit_changes
from core.simple_token_tracker import token_tracker
from rich import print
from rich.console import Console

def get_relative_path(base_dir: str, file_path: str) -> str:
    """Convert absolute path to relative path based on base directory."""
    try:
        return os.path.relpath(file_path, base_dir)
    except ValueError:
        return file_path

def print_ast_state(ast):
    current_node = ast.first()
    while current_node:
        if current_node.type == NodeType.OPERATION:
            print(f"\nNode Hash: {current_node.hash}, Type: {current_node.type}, Operation: @{current_node.name} {current_node.content}, Enabled: {current_node.enabled}")
        else:
            print(f"Node Hash: {current_node.hash}, Type: {current_node.type}, Enabled: {current_node.enabled}")
        current_node = current_node.next

def run(filename: str, param_node: Optional[Union[Node, AST]] = None, create_new_branch: bool = True,
        p_parent_filename=None, p_parent_operation: str = None, p_call_tree_node=None,
        committed_files=None, file_commit_hashes=None, base_dir=None) -> Tuple[AST, CallTreeNode, str, str, str, str, str, bool]:
    """Modified return signature to include the return_mode flag at the end."""
 
    console = Console(force_terminal=True, color_system="auto")
    if committed_files is None:
        committed_files = set()
    if file_commit_hashes is None:
        file_commit_hashes = {}

    abs_path = os.path.abspath(filename)
    file_dir = os.path.dirname(abs_path)
    local_file_name = os.path.basename(abs_path)

    if base_dir is None and create_new_branch:
        base_dir = file_dir

    goto_count = {}
    branch_name = None
    original_cwd = os.getcwd()
    
    # Flag to track if execution ended with @return operation
    explicit_return = False
    
    try:
        os.chdir(file_dir)

        if create_new_branch:
            ensure_git_repo(base_dir)
            branch_name = create_session_branch(base_dir, "Testing-git-operations")

            console.print(f"[light_green]✓[/light_green] git. new branch created: [cyan]{branch_name}[/cyan]")

        relative_file_path = os.path.relpath(abs_path, base_dir)

        if not os.path.exists(local_file_name):
            raise FileNotFoundError(f"File not found: {local_file_name}")

        if relative_file_path not in committed_files:
            try:
                md_commit_hash = commit_changes(
                    base_dir,
                    "Operation [@run] execution start",
                    [local_file_name],
                    p_parent_filename,
                    p_parent_operation
                )
                committed_files.add(relative_file_path)
                file_commit_hashes[relative_file_path] = md_commit_hash
            except Exception as e:
                print(f"[ERROR runner.py] Error committing file {relative_file_path}: {str(e)}")
                raise

        # RESTORING LOGIC    
        else:
            md_commit_hash = file_commit_hashes[relative_file_path]

        # RESTORING LOGIC  
        # Process the AST
        try:
            ast = parse_file(local_file_name)
            
            # Set source file attribute on AST for token tracking
            ast.source_file = local_file_name
            
            # Initialize token tracking for this file
            token_tracker.start_file(local_file_name)
            
            # Runner py run logic created_by_file setup
            """
            After parsing, iterate through all nodes in the AST to set the 'created_by_file' attribute.
            This attribute is crucial for tracking the origin of each node, especially when dealing with multiple files or nested operations.
            The value should be the absolute path of the file being processed.
            """
            for node in ast.parser.nodes.values():
                node.created_by_file = local_file_name    
        except Exception as e:
            print(f"[ERROR runner.py] Error parsing file {local_file_name}: {str(e)}")
            print(f"[ERROR runner.py] Current directory: {os.getcwd()}")
            print(f"[ERROR runner.py] File exists: {os.path.exists(local_file_name)}")
            print(f"[ERROR runner.py] File contents:")
            try:
                with open(local_file_name, 'r', encoding='utf-8') as f:
                    print(f.read())
            except Exception as read_error:
                print(f"[ERROR runner.py] Could not read file: {str(read_error)}")
            raise

        # RESTORING LOGIC 
        # Initialize call tree node with relative path
        if p_call_tree_node is None:
            call_tree_node = CallTreeNode(
                operation='@run',
                operation_src=None,
                filename=relative_file_path,  # Use relative path
                md_commit_hash=md_commit_hash,
                ctx_commit_hash=None,
                ctx_file=None,
                parent=None
            )
            new_node = call_tree_node
        else:
            new_node = CallTreeNode(
                operation='@run',
                operation_src=p_parent_operation,
                filename=relative_file_path,  # Use relative path
                md_commit_hash=md_commit_hash,
                ctx_commit_hash=None,
                ctx_file=None,
                parent=p_call_tree_node
            )
            p_call_tree_node.add_child(new_node)

        if param_node:
            if isinstance(param_node, AST):
                ast.prepend_node_with_ast(ast.first().key, param_node)
            else:
                decorated_param_node = Node(
                    type=NodeType.HEADING,
                    name="Input Parameters",
                    level=1,
                    content=f"{param_node.content}",
                    id="input-parameters",
                    key=str(uuid.uuid4())[:8]
                )
                param_ast = AST("")
                param_ast.parser.nodes = {decorated_param_node.key: decorated_param_node}
                param_ast.parser.head = decorated_param_node
                param_ast.parser.tail = decorated_param_node
                ast.prepend_node_with_ast(ast.first().key, param_ast)

        # RESTORING LOGIC
        current_node = ast.first()

        while current_node:
            # Skip processing if the node is disabled
            if hasattr(current_node, 'enabled') and current_node.enabled is False:
                current_node = current_node.next
                continue
            
            if current_node.params and current_node.params.get("run-once") is True:
                current_node.enabled = False

            if current_node.type == NodeType.OPERATION:
                operation_name = f"@{current_node.name}"
                if operation_name == "@import":
                    current_node = process_import(ast, current_node)
                elif operation_name == "@run":
                    current_node, child_node, run_ctx_file, run_ctx_hash, run_trc_file, run_trc_hash, _, child_explicit_return = process_run(
                        ast,
                        current_node,
                        local_file_name,
                        current_node.content.strip(),
                        new_node,  # Pass new_node instead of p_call_tree_node
                        committed_files=committed_files,
                        file_commit_hashes=file_commit_hashes,
                        base_dir=base_dir
                    )
                elif operation_name == "@llm":
                    current_node = process_llm(
                        ast, 
                        current_node,
                        call_tree_node=new_node,
                        committed_files=committed_files,
                        file_commit_hashes=file_commit_hashes,
                        base_dir=base_dir
                    )
                elif operation_name == "@goto":
                    current_node = process_goto(ast, current_node, goto_count)
                elif operation_name == "@shell":
                    current_node = process_shell(ast, current_node)
                elif operation_name == "@return":
                    return_result = process_return(ast, current_node)
                    if return_result:
                        ctx_filename = Path(local_file_name).with_suffix('.ctx')
                        output_file = os.path.join(file_dir, ctx_filename)

                        trc_filename = Path(local_file_name).with_suffix('.trc')
                        trc_output_file = os.path.join(file_dir, trc_filename)

                        relative_ctx_path = get_relative_path(base_dir, output_file)
                        relative_trc_path = get_relative_path(base_dir, trc_output_file)
                        
                        render_ast_to_markdown(ast, output_file)
                        render_ast_to_trace(ast, trc_output_file)

                        ctx_commit_hash = commit_changes(
                            base_dir,
                            "@return operation",
                            [local_file_name, ctx_filename, trc_filename],  # Include trc_filename
                            p_parent_filename,
                            p_parent_operation
                        )
                        
                        console.print(f"[light_green]✓[/light_green] git. context commited: [light_green]{ctx_filename}[/light_green]")
                        console.print(f"[light_green]✓[/light_green] git. trace file commited: [light_green]{trc_filename}[/light_green]")

                        new_node.ctx_file = relative_ctx_path
                        new_node.ctx_commit_hash = ctx_commit_hash
                        new_node.trc_file = relative_trc_path
                        new_node.trc_commit_hash = ctx_commit_hash  # Same commit hash as ctx

                        # Set explicit return flag to True
                        explicit_return = True
                        return return_result, new_node, relative_ctx_path, ctx_commit_hash, relative_trc_path, ctx_commit_hash, branch_name, explicit_return
                    break  # Exit processing on return
                else:
                    raise UnknownOperationError(f"Unknown operation: {operation_name}")
            else:
                current_node = current_node.next

        ctx_filename = Path(local_file_name).with_suffix('.ctx')
        output_file = os.path.join(file_dir, ctx_filename)

        trc_filename = Path(local_file_name).with_suffix('.trc')
        trc_output_file = os.path.join(file_dir, trc_filename)
        
        relative_ctx_path = os.path.relpath(output_file, base_dir)
        relative_trc_path = os.path.relpath(trc_output_file, base_dir)
        
        render_ast_to_markdown(ast, output_file)
        render_ast_to_trace(ast, trc_output_file)

        ctx_commit_hash = commit_changes(
            base_dir,
            "Final processed files",
            [local_file_name, ctx_filename, trc_filename],  # Include trc_filename in commit
            p_parent_filename,
            p_parent_operation
        )
        console.print(f"[light_green]✓[/light_green] git. main context commited: [light_green]{ctx_filename}[/light_green]")
        console.print(f"[light_green]✓[/light_green] git. trace file commited: [light_green]{trc_filename}[/light_green]")

        # Update node with ctx and trc file information
        new_node.ctx_file = relative_ctx_path
        new_node.ctx_commit_hash = ctx_commit_hash
        new_node.trc_file = relative_trc_path
        new_node.trc_commit_hash = ctx_commit_hash  # Same commit hash as ctx

        return ast, new_node, relative_ctx_path, ctx_commit_hash, relative_trc_path, ctx_commit_hash, branch_name, explicit_return

    except Exception as e:
        import traceback
        tb = traceback.format_exc()

        # Same logic to render .ctx:
        ctx_filename = Path(local_file_name).with_suffix('.ctx')
        output_file = os.path.join(file_dir, ctx_filename)

        trc_filename = Path(local_file_name).with_suffix('.trc')
        trc_output_file = os.path.join(file_dir, trc_filename)

        # Only render AST if it was successfully created (linting passed)
        if 'ast' in locals():
            render_ast_to_markdown(ast, output_file)
            render_ast_to_trace(ast, trc_output_file)
        else:
            # Create context file for linting errors with actual error details
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Linting Errors in {os.path.basename(local_file_name)}\n\n")
                f.write(f"File failed linting validation before parsing.\n\n")
                
                # Include formatted linting errors if available
                if hasattr(e, 'formatted_errors') and e.formatted_errors:
                    f.write("## Linting Error Details\n\n")
                    f.write("```\n")
                    f.write(e.formatted_errors)
                    f.write("\n```\n\n")
            
            # Create empty trace file
            with open(trc_output_file, 'w', encoding='utf-8') as f:
                f.write("[]")  # Empty JSON array

        # Append traceback and exception text to the .ctx file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\n# Exception Trace\n")
            
            # For linting errors, show the basic message
            if hasattr(e, 'formatted_errors') and e.formatted_errors:
                f.write("Linting validation failed\n")
            else:
                f.write(str(e))
            
            f.write("\n```\n")
            f.write(tb)
            f.write("```\n")

        # Commit changes without modifying git functions
        ctx_commit_hash = commit_changes(
            base_dir,
            "Exception caught: appended traceback",
            [local_file_name, ctx_filename, trc_filename],  # Include trc_filename
            p_parent_filename,
            p_parent_operation
        )
        
        relative_ctx_path = get_relative_path(base_dir, output_file)
        relative_trc_path = get_relative_path(base_dir, trc_output_file)
        
        console.print(f"[bright_red]✓[/bright_red] git. context commited with exception info: [bright_red]{ctx_filename}[/bright_red]")
        console.print(f"[bright_red]✓[/bright_red] git. trace file commited with exception info: [bright_red]{trc_filename}[/bright_red]")

        # Make sure new_node references updated ctx_file and trc_file data (only if it exists)
        if 'new_node' in locals():
            new_node.ctx_file = relative_ctx_path
            new_node.ctx_commit_hash = ctx_commit_hash
            new_node.trc_file = relative_trc_path
            new_node.trc_commit_hash = ctx_commit_hash  # Same commit hash as ctx

            # Return results back to fractalic with trace information
            return ast, new_node, new_node.ctx_file, ctx_commit_hash, new_node.trc_file, new_node.trc_commit_hash, branch_name, explicit_return
        else:
            # For linting errors, return minimal valid response
            return None, None, relative_ctx_path, ctx_commit_hash, relative_trc_path, ctx_commit_hash, branch_name, False

    finally:
        os.chdir(original_cwd)

def process_run(ast: AST, current_node: Node, local_file_name, parent_operation, call_tree_node,
                committed_files=None, file_commit_hashes=None, base_dir=None) -> Optional[Tuple[Node, CallTreeNode, str, str, str, str, str, bool]]:
    """Modified return signature to include explicit_return flag."""
    
    params = current_node.params
    if not params:
        raise ValueError("No parameters found for @run operation.")

    # Source file parameters
    src_params = params.get('file', {})
    src_file_path = src_params.get('path', '')
    src_file_name = src_params.get('file', '')

    # Action and operation type
    action = params.get('mode', Config.DEFAULT_OPERATION) 
    operation_type = OperationType(action)

    # Target parameters
    target_params = params.get('to', {})
    target_block_id = target_params.get('block_uri', '')
    target_nested = target_params.get('nested_flag', False)

    # Handle prompt or block parameter
    prompt = params.get('prompt')
    block_params = params.get('block', {})
    block_uri = block_params.get('block_uri', '')
    nested_flag = block_params.get('nested_flag', False)
    use_header = params.get('use-header')
    
    # Initialize parameter_value to avoid UnboundLocalError
    parameter_value = None

    # Create an empty input AST that will hold all input blocks
    input_ast = None
    
    # Handle blocks first (can be single block or array)
    if block_params:
        try:
            if block_params.get('is_multi'):
                # Handle array of blocks
                blocks = block_params.get('blocks', [])
                for block_info in blocks:
                    block_uri = block_info.get('block_uri')
                    nested_flag = block_info.get('nested_flag', False)
                    
                    block_ast = get_ast_part_by_path(ast, block_uri, nested_flag)
                    if not block_ast.parser.nodes:
                        raise BlockNotFoundError(f"Block with path '{block_uri}' is empty.")
                        
                    if input_ast:
                        # Stack blocks by appending
                        perform_ast_operation(
                            src_ast=block_ast,
                            src_path='',
                            src_hierarchy=False,
                            dest_ast=input_ast,
                            dest_path=input_ast.parser.tail.key,
                            dest_hierarchy=False,
                            operation=OperationType.APPEND
                        )
                    else:
                        # First block becomes base AST
                        input_ast = block_ast
            else:
                # Handle single block (existing logic)
                block_uri = block_params.get('block_uri')
                nested_flag = block_params.get('nested_flag', False)
                
                block_ast = get_ast_part_by_path(ast, block_uri, nested_flag)
                if not block_ast.parser.nodes:
                    raise BlockNotFoundError(f"Block with path '{block_uri}' is empty.")
                input_ast = block_ast
                
        except BlockNotFoundError as e:
            raise BlockNotFoundError(f"Error processing blocks: {str(e)}")
    
    # Handle prompt if specified (append to blocks if present)
    if prompt:
        header = ""
        if use_header is not None:
            if use_header.lower() != "none":
                header = f"{use_header}\n"
        else:
            header = "# Input Parameters {id=input-parameters}\n"
            
        parameter_value = f"{header}{prompt}"
        param_node = Node(
            type=NodeType.HEADING,
            name="Input Parameters",
            level=1,
            content=parameter_value,
            id="input-parameters",
            role="user",
            key=str(uuid.uuid4())[:8],
            created_by = current_node.key, # Store the key of the operation node that triggered this response
            created_by_file = local_file_name
        )
        
        # Create prompt AST
        prompt_ast = AST("")
        prompt_ast.parser.nodes = {param_node.key: param_node}
        prompt_ast.parser.head = param_node
        prompt_ast.parser.tail = param_node
        
        if input_ast:
            # Append prompt to existing blocks
            perform_ast_operation(
                src_ast=prompt_ast,
                src_path='',
                src_hierarchy=False,
                dest_ast=input_ast,
                dest_path=input_ast.parser.tail.key,
                dest_hierarchy=False,
                operation=OperationType.APPEND
            )
        else:
            # No blocks, use prompt as input
            input_ast = prompt_ast

    # Handle file execution
    current_dir = os.path.dirname(os.path.abspath(local_file_name))
    source_path = os.path.abspath(os.path.join(current_dir, src_file_path, src_file_name))

    if not os.path.exists(source_path):
        raise ValueError(f"Source file not found: {source_path}")

    # Create parameter node if we have content
    param_node = None
    if parameter_value:
        param_node = Node(
            type=NodeType.HEADING,
            name="Input Parameters",
            level=1,
            content=parameter_value,
            id="input-parameters",
            key=str(uuid.uuid4())[:8]
        )

    # Execute run with updated return signature
    if input_ast and input_ast.parser.nodes:
        run_result, child_call_tree_node, ctx_file, ctx_file_hash, trc_file, trc_file_hash, branch_name, explicit_return = run(
            source_path,
            input_ast,  # Pass the complete input AST
            False,
            local_file_name,
            parent_operation,
            call_tree_node,
            committed_files=committed_files,
            file_commit_hashes=file_commit_hashes,
            base_dir=base_dir
        )
    else:
        run_result, child_call_tree_node, ctx_file, ctx_file_hash, trc_file, trc_file_hash, branch_name, explicit_return = run(
            source_path,
            None,
            False,
            local_file_name,
            parent_operation,
            call_tree_node,
            committed_files=committed_files,
            file_commit_hashes=file_commit_hashes,
            base_dir=base_dir
        )

    # Handle results insertion
    if target_block_id:
        perform_ast_operation(
            run_result,
            run_result.first().key,
            True,
            ast,
            target_block_id,
            target_nested,
            operation_type,
            False
        )
    else:
        perform_ast_operation(
            run_result,
            run_result.first().key,
            True,
            ast,
            current_node.key,
            False,
            operation_type,
            False
        )

    # Return the explicit_return flag as well
    return current_node.next, child_call_tree_node, ctx_file, ctx_file_hash, trc_file, trc_file_hash, branch_name, explicit_return