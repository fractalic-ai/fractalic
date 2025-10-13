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
from core.render.render_ast import render_ast_to_markdown, render_ast_to_trace, render_ast_to_markdown_string, render_ast_to_trace_string
from core.operations.import_op import process_import
from core.operations.llm_op import process_llm
from core.operations.goto_op import process_goto
from core.operations.shell_op import process_shell
from core.operations.return_op import process_return
from core.operations.call_tree import CallTreeNode
from core.storage import get_session_storage
from core.simple_token_tracker import token_tracker
from rich import print
from rich.console import Console
from core.paths import set_session_cwd

# Import event emitters for AST visualization and block tracking
from core.event_emitters import emit_ast_snapshot, emit_event
from core.events.types import EventType

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
        committed_files=None, file_commit_hashes=None, base_dir=None, nested_execution_id: str = None) -> Tuple[AST, CallTreeNode, str, str, str, str, str, bool]:
    """Modified return signature to include the return_mode flag at the end.

    Args:
        nested_execution_id: Optional nested execution ID when this run is a child workflow (@run operation)
    """

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

    # Ephemeral session detection (no git side-effects)
    ephemeral = os.environ.get('FRACTALIC_EPHEMERAL_SESSION') == '1' or filename.endswith('.chat_run.md')
    if ephemeral:
        create_new_branch = False  # force disable branch creation / commits

    # Get storage instance and execution_id
    storage = get_session_storage()
    execution_id = os.getenv('FRACTALIC_EXECUTION_ID')

    goto_count = {}
    branch_name = None
    original_cwd = os.getcwd()

    # Flag to track if execution ended with @return operation
    explicit_return = False

    try:
        os.chdir(file_dir)
        # Keep paths session_cwd in sync with the currently executing file directory
        set_session_cwd(file_dir)

        # Set branch_name to execution_id (always set by fractalic.py)
        branch_name = execution_id

        relative_file_path = os.path.relpath(abs_path, base_dir)

        if not os.path.exists(local_file_name):
            raise FileNotFoundError(f"File not found: {local_file_name}")

        # Generate content hash for file
        if not ephemeral:
            with open(local_file_name, 'r', encoding='utf-8') as f:
                file_content = f.read()
            # Generate hash for tracking file version
            import hashlib
            md_commit_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()[:40]
            file_commit_hashes[relative_file_path] = md_commit_hash
        else:
            md_commit_hash = None  # No hash in ephemeral mode

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

            # Emit initial AST snapshot after parsing (before param injection)
            emit_ast_snapshot(ast, operation_type="parse", nested_execution_id=nested_execution_id)
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
        # Initialize call tree node with relative path and new storage fields
        if p_call_tree_node is None:
            call_tree_node = CallTreeNode(
                operation='@run',
                operation_src=None,
                filename=relative_file_path,  # Use relative path
                md_commit_hash=md_commit_hash,
                ctx_commit_hash=None,
                ctx_file=None,
                parent=None,
                # NEW storage fields
                node_id=str(uuid.uuid4())[:12],
                execution_id=execution_id,
                original_file_path=relative_file_path,
                workspace_file_path=abs_path,
                workspace_cwd=file_dir
            )
            call_tree_node = call_tree_node
            new_node = call_tree_node
        else:
            new_node = CallTreeNode(
                operation='@run',
                operation_src=p_parent_operation,
                filename=relative_file_path,  # Use relative path
                md_commit_hash=md_commit_hash,
                ctx_commit_hash=None,
                ctx_file=None,
                parent=p_call_tree_node,
                # NEW storage fields
                node_id=str(uuid.uuid4())[:12],
                execution_id=execution_id,
                original_file_path=relative_file_path,
                workspace_file_path=abs_path,
                workspace_cwd=file_dir
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

        # Emit AST snapshot after param injection (shows input parameters added)
        if param_node:
            emit_ast_snapshot(ast, operation_type="param_inject", nested_execution_id=nested_execution_id)

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
                # Emit block processing event for UI highlighting
                emit_event(
                    EventType.BLOCK_PROCESSING,
                    nested_execution_id=nested_execution_id,
                    block_id=current_node.key or current_node.hash,
                    operation=current_node.name
                )

                operation_name = f"@{current_node.name}"
                if operation_name == "@import":
                    current_node = process_import(ast, current_node)
                    # Emit AST snapshot after import operation
                    emit_ast_snapshot(ast, operation_type="import", nested_execution_id=nested_execution_id)
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
                        base_dir=base_dir,
                        nested_execution_id=nested_execution_id
                    )
                    # Emit AST snapshot after LLM operation
                    emit_ast_snapshot(ast, operation_type="llm", nested_execution_id=nested_execution_id)
                elif operation_name == "@goto":
                    current_node = process_goto(ast, current_node, goto_count)
                    # No AST snapshot for goto (doesn't modify structure)
                elif operation_name == "@shell":
                    current_node = process_shell(ast, current_node)
                    # Emit AST snapshot after shell operation
                    emit_ast_snapshot(ast, operation_type="shell", nested_execution_id=nested_execution_id)
                elif operation_name == "@return":
                    return_result = process_return(ast, current_node)
                    # Emit AST snapshot after return operation
                    emit_ast_snapshot(ast, operation_type="return", nested_execution_id=nested_execution_id)
                    if return_result:
                        ctx_filename = Path(local_file_name).with_suffix('.ctx')
                        trc_filename = Path(local_file_name).with_suffix('.trc')

                        if execution_id:
                            # Storage mode: Save artifacts via storage API
                            ctx_content = render_ast_to_markdown_string(ast)
                            trc_content = render_ast_to_trace_string(ast)

                            ctx_artifact_path = storage.save_node_artifact(
                                execution_id=execution_id,
                                node_id=new_node.node_id,
                                artifact_type='ctx',
                                content=ctx_content,
                                filename=ctx_filename.name
                            )

                            trc_artifact_path = storage.save_node_artifact(
                                execution_id=execution_id,
                                node_id=new_node.node_id,
                                artifact_type='trc',
                                content=trc_content,
                                filename=trc_filename.name
                            )

                            # Update node with artifact paths (relative to artifacts dir)
                            relative_ctx_path = f"{new_node.artifacts_dir}/{ctx_filename.name}"
                            relative_trc_path = f"{new_node.artifacts_dir}/{trc_filename.name}"

                            console.print(f"[light_green]✓[/light_green] storage. context saved: [light_green]{ctx_filename}[/light_green]")
                            console.print(f"[light_green]✓[/light_green] storage. trace file saved: [light_green]{trc_filename}[/light_green]")

                            # For EventMessage compatibility: use md_commit_hash (not artifact path)
                            ctx_hash = md_commit_hash
                            trc_hash = md_commit_hash

                            new_node.ctx_file = relative_ctx_path
                            new_node.ctx_commit_hash = ctx_hash
                            new_node.trc_file = relative_trc_path
                            new_node.trc_commit_hash = trc_hash

                        # Set explicit return flag to True
                        explicit_return = True
                        ctx_hash = new_node.ctx_commit_hash
                        trc_hash = new_node.trc_commit_hash
                        return return_result, new_node, relative_ctx_path, ctx_hash, relative_trc_path, trc_hash, branch_name, explicit_return
                    break  # Exit processing on return
                else:
                    raise UnknownOperationError(f"Unknown operation: {operation_name}")
            else:
                current_node = current_node.next

        ctx_filename = Path(local_file_name).with_suffix('.ctx')
        trc_filename = Path(local_file_name).with_suffix('.trc')

        if execution_id:
            # Storage mode: Save artifacts via storage API
            ctx_content = render_ast_to_markdown_string(ast)
            trc_content = render_ast_to_trace_string(ast)

            # Save artifacts and get relative paths
            ctx_artifact_path = storage.save_node_artifact(
                execution_id=execution_id,
                node_id=new_node.node_id,
                artifact_type='ctx',
                content=ctx_content,
                filename=ctx_filename.name
            )

            trc_artifact_path = storage.save_node_artifact(
                execution_id=execution_id,
                node_id=new_node.node_id,
                artifact_type='trc',
                content=trc_content,
                filename=trc_filename.name
            )

            # Update node with artifact paths (relative to artifacts dir)
            relative_ctx_path = f"{new_node.artifacts_dir}/{ctx_filename.name}"
            relative_trc_path = f"{new_node.artifacts_dir}/{trc_filename.name}"

            # For EventMessage compatibility: use md_commit_hash (not artifact path)
            ctx_commit_hash = md_commit_hash
            trc_commit_hash = md_commit_hash

            console.print(f"[light_green]✓[/light_green] storage. main context saved: [light_green]{ctx_filename}[/light_green]")
            console.print(f"[light_green]✓[/light_green] storage. trace file saved: [light_green]{trc_filename}[/light_green]")

        # Update node with ctx and trc file information
        new_node.ctx_file = relative_ctx_path
        new_node.ctx_commit_hash = ctx_commit_hash
        new_node.trc_file = relative_trc_path
        new_node.trc_commit_hash = trc_commit_hash

        return ast, new_node, relative_ctx_path, ctx_commit_hash, relative_trc_path, trc_commit_hash, branch_name, explicit_return

    except Exception as e:
        import traceback
        import hashlib
        tb = traceback.format_exc()

        ctx_filename = Path(local_file_name).with_suffix('.ctx')
        trc_filename = Path(local_file_name).with_suffix('.trc')

        # Generate error content
        ctx_content = ""
        trc_content = "[]"  # Empty JSON array for trace

        # Only render AST if it was successfully created (linting passed)
        if 'ast' in locals():
            ctx_content = render_ast_to_markdown_string(ast)
            trc_content = render_ast_to_trace_string(ast)
        else:
            # Create context file for linting errors with actual error details
            ctx_content = f"# Linting Errors in {os.path.basename(local_file_name)}\n\n"
            ctx_content += f"File failed linting validation before parsing.\n\n"

            # Include formatted linting errors if available
            if hasattr(e, 'formatted_errors') and e.formatted_errors:
                ctx_content += "## Linting Error Details\n\n"
                ctx_content += "```\n"
                ctx_content += e.formatted_errors
                ctx_content += "\n```\n\n"

        # Append traceback and exception text to the content
        ctx_content += "\n# Exception Trace\n"

        # For linting errors, show the basic message
        if hasattr(e, 'formatted_errors') and e.formatted_errors:
            ctx_content += "Linting validation failed\n"
        else:
            ctx_content += str(e)

        ctx_content += "\n```\n"
        ctx_content += tb
        ctx_content += "```\n"

        if execution_id and 'new_node' in locals():
            # Storage mode: Save error artifacts via storage API
            ctx_artifact_path = storage.save_node_artifact(
                execution_id=execution_id,
                node_id=new_node.node_id,
                artifact_type='ctx',
                content=ctx_content,
                filename=ctx_filename.name
            )

            trc_artifact_path = storage.save_node_artifact(
                execution_id=execution_id,
                node_id=new_node.node_id,
                artifact_type='trc',
                content=trc_content,
                filename=trc_filename.name
            )

            relative_ctx_path = f"{new_node.artifacts_dir}/{ctx_filename.name}"
            relative_trc_path = f"{new_node.artifacts_dir}/{trc_filename.name}"

            # For EventMessage compatibility: use md_commit_hash (not artifact path)
            ctx_commit_hash = md_commit_hash
            trc_commit_hash = md_commit_hash

            console.print(f"[bright_red]✓[/bright_red] storage. context saved with exception info: [bright_red]{ctx_filename}[/bright_red]")
            console.print(f"[bright_red]✓[/bright_red] storage. trace file saved with exception info: [bright_red]{trc_filename}[/bright_red]")

        # Make sure new_node references updated ctx_file and trc_file data (only if it exists)
        if 'new_node' in locals():
            new_node.ctx_file = relative_ctx_path
            new_node.ctx_commit_hash = ctx_commit_hash
            new_node.trc_file = relative_trc_path
            new_node.trc_commit_hash = trc_commit_hash

            # Return results back to fractalic with trace information
            return ast, new_node, new_node.ctx_file, ctx_commit_hash, new_node.trc_file, new_node.trc_commit_hash, branch_name, explicit_return
        else:
            # For linting errors, return minimal valid response
            return None, None, relative_ctx_path, ctx_commit_hash, relative_trc_path, trc_commit_hash, branch_name, False

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

    # Get parent execution_id for nested workflow tracking
    parent_execution_id = os.getenv('FRACTALIC_EXECUTION_ID')

    # Generate UNIQUE node_id for THIS @run invocation
    # NOTE: Each @run must have its own unique node_id for proper nested execution tracking
    unique_node_id = str(uuid.uuid4())[:12]

    # Generate unique execution_id for this nested workflow using the NEW node_id
    nested_execution_id = f"{parent_execution_id}_{unique_node_id}"

    # Emit WORKFLOW_START event for nested agent execution
    # Pass nested_execution_id explicitly instead of using environment variables
    try:
        print(f"[DEBUG @run] Emitting WORKFLOW_START: nested_exec_id={nested_execution_id}, parent={parent_execution_id}, node={unique_node_id}, block={current_node.key or current_node.hash}")
        emit_event(
            EventType.WORKFLOW_START,
            nested_execution_id=nested_execution_id,
            target=os.path.basename(source_path),
            parent_execution_id=parent_execution_id,
            node_id=unique_node_id,
            block_id=current_node.key or current_node.hash
        )
    except Exception as e:
        # Don't fail execution if event emission fails
        print(f"[WARNING] Failed to emit WORKFLOW_START event: {e}")

    # Execute run with updated return signature
    workflow_error = None
    workflow_return_content = None

    try:
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
                base_dir=base_dir,
                nested_execution_id=nested_execution_id  # Pass explicitly to child run
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
                base_dir=base_dir,
                nested_execution_id=nested_execution_id  # Pass explicitly to child run
            )

        # Extract return content if available (after successful run)
        if explicit_return and run_result:
            from core.render.render_ast import render_ast_to_markdown_string
            workflow_return_content = render_ast_to_markdown_string(run_result)

        # Emit TOKEN_USAGE_SUMMARY for nested module BEFORE WORKFLOW_COMPLETE
        # This ensures it appears after all operations in the nested module completed
        try:
            from core.simple_token_tracker import token_tracker
            source_file = os.path.basename(source_path)
            file_stats = token_tracker.get_file_stats(source_file)
            global_stats = token_tracker.get_global_stats()

            if file_stats and (file_stats['file_input_tokens'] > 0 or file_stats['file_output_tokens'] > 0):
                # Get the model from last LLM call in this nested module
                last_call_stats = token_tracker.get_last_call_stats(source_file)
                actual_model = last_call_stats['model'] if last_call_stats else 'unknown'

                emit_event(EventType.TOKEN_USAGE_SUMMARY,
                         nested_execution_id=nested_execution_id,  # This is nested execution
                         block_id=current_node.key or current_node.hash,  # The @run operation block
                         model=actual_model,
                         input_tokens=file_stats['file_input_tokens'],
                         output_tokens=file_stats['file_output_tokens'],
                         total_input=global_stats['global_input_tokens'],
                         total_output=global_stats['global_output_tokens'],
                         source_file=source_file,
                         response_cost=file_stats.get('file_cost', 0.0))
        except Exception as token_e:
            # Don't fail workflow completion if token summary fails
            print(f"[WARNING] Failed to emit nested module token usage summary: {token_e}")

        # Emit WORKFLOW_COMPLETE event IMMEDIATELY after successful execution
        # while nested_execution_id is still in scope (before any cleanup)
        try:
            emit_event(
                EventType.WORKFLOW_COMPLETE,
                nested_execution_id=nested_execution_id,
                target=os.path.basename(source_path),
                parent_execution_id=parent_execution_id,
                node_id=unique_node_id,
                return_content=workflow_return_content,
                explicit_return=explicit_return
            )
        except Exception as e:
            # Don't fail execution if event emission fails
            print(f"[WARNING] Failed to emit WORKFLOW_COMPLETE event: {e}")

    except Exception as workflow_err:
        # Capture error and emit WORKFLOW_ERROR event IMMEDIATELY
        # while nested_execution_id is still in scope
        workflow_error = str(workflow_err)

        try:
            emit_event(
                EventType.WORKFLOW_ERROR,
                nested_execution_id=nested_execution_id,
                target=os.path.basename(source_path),
                parent_execution_id=parent_execution_id,
                node_id=unique_node_id,
                error_message=workflow_error
            )
        except Exception as e:
            # Don't fail execution if event emission fails
            print(f"[WARNING] Failed to emit WORKFLOW_ERROR event: {e}")

        # Re-raise to preserve existing error handling
        raise

    # Handle results insertion
    # If child module didn't execute @return, create error block instead of returning full context
    if not explicit_return:
        # Create error block for missing @return
        error_content = "# Error in module\n\nError - no @return result\n"
        error_ast = AST(error_content)
        run_result = error_ast

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