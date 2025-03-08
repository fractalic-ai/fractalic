# LLM Operation
# - process_llm

from typing import Optional
from pathlib import Path
import time

from core.ast_md.node import Node, OperationType, NodeType
from core.ast_md.ast import AST, get_ast_part_by_path, perform_ast_operation
from core.errors import BlockNotFoundError
from core.config import Config
from core.llm.llm_client import LLMClient  # Import the LLMClient class
from rich.console import Console
from rich.spinner import Spinner
from rich import print
from rich.status import Status

# Assuming LLM_PROVIDER and API_KEY are globally set in fractalic.py
# You can initialize LLMClient here if it's a singleton


def process_llm(ast: AST, current_node: Node) -> Optional[Node]:
    """Process @llm operation with updated schema support"""
    console = Console(force_terminal=True)

    def get_previous_headings(node: Node) -> str:
        context = []
        current = ast.first()
        while current and current != node:
            if current.type == NodeType.HEADING:
                context.append(current.content)
            current = current.next
        return "\n\n".join(context)
    
    def get_previous_heading_messages(node: Node) -> list:
        """Return a list of messages for each heading node encountered before the given node."""
        messages = []
        current = ast.first()
        while current and current != node:
            if current.type == NodeType.HEADING:
                # Use the node's role attribute, defaulting to "user" if not specified
                role = getattr(current, "role", "user")
                messages.append({"role": role, "content": current.content})
            current = current.next
        return messages

    # Get parameters
    params = current_node.params or {}
    prompt = params.get('prompt')
    block_params = params.get('block', {})

    # New optional fields
    provider = params.get('provider')
    model = params.get('model')

    # Validate at least one of prompt/block is provided
    if not prompt and not block_params:
        raise ValueError("@llm operation requires either 'prompt' or 'block' parameter")

    # Get target parameters 
    to_params = params.get('to', {})
    target_block_uri = to_params.get('block_uri') if to_params else None
    target_nested = to_params.get('nested_flag', False) if to_params else False

    # Build prompt parts based on parameters
    prompt_parts = []
    messages = []  # Parallel collection of messages with roles

    # Handle blocks first - can be single block or array
    if block_params:
        if block_params.get('is_multi'):
            # Handle array of blocks
            blocks = block_params.get('blocks', [])
            for block_info in blocks:
                try:
                    block_uri = block_info.get('block_uri')
                    nested_flag = block_info.get('nested_flag', False)
                    block_ast = get_ast_part_by_path(ast, block_uri, nested_flag)
                    if block_ast.parser.nodes:
                        # Keep existing prompt_parts logic
                        block_content = "\n\n".join(node.content for node in block_ast.parser.nodes.values())
                        prompt_parts.append(block_content)
                        
                        # Build messages - one message per block with individual node contents
                        role = block_info.get('role', 'user')
                        # Add each node's content as a separate message with the same role
                        for node in block_ast.parser.nodes.values():
                            messages.append({"role": role, "content": node.content})
                except BlockNotFoundError:
                    raise ValueError(f"Block with URI '{block_uri}' not found")
        else:
            # Handle single block
            try:
                block_uri = block_params.get('block_uri')
                nested_flag = block_params.get('nested_flag', False)
                block_ast = get_ast_part_by_path(ast, block_uri, nested_flag)
                if block_ast.parser.nodes:
                    # Keep existing prompt_parts logic
                    block_content = "\n\n".join(node.content for node in block_ast.parser.nodes.values())
                    prompt_parts.append(block_content)
                    
                    # Build messages - one message per node in the block
                    role = block_params.get('role', 'user')
                    # Add each node's content as a separate message with the same role
                    for node in block_ast.parser.nodes.values():
                        messages.append({"role": role, "content": node.content})
            except BlockNotFoundError:
                raise ValueError(f"Block with URI '{block_uri}' not found")

    # Add context if no blocks are explicitly specified 
    elif prompt:
        # Keep existing prompt_parts logic
        context = get_previous_headings(current_node)
        if context:
            prompt_parts.append(context)
            
        # Add heading messages  
        heading_messages = get_previous_heading_messages(current_node)
        messages.extend(heading_messages)
        # if Config.DEBUG and heading_messages:
        #    console.print(f"[yellow]Added {len(heading_messages)} previous heading messages[/yellow]")

    # Add prompt if specified (always last)
    if prompt:
        prompt_parts.append(prompt)
        messages.append({"role": "user", "content": prompt})

    # Combine all parts with proper spacing
    prompt_text = "\n\n".join(part.strip() for part in prompt_parts if part.strip())

    # Call LLM - use messages if available, otherwise fall back to prompt_text
    llm_provider = provider if provider else Config.LLM_PROVIDER
    llm_model = model if model else Config.MODEL
    llm_client = LLMClient(provider=llm_provider, model=llm_model)
    actual_model = model or (getattr(llm_client.client, "settings", {}).get("model"))

    # Determine whether to use messages or prompt_text
    #use_messages = params.get('use_messages', False) and len(messages) > 0
    #llm_input = messages if use_messages else prompt_text

    start_time = time.time()
    try:
        with console.status(
            f"[cyan] @llm [turquoise2]({llm_provider}/{actual_model}"
            f"{('/' + llm_client.base_url) if hasattr(llm_client, 'base_url') and llm_client.base_url else ''})[/turquoise2]"
            f"[/cyan] processing...",
            spinner="dots"
        ) as status:
            response = llm_client.llm_call(prompt_text, messages, params)

        duration = time.time() - start_time
        mins, secs = divmod(int(duration), 60)
        duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        console.print(
            f"[light_green]✓[/light_green][cyan] @llm [turquoise2]({llm_provider}/{actual_model}"
            f"{('/' + llm_client.base_url) if hasattr(llm_client, 'base_url') and llm_client.base_url else ''})[/turquoise2]"
            f"[/cyan] completed ({duration_str})"
        )
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed: {str(e)}[/bold red]")
        console.print(f"[bold red]  Operation content:[/bold red]\n{current_node.content}")
        raise

    # Get save-to-file parameter
    save_to_file = params.get('save-to-file')

    # Save raw response to file if save_to_file is specified
    if save_to_file:
        file_path = Path(save_to_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(response)

    # Handle header
    header = ""
    use_header = params.get('use-header')
    if use_header is not None:
        if use_header.lower() != "none":
            header = f"{use_header}\n"
    else:
        header = "# LLM response block\n"

    response_ast = AST(f"{header}{response}\n")
    for node_key, node in response_ast.parser.nodes.items():
        node.role = "assistant"

    # Handle target block insertion
    operation_type = OperationType(params.get('mode', Config.DEFAULT_OPERATION))

    if target_block_uri:
        try:
            target_node = ast.get_node_by_path(target_block_uri)
            target_key = target_node.key
        except BlockNotFoundError:
            raise ValueError(f"Target block '{target_block_uri}' not found")
    else:
        target_key = current_node.key

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