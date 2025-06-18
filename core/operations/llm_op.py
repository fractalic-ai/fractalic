# LLM Operation
# - process_llm

from typing import Optional
from pathlib import Path
import time

from core.ast_md.node import Node, OperationType, NodeType
from core.ast_md.ast import AST, get_ast_part_by_path, get_ast_parts_by_uri_array, perform_ast_operation
from core.errors import BlockNotFoundError
from core.config import Config
from core.llm.llm_client import LLMClient  # Import the LLMClient class
from rich.console import Console
from rich.spinner import Spinner
from rich import print
from rich.status import Status
from rich.panel import Panel
from rich.box import SQUARE
import json
import re

# Assuming LLM_PROVIDER and API_KEY are globally set in fractalic.py
# You can initialize LLMClient here if it's a singleton

def process_tool_calls(ast: AST, tool_messages: list) -> AST:
    """Process tool call responses and build Tool Loop AST"""
    tool_loop_ast = AST("")
    all_tool_content = []
    
    for message in tool_messages:
        if message.get('role') == 'tool':
            # Extract content from tool response
            content = message.get('content', '')
            print(f"[DEBUG] Processing tool message with content length: {len(content)}")
            
            # Try to parse as JSON to extract response fields
            try:
                # First try direct JSON parsing
                tool_response = json.loads(content)
                print(f"[DEBUG] Parsed tool response JSON with keys: {tool_response.keys()}")
                if isinstance(tool_response, dict):
                    # Look for common response fields that contain content
                    content_fields = ['return_content', 'content', 'result', 'response', 'output']
                    for field in content_fields:
                        if field in tool_response and tool_response[field]:
                            field_content = tool_response[field]
                            print(f"[DEBUG] Found content field '{field}' with length: {len(str(field_content))}")
                            if isinstance(field_content, str) and field_content.strip():
                                # Handle escaped newlines in JSON strings
                                if '\\n' in field_content:
                                    field_content = field_content.replace('\\n', '\n')
                                if '\\r' in field_content:
                                    field_content = field_content.replace('\\r', '\r')
                                if '\\t' in field_content:
                                    field_content = field_content.replace('\\t', '\t')
                                all_tool_content.append(field_content)
                                print(f"[DEBUG] Added content from field '{field}' to tool content list")
                            break
                    else:
                        # If no recognized content field, use the raw JSON
                        print(f"[DEBUG] No recognized content field found, using raw JSON")
                        all_tool_content.append(content)
                else:
                    print(f"[DEBUG] Tool response is not a dict, using as-is")
                    all_tool_content.append(content)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON decode error: {e}")
                # Try to extract the return_content field specifically for fractalic_run responses
                try:
                    # Look for return_content field with proper JSON string handling
                    # First find the return_content field start
                    import re
                    
                    # Find the start of return_content field
                    return_content_start = content.find('"return_content":')
                    if return_content_start != -1:
                        # Find the opening quote of the value
                        value_start = content.find('"', return_content_start + len('"return_content":'))
                        if value_start != -1:
                            # Track nested quotes and escapes to find the end of the string value
                            pos = value_start + 1
                            while pos < len(content):
                                char = content[pos]
                                if char == '\\':
                                    # Skip the next character (it's escaped)
                                    pos += 2
                                    continue
                                elif char == '"':
                                    # Found the closing quote
                                    field_content = content[value_start + 1:pos]
                                    # Unescape JSON string literals
                                    field_content = field_content.replace('\\"', '"').replace('\\\\', '\\')
                                    # Handle escaped newlines
                                    if '\\n' in field_content:
                                        field_content = field_content.replace('\\n', '\n')
                                    if '\\r' in field_content:
                                        field_content = field_content.replace('\\r', '\r')
                                    if '\\t' in field_content:
                                        field_content = field_content.replace('\\t', '\t')
                                    all_tool_content.append(field_content)
                                    print(f"[DEBUG] Extracted return_content with manual parsing, length: {len(field_content)}")
                                    break
                                pos += 1
                            else:
                                print(f"[DEBUG] Could not find closing quote for return_content")
                                all_tool_content.append(content)
                        else:
                            print(f"[DEBUG] Could not find return_content value start")
                            all_tool_content.append(content)
                    else:
                        print(f"[DEBUG] No return_content field found, using content as-is")
                        all_tool_content.append(content)
                except Exception as parse_error:
                    print(f"[DEBUG] Manual parsing failed: {parse_error}, using content as-is")
                    all_tool_content.append(content)
    
    # Combine all tool content and create AST
    if all_tool_content:
        combined_content = "\n\n".join(all_tool_content)
        tool_loop_ast = AST(combined_content)
        
        # Extract attribution metadata from tool responses
        all_return_nodes_attribution = []
        for message in tool_messages:
            if message.get('role') == 'tool':
                content = message.get('content', '')
                try:
                    tool_response = json.loads(content)
                    if isinstance(tool_response, dict) and 'return_nodes_attribution' in tool_response:
                        all_return_nodes_attribution.extend(tool_response['return_nodes_attribution'])
                except json.JSONDecodeError:
                    pass
        
        # Mark nodes as tool-generated context and apply attribution if available
        node_index = 0
        for node in tool_loop_ast.parser.nodes.values():
            node.role = "user"  # Use user role so content is treated as context, not tool responses
            node.is_tool_generated = True
            
            # Apply attribution metadata from fractalic_run returns if available
            if node_index < len(all_return_nodes_attribution):
                attribution = all_return_nodes_attribution[node_index]
                node.created_by = attribution.get('created_by')
                node.created_by_file = attribution.get('created_by_file')
                print(f"[DEBUG] Applied attribution to Tool Loop AST node: created_by={node.created_by}, created_by_file={node.created_by_file}")
            
            node_index += 1
    
    return tool_loop_ast

def insert_direct_context(ast: AST, tool_loop_ast: AST, current_node: Node):
    """Insert tool loop content with _IN_CONTEXT_BELOW_ markers and replace JSON with markdown"""
    if not tool_loop_ast.parser.nodes:
        return
    
    # Update current node's response
    if hasattr(current_node, 'response_content'):
        current_response = current_node.response_content or ""
        
        # Simple regex to find and replace JSON blocks containing return_content
        # This pattern matches the entire JSON block after "response:" until the next blank line or "> TOOL"
        import re
        
        # Note: JSON replacement now happens at the tool response generation level,
        # not here in post-processing. This is just a placeholder for any future
        # response content modifications if needed.
        
        # Add actual tool content at the end
        context_content = "\n\n> TOOL RESPONSE\ncontent: \"_IN_CONTEXT_BELOW_\"\n\n"
        for node in tool_loop_ast.parser.nodes.values():
            context_content += node.content + "\n\n"
        
        current_response += context_content
        current_node.response_content = current_response


def process_llm(ast: AST, current_node: Node, call_tree_node=None, committed_files=None, file_commit_hashes=None, base_dir=None) -> Optional[Node]:
    """Process @llm operation with updated schema support"""
    console = Console(force_terminal=True)

    # Extract system prompts first (always available)
    system_prompt = ast.get_system_prompts()

    def get_previous_headings(node: Node) -> str:
        context = []
        current = ast.first()
        while current and current != node:
            if current.type == NodeType.HEADING and not (hasattr(current, 'is_system') and current.is_system):
                context.append(current.content)
            current = current.next
        return "\n\n".join(context)
    
    
    def get_previous_heading_messages(node: Node) -> list:
        """Return a list of messages for each heading node encountered before the given node."""
        messages = []
        current = ast.first()
        
        # Get the enableOperationsVisibility setting from Config - fixed to look in runtime section
        enable_operations_visibility = Config.TOML_SETTINGS.get('runtime', {}).get('enableOperationsVisibility', False)
        # print(f"enableOperationsVisibility: {enable_operations_visibility}")

        while current and current != node:
            # Skip system blocks from context building  
            if hasattr(current, 'is_system') and current.is_system:
                current = current.next
                continue
                
            # If enableOperationsVisibility is True, include all nodes
            # Otherwise, only include HEADING nodes (original behavior)
            if enable_operations_visibility or current.type == NodeType.HEADING:
                # Use the node's role attribute, defaulting to "user" if not specified
                role = getattr(current, "role", "user")
                messages.append({"role": role, "content": current.content})
        
            current = current.next
        return messages
    

    # Get parameters
    params = current_node.params or {}
    prompt = params.get('prompt')
    block_params = params.get('block', {})

    # New optional field
    model = params.get('model')

    # Always infer provider by matching model field in settings
    found = False
    all_models = Config.TOML_SETTINGS.get('settings', {})
    provider = None
    if model:
        for key, conf in all_models.items():
            name = conf.get('model', key)
            if model == name or model == name.replace('.', '-') or model == name.replace('.', '_'):
                provider = key
                found = True
                break
        if not found:
            raise KeyError(f'Model "{model}" not found under [settings] in settings.toml')
    else:
        # fallback to default provider from config
        provider = Config.LLM_PROVIDER

    # Map the new stop-sequences parameter into stop_sequences for the LLM client
    stop_seqs = params.get('stop-sequences')
    if stop_seqs:
        params['stop_sequences'] = stop_seqs

    # Get tools-turns-max parameter and pass to LLM client if present
    tools_turns_max = params.get('tools-turns-max')
    if tools_turns_max is not None:
        params['tools-turns-max'] = tools_turns_max

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
        # Check if block_uri is an array (new enhanced functionality)
        block_uri = block_params.get('block_uri')
        if isinstance(block_uri, list):
            # Handle array of block URIs with wildcard support
            try:
                block_ast = get_ast_parts_by_uri_array(ast, block_uri, use_hierarchy=any(uri.endswith("/*") for uri in block_uri), tool_loop_ast=tool_loop_ast)
                if block_ast.parser.nodes:
                    # Keep existing prompt_parts logic
                    block_content = "\n\n".join(node.content for node in block_ast.parser.nodes.values())
                    prompt_parts.append(block_content)
                    
                    # Build messages - one message per node in the combined blocks
                    role = block_params.get('role', 'user')
                    for node in block_ast.parser.nodes.values():
                        messages.append({"role": role, "content": node.content})
            except BlockNotFoundError:
                raise ValueError(f"One or more blocks in array '{block_uri}' not found")
        elif block_params.get('is_multi'):
            # Handle legacy array of blocks format
            blocks = block_params.get('blocks', [])
            for block_info in blocks:
                try:
                    block_uri = block_info.get('block_uri')
                    nested_flag = block_info.get('nested_flag', False)
                    block_ast = get_ast_part_by_path(ast, block_uri, nested_flag, tool_loop_ast)
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
                block_ast = get_ast_part_by_path(ast, block_uri, nested_flag, tool_loop_ast)
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
        messages.append({"role": "user", "content": prompt})    # Combine all parts with proper spacing
    prompt_text = "\n\n".join(part.strip() for part in prompt_parts if part.strip())

    # Prepend system prompt to messages if messages exist
    if messages:
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Call LLM - use messages if available, otherwise fall back to prompt_text
    llm_provider = provider
    llm_model = model if model else Config.MODEL
    Config.LLM_PROVIDER = llm_provider
    Config.MODEL = llm_model
    provider_cfg = Config.TOML_SETTINGS.get('settings', {}).get(llm_provider, {})
    Config.API_KEY = provider_cfg.get('apiKey')
    llm_client = LLMClient(model=llm_model)
    
    # Initialize Tool Loop AST for this LLM operation
    tool_loop_ast = AST("")
    
    # Set execution context for tool registry if available
    if hasattr(llm_client.client, 'registry'):
        current_file = getattr(current_node, 'created_by_file', None)
        llm_client.client.registry.set_execution_context(
            ast=ast,
            current_file=current_file,
            call_tree_node=call_tree_node,
            committed_files=committed_files,
            file_commit_hashes=file_commit_hashes,
            base_dir=base_dir,
            tool_loop_ast=tool_loop_ast,  # Pass Tool Loop AST to registry
            current_node=current_node  # Pass current @llm operation node for attribution
        )
        
        # Pass Tool Loop AST to the LLM client for real-time updates
        if hasattr(llm_client.client, 'tool_loop_ast'):
            llm_client.client.tool_loop_ast = tool_loop_ast
    
    actual_model = model if model else getattr(llm_client.client, 'settings', {}).get('model', llm_model)    # Add system prompt to params for LLM clients that use it
    params['system_prompt'] = system_prompt

    # Print a simple header for the LLM call (streaming is now always enabled)
    console.print(f"[cyan]@llm ({llm_provider}/{actual_model}) streaming...[/cyan]")

    start_time = time.time()
    try:
        response = llm_client.llm_call(prompt_text, messages, params)
        # Always extract text for use, and only save messages for trace
        if isinstance(response, dict) and 'text' in response:
            response_text = response['text']
            current_node.response_content = response_text
            if 'messages' in response:
                current_node.response_messages = response['messages']
                
                # Process tool calls to build Tool Loop AST
                tool_messages = [msg for msg in response['messages'] if msg.get('role') == 'tool']
                if tool_messages:
                    # Update Tool Loop AST with tool responses
                    new_tool_ast = process_tool_calls(ast, tool_messages)
                    if new_tool_ast.parser.nodes:
                        # Merge with existing Tool Loop AST
                        if tool_loop_ast.parser.nodes:
                            # Combine existing and new tool content
                            combined_nodes = {}
                            combined_nodes.update(tool_loop_ast.parser.nodes)
                            combined_nodes.update(new_tool_ast.parser.nodes)
                            tool_loop_ast.parser.nodes = combined_nodes
                            
                            # Update head and tail
                            all_nodes = list(combined_nodes.values())
                            tool_loop_ast.parser.head = all_nodes[0] if all_nodes else None
                            tool_loop_ast.parser.tail = all_nodes[-1] if all_nodes else None
                        else:
                            tool_loop_ast = new_tool_ast
                        
                        # Update the tool registry with the new Tool Loop AST
                        if hasattr(llm_client.client, 'registry'):
                            llm_client.client.registry._tool_loop_ast = tool_loop_ast
                    
                    # Insert context integration markers
                    insert_direct_context(ast, tool_loop_ast, current_node)
        else:
            response_text = response
            current_node.response_content = response_text

        duration = time.time() - start_time
        mins, secs = divmod(int(duration), 60)
        duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        console.print(
            f"[light_green]✓[/light_green][cyan] @llm [turquoise2]({llm_provider}/{actual_model}"
            f"{('/' + llm_client.base_url) if hasattr(llm_client, 'base_url') and llm_client.base_url else ''})[/turquoise2]"
            f"[/cyan] completed ({duration_str})"
        )
        
    except Exception as e:
        # Check for LLMCallException with partial result
        partial = None
        if hasattr(e, 'partial_result') and getattr(e, 'partial_result'):
            partial = getattr(e, 'partial_result')
            console.print(f"[yellow]Partial LLM response before error:[/yellow]\n{partial}")
            current_node.response_content = f"PARTIAL RESPONSE BEFORE ERROR:\n{partial}\n\nERROR: {str(e)}"
        else:
            current_node.response_content = f"ERROR: {str(e)}"
        console.print(f"[bold red]✗ Failed: {str(e)}[/bold red]")
        console.print(f"[bold red]  Operation content:[/bold red]\n{current_node.content}")
        raise

    # Get save-to-file parameter
    save_to_file = params.get('save-to-file')

    # Save raw response to file if save_to_file is specified
    if save_to_file:
        file_path = Path(save_to_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(response_text)

    # Handle header
    header = ""
    use_header = params.get('use-header')
    if use_header is not None:
        if use_header.lower() != "none":
            header = f"{use_header}\n"
    else:
        header = "# LLM response block\n"

    response_ast = AST(f"{header}{response_text}\n")
    for node_key, node in response_ast.parser.nodes.items():
        node.role = "assistant"
        node.created_by = current_node.key  # Store the ID of the operation node that triggered this response
        node.created_by_file = current_node.created_by_file # set the file path
        

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