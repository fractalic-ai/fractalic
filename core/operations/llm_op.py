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
from rich.markup import escape
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
            # print(f"[DEBUG] Processing tool message with content length: {len(content)}")
            
            # Try to parse as JSON to extract response fields
            try:
                # First try direct JSON parsing
                tool_response = json.loads(content)
                # print(f"[DEBUG] Parsed tool response JSON with keys: {tool_response.keys()}")
                if isinstance(tool_response, dict):
                    # Look for common response fields that contain content
                    content_fields = ['return_content', 'content', 'result', 'response', 'output']
                    for field in content_fields:
                        if field in tool_response and tool_response[field]:
                            field_content = tool_response[field]
                            # print(f"[DEBUG] Found content field '{field}' with type: {type(field_content)}")
                            
                            # Handle MCP response format: content is an array of objects with type/text
                            if field == 'content' and isinstance(field_content, list):
                                # Extract text from MCP content array
                                text_parts = []
                                for item in field_content:
                                    if isinstance(item, dict):
                                        if item.get('type') == 'text' and 'text' in item:
                                            text_parts.append(str(item['text']))
                                        elif 'text' in item:  # fallback for any object with 'text'
                                            text_parts.append(str(item['text']))
                                if text_parts:
                                    combined_text = '\n'.join(text_parts)
                                    all_tool_content.append(combined_text)
                                    # print(f"[DEBUG] Extracted {len(text_parts)} text parts from MCP content array")
                                    break
                            
                            # Handle regular string content
                            elif isinstance(field_content, str) and field_content.strip():
                                # Handle escaped newlines in JSON strings
                                if '\\n' in field_content:
                                    field_content = field_content.replace('\\n', '\n')
                                if '\\r' in field_content:
                                    field_content = field_content.replace('\\r', '\r')
                                if '\\t' in field_content:
                                    field_content = field_content.replace('\\t', '\t')
                                all_tool_content.append(field_content)
                                # print(f"[DEBUG] Added string content from field '{field}' to tool content list")
                                break
                    else:
                        # If no recognized content field, use the raw JSON
                        # print(f"[DEBUG] No recognized content field found, using raw JSON")
                        all_tool_content.append(content)
                else:
                    # print(f"[DEBUG] Tool response is not a dict, using as-is")
                    all_tool_content.append(content)
            except json.JSONDecodeError as e:
                # print(f"[DEBUG] JSON decode error: {e}")
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
                                    # print(f"[DEBUG] Extracted return_content with manual parsing, length: {len(field_content)}")
                                    break
                                pos += 1
                            else:
                                # print(f"[DEBUG] Could not find closing quote for return_content")
                                all_tool_content.append(content)
                        else:
                            # print(f"[DEBUG] Could not find return_content value start")
                            all_tool_content.append(content)
                    else:
                        # print(f"[DEBUG] No return_content field found, using content as-is")
                        all_tool_content.append(content)
                except Exception as parse_error:
                    # CRITICAL: Log manual parsing failures that could stop tool processing
                    print(f"[ERROR] Tool parsing failed completely: {parse_error}")
                    all_tool_content.append(content)
    
    # Extract attribution metadata from tool responses first
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
    
    # Combine all tool content and create AST with preserved attribution
    if all_tool_content:
        combined_content = "\n\n".join(all_tool_content)
        
        # Use specialized AST creation that preserves keys and attribution
        tool_loop_ast = AST.create_with_attribution(combined_content, all_return_nodes_attribution)
        
        # Mark nodes as tool-generated context
        for node in tool_loop_ast.parser.nodes.values():
            node.role = "user"  # Use user role so content is treated as context, not tool responses
            node.is_tool_generated = True
            # print(f"[DEBUG] Tool Loop AST node with preserved attribution: key={node.key}, id={node.id}, created_by={node.created_by}, created_by_file={node.created_by_file}")
    else:
        # CRITICAL: Log when no tool content is found - this could indicate silent tool failure
        print(f"[ERROR] No tool content extracted from {len(tool_messages)} tool messages - tool processing may have failed silently")
    
    return tool_loop_ast

def insert_direct_context(ast: AST, tool_loop_ast: AST, current_node: Node):
    """Insert tool loop AST nodes directly into main AST while preserving keys and identity"""
    if not tool_loop_ast.parser.nodes:
        return
    
    # Instead of string concatenation, directly merge Tool Loop AST nodes into main AST
    # This preserves the original keys and node identity from cross-agent execution
    
    # Find the insertion point (after current_node)
    insertion_point = current_node
    
    # Insert Tool Loop AST nodes directly into the main AST
    for tool_node in tool_loop_ast.parser.nodes.values():
        # Create a copy of the tool node to avoid reference issues
        import copy
        new_node = copy.deepcopy(tool_node)
        
        # Preserve the original key and identity from Tool Loop AST
        new_node.key = tool_node.key
        new_node.created_by = tool_node.created_by
        new_node.created_by_file = tool_node.created_by_file
        new_node.is_tool_generated = True
        new_node.role = "user"  # Mark as user content for context
        
        # Insert the node into the main AST after the current node
        new_node.prev = insertion_point
        new_node.next = insertion_point.next
        
        if insertion_point.next:
            insertion_point.next.prev = new_node
        else:
            # This is the new tail
            ast.parser.tail = new_node
            
        insertion_point.next = new_node
        
        # Add to the main AST nodes dictionary with original key
        ast.parser.nodes[new_node.key] = new_node
        
        # Move insertion point for next node
        insertion_point = new_node
        
        # print(f"[DEBUG] Direct AST merge: inserted node {new_node.key} (id: {new_node.id}) with preserved identity")
    
    # Update current node's response to include reference markers
    if hasattr(current_node, 'response_content'):
        current_response = current_node.response_content or ""
        context_content = "\n\n> TOOL RESPONSE\ncontent: \"_IN_CONTEXT_BELOW_\"\n\n"
        current_response += context_content
        current_node.response_content = current_response


def process_llm(ast: AST, current_node: Node, call_tree_node=None, committed_files=None, file_commit_hashes=None, base_dir=None) -> Optional[Node]:
    """Process @llm operation with updated schema support"""
    console = Console(force_terminal=True)
    
    # Extract system prompts first (always available)
    system_prompt = ast.get_system_prompts()

    def get_current_header_level(node: Node) -> int:
        """Get the level of the last heading before the given node."""
        current = ast.first()
        last_header_level = 0  # Default to root level if no headers found
        
        while current and current != node:
            if current.type == NodeType.HEADING and not (hasattr(current, 'is_system') and current.is_system):
                last_header_level = current.level
            current = current.next        
        return last_header_level

    def adjust_header_levels(text: str, base_level: int) -> str:
        """Adjust header levels in text so minimum header becomes base_level + 1."""
        if base_level <= 0:
            return text
            
        lines = text.split('\n')
        
        # First pass: find the minimum header level in the text
        min_header_level = None
        for line in lines:
            if line.strip().startswith('#'):
                header_match = re.match(r'^(#+)\s*(.*)', line.strip())
                if header_match:
                    current_level = len(header_match.group(1))
                    if min_header_level is None or current_level < min_header_level:
                        min_header_level = current_level
        
        # If no headers found, return original text
        if min_header_level is None:
            return text
        
        # Calculate the adjustment needed
        # We want min_header_level to become base_level + 1
        target_min_level = base_level + 1
        level_adjustment = target_min_level - min_header_level
        
        # Second pass: adjust all headers
        adjusted_lines = []
        for line in lines:
            if line.strip().startswith('#'):
                header_match = re.match(r'^(#+)\s*(.*)', line.strip())
                if header_match:
                    current_hashes = header_match.group(1)
                    content = header_match.group(2)
                    current_level = len(current_hashes)
                    
                    # Apply the level adjustment
                    new_level = current_level + level_adjustment
                    # Ensure we don't go below level 1
                    new_level = max(1, new_level)
                    new_hashes = '#' * new_level
                    adjusted_lines.append(f"{new_hashes} {content}")
                else:
                    adjusted_lines.append(line)
            else:
                adjusted_lines.append(line)
        
        return '\n'.join(adjusted_lines)

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
    context_mode = params.get('context', 'auto')  # new context control

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

    # Check if tools are being used for this operation
    tools_param = params.get('tools', 'none')
    using_tools = tools_param != 'none'
    
    # Initialize Tool Loop AST only if tools are being used
    tool_loop_ast = AST("") if using_tools else None

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
        # Only add implicit preceding context when context_mode != 'none'
        if context_mode != 'none':
            context = get_previous_headings(current_node)
            if context:
                prompt_parts.append(context)
            
            # Add heading messages  
            heading_messages = get_previous_heading_messages(current_node)
            messages.extend(heading_messages)
        # else: skip adding any preceding context

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
    
    # Get the correct model - if no explicit model specified, use the provider's default model
    if model:
        llm_model = model
    else:
        # Get the default model from the provider's settings
        provider_cfg = Config.TOML_SETTINGS.get('settings', {}).get(llm_provider, {})
        llm_model = provider_cfg.get('model', llm_provider)  # fallback to provider name as model
    
    # Don't modify global Config values - use local variables only
    provider_cfg = Config.TOML_SETTINGS.get('settings', {}).get(llm_provider, {})
    local_api_key = provider_cfg.get('apiKey')
    
    # Temporarily set the API key for this operation only
    original_api_key = Config.API_KEY
    Config.API_KEY = local_api_key
    
    llm_client = LLMClient(model=llm_model, provider=llm_provider)
    
    # Set execution context for tool registry if available
    if hasattr(llm_client.client, 'registry'):
        current_file = getattr(current_node, 'created_by_file', None)
        try:
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
        except Exception as registry_error:
            # CRITICAL: Registry context setting failure could break tool execution
            print(f"[ERROR] Failed to set tool registry context: {registry_error}")
            # Continue execution but tools may not work properly
        
        # Pass Tool Loop AST to the LLM client for real-time updates (only if using tools)
        if hasattr(llm_client.client, 'tool_loop_ast') and tool_loop_ast is not None:
            llm_client.client.tool_loop_ast = tool_loop_ast
    
    actual_model = model if model else getattr(llm_client.client, 'settings', {}).get('model', llm_model)    # Add system prompt to params for LLM clients that use it
    params['system_prompt'] = system_prompt

    # Print a simple header for the LLM call (streaming is now always enabled)
    console.print(f"[cyan]@llm ({llm_provider}/{actual_model}) streaming...[/cyan]")

    start_time = time.time()
    
    # Preserve context for error recovery
    operation_context = {
        'prompt_text': prompt_text,
        'messages': messages.copy() if messages else None,
        'params': params.copy(),
        'model': actual_model,
        'provider': llm_provider,
        'operation_id': f"llm_{current_node.id}",
        'source_file': getattr(ast, 'source_file', None) or getattr(ast, 'filename', None) or 'unknown',
        'start_time': start_time
    }
    
    # Add source file to params so LLM client can use it for token tracking
    params['_source_file'] = operation_context['source_file']
    
    try:
        response = llm_client.llm_call(prompt_text, messages, params)
        
        # Log token usage if available
        if isinstance(response, dict) and 'usage' in response and response['usage']:
            # Get source file context for tracking
            source_file = getattr(ast, 'source_file', None) or getattr(ast, 'filename', None) or 'unknown'
            
            # Store usage data in node for trace files
            current_node.token_usage = {
                'usage': response['usage'],
                'model': actual_model,
                'operation_id': f"llm_{current_node.id}",
                'operation_type': "llm_call",
                'source_file': source_file,
                'timestamp': time.time()
            }
            
            # Token tracking is handled by the OpenAI client in real-time
            # No need to log again here to avoid double-counting
        
        # Always extract text for use, and only save messages for trace
        if isinstance(response, dict) and 'text' in response:
            response_text = response['text']
            current_node.response_content = response_text
            if 'messages' in response:
                current_node.response_messages = response['messages']
                
                # Process tool calls to build Tool Loop AST (only if using tools)
                tool_messages = [msg for msg in response['messages'] if msg.get('role') == 'tool']
                if tool_messages and tool_loop_ast is not None:
                    # Update Tool Loop AST with tool responses
                    new_tool_ast = process_tool_calls(ast, tool_messages)
                    if new_tool_ast.parser.nodes:
                        # Merge with existing Tool Loop AST, avoiding duplicates by key
                        if tool_loop_ast.parser.nodes:
                            # Check for duplicate keys and avoid adding them
                            for key, new_node in new_tool_ast.parser.nodes.items():
                                if key not in tool_loop_ast.parser.nodes:
                                    tool_loop_ast.parser.nodes[key] = new_node
                                    # print(f"[DEBUG] Added new Tool Loop AST node: {key} (id: {new_node.id})")
                                else:
                                    # print(f"[DEBUG] Skipped duplicate Tool Loop AST node: {key} (id: {new_node.id})")
                                    pass
                            
                            # Update head and tail based on document order
                            all_nodes = list(tool_loop_ast.parser.nodes.values())
                            if all_nodes:
                                # Sort by creation order or maintain linked list order
                                tool_loop_ast.parser.head = all_nodes[0]
                                tool_loop_ast.parser.tail = all_nodes[-1]
                                
                                # Rebuild linked list to maintain proper order
                                prev_node = None
                                for node in all_nodes:
                                    node.prev = prev_node
                                    node.next = None
                                    if prev_node:
                                        prev_node.next = node
                                    prev_node = node
                        else:
                            tool_loop_ast = new_tool_ast
                        
                        # Update the tool registry with the new Tool Loop AST
                        if hasattr(llm_client.client, 'registry'):
                            llm_client.client.registry._tool_loop_ast = tool_loop_ast
                    else:
                        # Tool messages exist but no AST nodes created - check if this is expected
                        # This can happen when tool responses don't contain extractable content or in error scenarios
                        # print(f"[DEBUG] Tool messages processed but no Tool Loop AST nodes created - responses may be empty or error responses")
                        pass
                    
                    # Insert context integration markers
                    insert_direct_context(ast, tool_loop_ast, current_node)
                elif tool_messages and tool_loop_ast is None:
                    # CRITICAL: Tool messages found but Tool Loop AST is None - configuration error
                    print(f"[ERROR] Tool messages found but Tool Loop AST is None - tools may not be properly configured")
                elif not tool_messages and using_tools:
                    # Check if there were any tool calls in the response - only error if tools were attempted but failed
                    assistant_messages = [msg for msg in response['messages'] if msg.get('role') == 'assistant']
                    has_tool_calls = any(msg.get('tool_calls') for msg in assistant_messages)
                    if has_tool_calls:
                        # Tools were called but no tool messages returned - this is an error
                        print(f"[ERROR] Tools called but no tool messages found in response - tool execution may have failed silently")
                    # If no tool calls were made, this is normal completion - no error needed
        else:
            response_text = response
            current_node.response_content = response_text

        duration = time.time() - start_time
        mins, secs = divmod(int(duration), 60)
        duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        # Get the usage text from the token tracker if available
        usage_text = ""
        # Token stats usage text removed        
        console.print(
            f"[light_green]✓[/light_green][green] @llm [turquoise2]({llm_provider}/{actual_model}"
            f"{('/' + llm_client.base_url) if hasattr(llm_client, 'base_url') and llm_client.base_url else ''})[/turquoise2]"
            f" completed ({duration_str})[/green]{usage_text}"
        )
        
    except Exception as e:
        # Restore original API key on error
        Config.API_KEY = original_api_key
        
        # Get source file context for error trace
        source_file = operation_context['source_file']
        
        # Create comprehensive error trace with context preservation
        error_trace = {
            'error': str(e),
            'operation_id': operation_context['operation_id'],
            'model': operation_context['model'],
            'provider': operation_context['provider'],
            'operation_type': "llm_call",
            'source_file': source_file,
            'timestamp': time.time(),
            'duration': time.time() - operation_context['start_time'],
            'context_preserved': True,
            'operation_context': {
                'prompt_length': len(operation_context['prompt_text']) if operation_context['prompt_text'] else 0,
                'messages_count': len(operation_context['messages']) if operation_context['messages'] else 0,
                'params_used': {k: v for k, v in operation_context['params'].items() if k not in ['api_key']},  # Exclude sensitive data
                'messages_preview': operation_context['messages'][-2:] if operation_context['messages'] else None  # Last 2 messages for context
            }
        }
        
        # Check for LLMCallException with partial result
        partial = None
        if hasattr(e, 'partial_result') and getattr(e, 'partial_result'):
            partial = getattr(e, 'partial_result')
            console.print(f"[yellow]Partial LLM response before error:[/yellow]\n{escape(partial)}")
            current_node.response_content = f"PARTIAL RESPONSE BEFORE ERROR:\n{partial}\n\nERROR: {str(e)}"
            error_trace['partial_response'] = partial
        else:
            current_node.response_content = f"ERROR: {str(e)}"
        
        # Try to get partial token usage data if available in exception
        if hasattr(e, 'partial_usage') and getattr(e, 'partial_usage'):
            error_trace['partial_usage'] = getattr(e, 'partial_usage')
            
            # TokenStats logging removed
            pass
        
        # Store error trace in node for trace files
        current_node.token_usage = error_trace
        
        # Use console.print without Rich markup to avoid conflicts
        console.print("✗ Failed:", escape(str(e)), style="bold red")
        console.print("  Operation content:", style="bold red")
        console.print(escape(current_node.content))
        raise
    
    # Restore original API key after successful operation
    Config.API_KEY = original_api_key

    # Get save-to-file parameter
    save_to_file = params.get('save-to-file')

    # Save raw response to file if save_to_file is specified
    if save_to_file:
        file_path = Path(save_to_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(response_text)

    # Always store response content for context file generation
    current_node.response_content = response_text
      # Handle response AST creation and insertion
    # Skip AST operation if direct context was already inserted to prevent duplicate nodes
    skip_ast_operation = bool(tool_loop_ast and tool_loop_ast.parser.nodes)
    
    if not skip_ast_operation:        # Check if header-auto-align is enabled
        header_auto_align = params.get('header-auto-align', False)
        processed_response_text = response_text
        
        if header_auto_align:
            current_header_level = get_current_header_level(current_node)
            processed_response_text = adjust_header_levels(response_text, current_header_level)
        
        # Handle header (don't auto-align the use-header itself)
        header = ""
        use_header = params.get('use-header')
        if use_header is not None:
            if use_header.lower() != "none":
                header = f"{use_header}\n"
        else:
            # Apply header-auto-align to default header if enabled
            if header_auto_align:
                current_header_level = get_current_header_level(current_node)
                adjusted_default_header = adjust_header_levels("# LLM response block", current_header_level)
                header = f"{adjusted_default_header}\n"
            else:
                header = "# LLM response block\n"

        response_ast = AST(f"{header}{processed_response_text}\n")
        for node_key, node in response_ast.parser.nodes.items():
            node.role = "assistant"
            node.created_by = current_node.key  # Store the ID of the operation node that triggered this response
            node.created_by_file = current_node.created_by_file # set the file path        # Handle target block insertion
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
    else:
        # print(f"[DEBUG] Skipping AST operation - direct context already inserted {len(tool_loop_ast.parser.nodes) if tool_loop_ast else 0} nodes")
        # When skipping AST operation, append response content to current node to preserve tool calls in context file
        
        # Check if header-auto-align is enabled
        header_auto_align = params.get('header-auto-align', False)
        processed_response_text = response_text        
        if header_auto_align:
            current_header_level = get_current_header_level(current_node)
            processed_response_text = adjust_header_levels(response_text, current_header_level)
        
        use_header = params.get('use-header')
        if use_header is not None and use_header.lower() != "none":
            header = f"\n{use_header}\n"
        else:
            # Apply header-auto-align to default header if enabled
            if header_auto_align:
                current_header_level = get_current_header_level(current_node)
                adjusted_default_header = adjust_header_levels("# Complete Workflow Results", current_header_level)
                header = f"\n{adjusted_default_header}\n"
            else:
                header = "\n# Complete Workflow Results\n"
        
        # Append response content to current node's content to ensure it appears in context file
        current_node.content += f"{header}{processed_response_text}\n"
        # print(f"[DEBUG] Appended response content to current node to preserve tool calls in context file")

    return current_node.next