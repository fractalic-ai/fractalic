# Parser
# - Parser
# - print_parsed_structure

from ast import AST
import re
from typing import Dict, Optional, Union
from core.ast_md.node import Node, NodeType
# from core.ast_md.operation_parser import OperationParser
import unicodedata

import re
import yaml
import jsonschema
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# YAML schema (complete schema provided)
schema_text = r'''
operations:
  import:
    description: "Import content from another file or block"
    type: object
    required: ["file"]
    properties:
      file:
        type: string
        x-process: path
        description: "Source path: folder/file.md or folder/file.ctx"
      block:
        type: string
        x-process: block-path
        description: "Source block path: block/subblock/* where /* is optional for nested blocks"    
      mode:
        type: string
        enum: ["append", "prepend", "replace"]
        default: "append"
        description: "How to insert content: append, prepend, replace"
      to:
        type: string
        x-process: block-path
        description: "Target block path where content will be placed, supports nested flag"
      run-once:
        type: boolean
        default: false
        description: "Whether this operation should only run once."

  llm:
    description: "Execute LLM model with prompt and handle response"
    type: object
    properties:
      prompt:
        type: string
        description: |
          Literal text in quotes or multiline or block reference.
      block:
        oneOf:
        - type: string
        - type: array
          items:
            type: string
        x-process: block-path
        description: |
          Block reference to use as prompt. 
          Can be used as collection of blocks. 
          At least one of `prompt` or `block` should be provided. 
          If both are provided - they are stacked together in order: [specified blocks]+prompt. 
          If only prompt, it stacked as [all previous blocks]+prompt.
          if only block, or blocks, they are stacked together and no other context is added.
      media:
        type: array
        items:
            type: string
            x-process: file-path
            description: "Path to media file to add with context or prompt: folder/image.png"
      save-to-file:
        type: string
        xProcess: file-path
        description: "Path to file where response will be saved, overwrites existing file. ! Important: header wouldn't be saved"
      use-header:
        type: string
        description: "if provided - header for the block that will contain LLM response. If contains {id=X}, creates block with that ID. Use 'none' to omit header completely (case-insensitive)"
      header-auto-align:
        type: boolean
        default: false
        description: "if true, automatically adjusts header levels in LLM response based on current context header level. All headers in response will be at least one level deeper than the current header context."
      mode:
        type: string
        enum: ["append", "prepend", "replace"]
        default: "append"
        description: "How to insert LLM response into target block"
      to:
        type: string
        x-process: block-path
        description: "Target block where LLM response will be placed"
      provider:        
        type: string
        description: "Optional LLM provider to override the default setting."
      model:          
        type: string
        description: "Optional model to override the default setting."
      temperature:
        type: number
        minimum: 0
        maximum: 1
        description: "Optional temperature setting for LLM call to control randomness"
      run-once:
        type: boolean
        default: false
        description: "Whether this operation should only run once."
      stop-sequences:
        type: array
        items:
          type: string
        description: "List of strings where the model should stop generation (for Anthropic models it maps to stop_sequences parameter)."
      tools:
        anyOf:
          - type: string
            enum: ["none", "all"]
          - type: string
            description: "Single tool name"
          - type: array
            items:
              type: string
        default: "none"
        description: "Specify which tools to use: 'none' for no tools (default), 'all' for all tools, an array of specific tool names, or 'mcp/server-name' patterns to include all tools from specific MCP servers"
      tools-turns-max:
        type: integer
        description: "Maximum number of tool calls allowed for this @llm operation. If set, overrides the default or global tool call limit for this operation only."
      context:
        type: string
        enum: ["auto", "none"]
        default: "auto"
        description: "Context assembly mode: auto (default) includes preceding blocks when only prompt is provided; none disables implicit preceding-context injection."
    anyOf:
      - required: ["prompt"]
      - required: ["block"]

  run:
    description: "Execute another markdown file as a workflow"
    type: object
    required: ["file"]
    properties:
      file:
        type: string
        x-process: file-path
        description: "Path to markdown file to execute: folder/file.md"
      prompt:
        type: string
        description: "Optional input text or block reference to pass to the executed file"
      block:
        oneOf:
        - type: string
        - type: array
          items:
            type: string
        x-process: block-path
        description: |
          Block reference to use as prompt. 
          Can be used as collection of blocks. 
          At least one of `prompt` or `block` should be provided. 
          If both are provided - they are stacked together in order: [specified blocks]+prompt. 
          If only prompt, it stacked as [all previous blocks]+prompt.
          if only block, or blocks, they are stacked together and no other context is added.
      use-header:
        type: string
        description: "if provided - with prompt, header would be appended with propmpt content to target file before execution"
      mode:
        type: string
        enum: ["append", "prepend", "replace"]
        default: "append"
        description: "How to insert execution results"
      to:
        type: string
        x-process: block-path
        description: "Target block where execution results will be placed"
      run-once:
        type: boolean
        default: false
        description: "Whether this operation should only run once."


  shell:
    description: "Execute shell command and capture output"
    type: object
    required: ["prompt"]
    properties:
      prompt:
        type: string
        description: "Shell command to execute (single line or multiline)"
      use-header:
        type: string
        default: "# OS Shell Tool response block"
        description: "Optional header for the block that will contain command output and replace default header. Use 'none' to omit header completely (case-insensitive)"
      mode:
        type: string
        enum: ["append", "prepend", "replace"]
        default: "append"
        description: "How to insert command output"
      to:
        type: string
        x-process: block-path
        description: "Target block where command output will be placed"
      run-once:
        type: boolean
        default: false
        description: "Whether this operation should only run once."


  return:
    description: "Return content and terminate execution"
    type: object
    properties:
      prompt:
        type: string
        description: "Literal text to return"
      block:
        oneOf:
        - type: string
        - type: array
          items:
            type: string
        x-process: block-path
        description: |
          Block reference to use as prompt. 
          Can be used as collection of blocks. 
          At least one of `prompt` or `block` should be provided. 
          If both are provided - they are stacked together in order: [specified blocks]+prompt. 
          If only prompt, it stacked as [all previous blocks]+prompt.
          if only block, or blocks, they are stacked together and no other context is added.
      use-header:
        type: string
        description: "Optional header for returned prompt, overwrites default. Use 'none' to omit header completely (case-insensitive)"
    anyOf:
      - required: ["prompt"]
      - required: ["block"]

  goto:
    description: "Navigate to another block in document"
    type: object
    required: ["block"]
    properties:
      block:
        type: string
        x-process: block-path-no-nested
        description: "Target block to navigate to (no nested flags allowed)"
      run-once:
        type: boolean
        default: false
        description: "Whether this operation should only run once."

 
processors:
  path:
    description: "Process full path with file and blocks"
    generates:
      - path: "Folder path component"
      - file: "File name component"
      - block_uri: "Block path component"
      - nested_flag: "Whether to include nested blocks"

  block-path:
    description: "Process block reference path"
    generates:
      - block_uri: "Block path component"
      - nested_flag: "Whether to include nested blocks"

  prompt-or-block:
    description: "Process prompt content"
    generates:
      - type: "literal or block"
      - content: "Actual content if literal"
      - block_uri: "Block path if reference"
      - nested_flag: "Whether to include nested blocks"

  file-path:
    description: "Process file path"
    generates:
      - path: "Folder path component"
      - file: "File name component"

  return-content:
    description: "Process return content"
    generates:
      - type: "literal or block"
      - content: "Content if literal"
      - block_uri: "Block path if reference"
      - nested_flag: "Whether to include nested blocks"

  block-path-no-nested:
    description: "Process block path without nested flag support"
    generates:
      - block_uri: "Block path component"



formats:
  block_id:
    pattern: '^[a-zA-Z][a-zA-Z0-9-_]*$'
    description: "Valid block ID format"
    examples:
      - "BlockId"
      - "test-block-2"
      - "Results_Section"

  block_path:
    pattern: '^[a-zA-Z0-9-_/]+(?:/\*)?$'
    description: "Valid block path format"
    examples:
      - "parent/child"
      - "section1/subsection/block/*"

  file_path:
    pattern: '^(?:[a-zA-Z0-9-_/]+/)?[a-zA-Z0-9-_]+\.(md|ctx)$'
    description: "Valid file path format"
    examples:
      - "file.md"
      - "file.ctx"
      - "folder/subfolder/file.md"
      - "folder/subfolder/file.ctx"

computed_fields:
  path_processor:
    input: "folder1/folder2/file.md/block1/block2/*"
    generates:
      from-path: "folder1/folder2"
      from-file: "file.md"
      from-block-URI: "block1/block2"
      from-nested-flag: true
  
  path_processor_ctx:
    input: "folder1/folder2/file.ctx/block1/block2/*"
    generates:
      from-path: "folder1/folder2"
      from-file: "file.ctx"
      from-block-URI: "block1/block2"
      from-nested-flag: true

  prompt_processor:
    input_literal: |
      Multiline
      prompt text
    generates:
      prompt-type: "literal"
      prompt-content: "Multiline\nprompt text"

    input_block: "templates/prompt/*"
    generates:
      prompt-type: "block"
      prompt-block-URI: "templates/prompt"
      prompt-nested-flag: true
'''

@dataclass
class HeadingBlock:
    level: int
    title: str
    id: Optional[str]
    content: str
    role: str = "user"  # Added field with default role "user"
    is_system: bool = False  # Track if this is a system prompt block
@dataclass
class OperationBlock:
    operation: str
    params: Dict[str, Any]
    content: str
    role: str = "user" #assistant"  # Added field with default role "assistant"

@dataclass
class SchemaProcessor:
    operations_schema: Dict[str, Any]
    processors: Dict[str, Any]
    settings: Dict[str, Any]
    formats: Dict[str, Any]
    computed_fields: Dict[str, Any]
    special_cases: Dict[str, Any]
    error_handling: Dict[str, Any]
    extension_points: Dict[str, Any]

    def validate_operation(self, operation_block: OperationBlock):
        console = Console()
        operation_name = operation_block.operation
        if operation_name not in self.operations_schema:
            raise ValueError(f"Unknown operation '{operation_name}'")

        schema = self.operations_schema[operation_name]

        # Remove the first line from operation_block.content
        operation_block_content_no_op = '\n'.join(operation_block.content.split('\n')[1:])

        try:
            params = yaml.safe_load(operation_block_content_no_op)
            if params is None:
                params = {}
        except yaml.YAMLError as e:
            # Display operation content on YAML parsing error
            console.print(f"\n[bold red]✗ YAML Parsing Error in operation '{operation_name}':[/bold red]")
            console.print(
                Syntax(
                    operation_block.content.strip(),
                    "yaml",
                    line_numbers=True,
                    theme="monokai",
                    word_wrap=True,
                    background_color="default"
                )
            )
            raise ValueError(f"YAML parsing error in operation '{operation_name}': {str(e)}")

        # Validate against schema before processing
        try:
            jsonschema.validate(instance=params, schema=schema)
        except jsonschema.ValidationError as e:
            # Display operation content on validation error
            console.print(f"\n[bold red]✗ Validation Error in operation '{operation_name}':[/bold red]")
            console.print(
                Syntax(
                    operation_block.content.strip(),
                    "yaml", 
                    line_numbers=True,
                    theme="monokai",
                    word_wrap=True,
                    background_color="default"
                )
            )
            raise ValueError(f"Validation error in operation '{operation_name}': {str(e)}")

        # Apply field processors
        params = self.apply_processors(params, schema)

        operation_block.params = params

    def apply_processors(self, params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        properties = schema.get('properties', {})
        for field_name, field_info in properties.items():
            if 'x-process' in field_info and field_name in params:
                processor_name = field_info['x-process']
                value = params.get(field_name)
                processor_func = getattr(self, f"process_{processor_name.replace('-', '_')}", None)
                if processor_func:
                    params[field_name] = processor_func(value, field_name, params)
                else:
                    raise NotImplementedError(f"Processor '{processor_name}' is not implemented")
        return params


    # Implement field processors
    def process_path(self, value: str, field_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        path_parts = value.split('/')
        nested_flag = False
        if path_parts[-1] == '*':
            nested_flag = True
            path_parts = path_parts[:-1]

        file_idx = None
        for idx, part in enumerate(path_parts):
            if part.endswith('.md') or part.endswith('.ctx'):
                file_idx = idx
                break

        if file_idx is not None:
            result['path'] = '/'.join(path_parts[:file_idx])
            result['file'] = path_parts[file_idx]
            block_uri = '/'.join(path_parts[file_idx+1:])
        else:
            result['path'] = ''
            result['file'] = ''
            block_uri = '/'.join(path_parts)

        result['block_uri'] = block_uri
        result['nested_flag'] = nested_flag
        return result

    def process_block_path(self, value: str, field_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        #print("[Parser.process_block_path] Processing block path: ", value)
        result = {}

        # Handle list of blocks
        if isinstance(value, list):
            blocks = []
            for block_path in value:
                block_info = self._process_single_block_path(block_path)
                blocks.append(block_info)
            result['blocks'] = blocks
            result['is_multi'] = True
        else:
            # Handle single block path (maintain backward compatibility)
            block_info = self._process_single_block_path(value)
            result.update(block_info)
            result['is_multi'] = False

        return result

    def _process_single_block_path(self, path: str) -> Dict[str, Any]:
        """Helper to process individual block path"""
        path_parts = path.split('/')
        nested_flag = False
        
        if path_parts[-1] == '*':
            nested_flag = True
            path_parts = path_parts[:-1]

        return {
            'block_uri': '/'.join(path_parts),
            'nested_flag': nested_flag
        }

    def process_prompt_or_block(self, value: str, field_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        stripped_value = value.strip()
        if re.match(r'^[a-zA-Z0-9\-_]+(?:/[a-zA-Z0-9\-_]+)*(?:/\*)?$', stripped_value) and '\n' not in stripped_value:
            result['type'] = 'block'
            processor_result = self.process_block_path(stripped_value, field_name, params)
            result.update(processor_result)
        else:
            result['type'] = 'literal'
            result['content'] = value
        return result

    def process_file_path(self, value: str, field_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        path_parts = value.split('/')
        result['file'] = path_parts[-1]
        result['path'] = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else ''
        return result

    def process_return_content(self, value: str, field_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.process_prompt_or_block(value, field_name, params)

    def process_block_path_no_nested(self, value: str, field_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if '*' in value:
            raise ValueError("Nested flags are not allowed in this context")
        return {'block_uri': value}


def parse_document(text: str, schema_text: str) -> List[Any]:
    # Run linter first to catch structural and YAML issues
    try:
        from core.linter import FractalicLinter, FractalicLintError
        linter = FractalicLinter(schema_text)
        errors = linter.lint_content(text)
        
        if linter.has_errors():
            linter.print_errors()
            error_count = len([e for e in errors if e.severity == "error"])
            raise FractalicLintError(f"Found {error_count} linting errors - aborting parsing")
    except ImportError:
        # Fallback if linter is not available
        pass
    
    schema = yaml.safe_load(schema_text)
    operations_schema = schema.get('operations', {})
    processors = schema.get('processors', {})
    settings = schema.get('settings', {})
    formats = schema.get('formats', {})
    computed_fields = schema.get('computed_fields', {})
    special_cases = schema.get('special_cases', {})
    error_handling = schema.get('error_handling', {})
    extension_points = schema.get('extension_points', {})

    schema_processor = SchemaProcessor(
        operations_schema=operations_schema,
        processors=processors,
        settings=settings.get('properties', {}),
        formats=formats,
        computed_fields=computed_fields,
        special_cases=special_cases,
        error_handling=error_handling,
        extension_points=extension_points
    )

    lines = text.splitlines()
    blocks = []
    parsing_state = 'normal'
    current_block = None
    previous_line = None
    
    # Track system block hierarchy - if parent is system, children are too
    system_block_levels = []  # Stack to track system block levels

    for idx, l in enumerate(lines):
        line = l.rstrip('\n')

        # Heading Block Detection
        if re.match(r'^#+ ', line) and parsing_state != 'operation_block' and (previous_line is None or previous_line.strip() == ''):
            parsing_state = 'heading_block'
            level = len(line) - len(line.lstrip('#'))
            heading_line = line
            m = re.match(r'^(.*?)\s*(\{id=([a-zA-Z][a-zA-Z0-9\-_]*)\})?$', heading_line)
            if m:
                title = heading_line
                id_value = m.group(3)
            else:
                title = heading_line
                id_value = None
            
            # Check if this is a system block
            # Extract heading text without # symbols and trim
            heading_text = re.sub(r'^#+\s*', '', heading_line).strip()
            # Remove {id=...} pattern if present for system detection
            heading_text_clean = re.sub(r'\s*\{id=[^}]+\}$', '', heading_text)
            is_system_block = heading_text_clean.startswith('.system')
            
            # Update system block hierarchy tracking
            # Remove levels deeper than current level
            system_block_levels = [l for l in system_block_levels if l < level]
            
            # If current block is system or parent was system, mark as system
            if is_system_block or system_block_levels:
                is_system_block = True
                system_block_levels.append(level)
            
            current_block = HeadingBlock(
                level=level,
                title=title,
                id=id_value,
                content=heading_line+'\n',
                is_system=is_system_block
            )
            blocks.append(current_block)
            parsing_state = 'heading_block'
            previous_line = l
            continue

        # Operation Block Detection
        if re.match(r'^@[a-zA-Z]+', line) and (previous_line is None or previous_line.strip() == ''):
            parsing_state = 'operation_block'
            operation = line.strip('@').strip()
            
            # Clear system block hierarchy when operation is encountered
            # Operations break the hierarchical inheritance of system blocks
            system_block_levels = []
            
            current_block = OperationBlock(
              operation=operation,
              params={},
              content=line+'\n'  # Remove '@' and operation name (first line actually) would be done in schema_processor.validate_operation(block)
            )
            blocks.append(current_block)
            previous_line = l
            continue

        # Content Addition
        if parsing_state == 'heading_block':
            current_block.content += l + '\n'
        elif parsing_state == 'operation_block':
            if line.strip() == '' and idx != len(lines) - 1:
                parsing_state = 'normal'
            else:
                current_block.content += l + '\n'
        else:
            pass

        previous_line = l

    # Handle end of file for operation block
    if parsing_state == 'operation_block':
        parsing_state = 'normal'

    # After parsing, process operation blocks
    for block in blocks:
        if isinstance(block, OperationBlock):
            try:
                # print(f"!!! Processing BLOCK '{block}'")
                schema_processor.validate_operation(block)
            except Exception as e:
                print(f"Error processing operation '{block.operation}': {str(e)}\n")
    return blocks

class Parser:
    HEADING_PATTERN = re.compile(r'^(#+)\s+(.*?)\s*(?:\{id=(\w+)\})?$')
    OPERATION_PATTERN = re.compile(r'^@(\w+)(?:\s*\((.*?)\))?$')

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None

        self.schema_text = schema_text  # Ensure schema_text is defined
    def generate_id_from_title(self, title: str) -> str:
        # Normalize the title, remove non-alphanumeric characters, and convert to kebab-case
        normalized_title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
        kebab_case_id = re.sub(r'[^a-zA-Z0-9]+', '-', normalized_title).strip('-').lower()
        return kebab_case_id

    def parse(self, text: str) -> Dict[str, Node]:
        blocks = parse_document(text, self.schema_text)
        nodes = {}

        for block in blocks:
            if isinstance(block, HeadingBlock):
                node_id = block.id or self.generate_id_from_title(block.title)
                node = Node(
                    type=NodeType.HEADING,
                    name=block.title,
                    level=block.level,
                    id=node_id,
                    indent=block.level * 4,
                    content= block.content.strip(),
                    role=block.role,  # Default role for heading blocks
                    is_system=block.is_system  # Pass system block flag
                )
            elif isinstance(block, OperationBlock):
                node = Node(
                    type=NodeType.OPERATION,
                    name=block.operation,
                    level=1,
                    indent=4,
                    params = block.params, # Its deconstructed YAML operation params
                    content=block.content.strip(),
                    source_path=block.params.get('path', ''), # TODO !!! ALL PARAMS LOOKS WRONG
                    source_block_id=block.params.get('block_uri', ''),
                    target_path=block.params.get('to', {}).get('path', ''), # 
                    target_block_id=block.params.get('to', {}).get('block_uri', ''),
                    role=block.role  # Default role for operation blocks
                )
            else:
                continue  # Handle other block types if necessary

            self.add_node(node)
            nodes[node.key] = node

        return nodes

    def add_node(self, node: Node) -> None:
        self.nodes[node.key] = node
        if not self.head:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
    def get_node_by_id(self, id: str) -> Optional[Node]:
            return next((node for node in self.nodes.values() if node.id == id), None)

    def replace_node(self, target_key: str, new_node: Node):
        target_node = self.nodes.get(target_key)
        if not target_node:
            raise KeyError(f"Node with key '{target_key}' not found.")

        if target_node.prev:
            target_node.prev.next = new_node
            new_node.prev = target_node.prev
        else:
            self.head = new_node

        if target_node.next:
            target_node.next.prev = new_node
            new_node.next = target_node.next
        else:
            self.tail = new_node

        del self.nodes[target_key]
        self.nodes[new_node.key] = new_node

    def replace_node_with_ast(self, target_key: str, new_ast: Dict[str, Node]):
        old_node = self.nodes.get(target_key)
        if not old_node:
            raise KeyError(f"Node with key '{target_key}' not found.")

        new_nodes = list(new_ast.values())
        first_new_node = new_nodes[0]
        last_new_node = new_nodes[-1]

        if old_node.prev:
            old_node.prev.next = first_new_node
            first_new_node.prev = old_node.prev
        else:
            self.head = first_new_node

        if old_node.next:
            old_node.next.prev = last_new_node
            last_new_node.next = old_node.next
        else:
            self.tail = last_new_node

        del self.nodes[target_key]
        self.nodes.update({node.key: node for node in new_nodes})

    def prepend_node_with_ast(self, target_key: str, new_ast: Dict[str, Node]):
        target_node = self.nodes.get(target_key)
        if not target_node:
            raise KeyError(f"Node with key '{target_key}' not found.")

        new_nodes = list(new_ast.values())
        first_new_node = new_nodes[0]
        last_new_node = new_nodes[-1]

        if target_node.prev:
            target_node.prev.next = first_new_node
            first_new_node.prev = target_node.prev
        else:
            self.head = first_new_node

        target_node.prev = last_new_node
        last_new_node.next = target_node

        self.nodes.update({node.key: node for node in new_nodes})

    def append_node_with_ast(self, target_key: str, new_ast: Dict[str, Node]):
        target_node = self.nodes.get(target_key)
        if not target_node:
            raise KeyError(f"Node with key '{target_key}' not found.")

        new_nodes = list(new_ast.values())
        first_new_node = new_nodes[0]
        last_new_node = new_nodes[-1]

        if target_node.next:
            target_node.next.prev = last_new_node
            last_new_node.next = target_node.next
        else:
            self.tail = last_new_node

        target_node.next = first_new_node
        first_new_node.prev = target_node

        self.nodes.update({node.key: node for node in new_nodes})

def get_preceding_node(nodes: Dict[str, Node], head_node: Node) -> Optional[Node]:
    return head_node.prev

def get_following_node(nodes: Dict[str, Node], tail_node: Node) -> Optional[Node]:
    return tail_node.next

def remove_nodes_by_keys(parser: Parser, keys: list[str]) -> None:
    for key in keys:
        del parser.nodes[key]

def connect_nodes(preceding_node: 'Optional[Node]', new_head: 'Node', new_tail: 'Node', following_node: 'Optional[Node]') -> None:
    if preceding_node:
        preceding_node.next = new_head
        new_head.prev = preceding_node
    if following_node:
        new_tail.next = following_node
        following_node.prev = new_tail

def get_head(ast_or_nodes: 'Union[AST, Dict[str, Node]]') -> 'Optional[Node]':
    if isinstance(ast_or_nodes, AST):
        return ast_or_nodes.parser.head
    elif isinstance(ast_or_nodes, dict):
        return next(iter(ast_or_nodes.values())) if ast_or_nodes else None
    else:
        raise TypeError("Expected AST or dict of nodes")

def get_tail(ast_or_nodes: 'Union[AST, Dict[str, Node]]') -> 'Optional[Node]':
    if isinstance(ast_or_nodes, AST):
        return ast_or_nodes.parser.tail
    elif isinstance(ast_or_nodes, dict):
        return list(ast_or_nodes.values())[-1] if ast_or_nodes else None
    else:
        raise TypeError("Expected AST or dict of nodes")

def print_node(node: Node, level: int) -> None:
    indent = "  " * level
    print(f"{indent}{'='*40} Node {'='*40}")
    print(f"{indent}[{node.hash}] {node.type.value.upper()}: {node.name} (Level: {node.level})")
    print(f"{indent}Key: {node.key}")
    print(f"{indent}ID: {node.id}")
    print(f"{indent}Level: {node.level}")
    print(f"{indent}Role: {node.role}")

    if node.type == NodeType.OPERATION:
        print(f"{indent}Source Path: {node.source_path}")
        print(f"{indent}Source Block ID: {node.source_block_id}")
        print(f"{indent}Target Path: {node.target_path}")
        print(f"{indent}Target Block ID: {node.target_block_id}")

    print(f"{indent}Content:")
    for line in node.content.strip().split('\n'):
        print(f"{indent}{line}")

def print_ast_as_doubly_linked_list(ast: AST):
    current = ast.first()
    while current:
        prev_key = current.prev.key if current.prev else "    None"
        next_key = current.next.key if current.next else "None    "
        node_name_preview = current.content.strip()[:50].replace('\n', ' ')
        print(f"[{prev_key}] <- [id:{current.id}]({current.key}) {node_name_preview} -> [{next_key}]")
        current = current.next

def print_parsed_structure(ast: AST) -> None:
    current = ast.first()
    while current:
        print_node(current, 0)
        current = current.next