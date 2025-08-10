# Fractalic Linter
# 
# This module provides validation for Fractalic markdown files before execution.
# It catches common parsing errors, YAML syntax issues, and structural problems
# that could cause runtime failures.

import re
import yaml
import jsonschema
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

@dataclass
class LintError:
    """Represents a linting error found in the document."""
    line_number: int
    error_type: str
    message: str
    content: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"

class FractalicLinter:
    """
    Linter for Fractalic markdown files.
    
    Validates:
    1. Operation block YAML syntax and schema compliance
    2. Proper block termination (blank lines after operations)
    3. Structural integrity of markdown format
    4. Missing required parameters and unknown operations
    """
    
    def __init__(self, schema_text: str):
        """Initialize linter with operation schema."""
        self.schema_text = schema_text
        self.schema = yaml.safe_load(schema_text)
        self.operations_schema = self.schema.get('operations', {})
        self.console = Console()
        self.errors: List[LintError] = []
    
    def lint_file(self, file_path: str) -> List[LintError]:
        """
        Lint a Fractalic markdown file and return list of errors.
        
        Args:
            file_path: Path to the markdown file to lint
            
        Returns:
            List of LintError objects representing issues found
        """
        self.errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            self.errors.append(LintError(
                line_number=0,
                error_type="file_not_found",
                message=f"File not found: {file_path}"
            ))
            return self.errors
        except Exception as e:
            self.errors.append(LintError(
                line_number=0,
                error_type="file_read_error",
                message=f"Error reading file: {str(e)}"
            ))
            return self.errors
        
        return self.lint_content(content, file_path)
    
    def lint_content(self, content: str, file_path: str = "<content>") -> List[LintError]:
        """
        Lint markdown content and return list of errors.
        
        Args:
            content: Markdown content to lint
            file_path: Path for error reporting (optional)
            
        Returns:
            List of LintError objects representing issues found
        """
        self.errors = []
        lines = content.splitlines()
        
        # Parse document structure and validate
        self._validate_document_structure(lines)
        self._validate_operation_blocks(lines)
        
        return self.errors
    
    def _validate_document_structure(self, lines: List[str]) -> None:
        """
        Validate the overall document structure.
        
        Checks for:
        - Proper block termination with blank lines
        - Orphaned content after operations
        - Malformed heading syntax
        """
        parsing_state = 'normal'
        operation_start_line = None
        
        for idx, line in enumerate(lines):
            line_num = idx + 1
            stripped_line = line.strip()
            
            # Detect operation blocks
            if re.match(r'^@[a-zA-Z]+', line):
                if parsing_state == 'operation_block':
                    # Previous operation wasn't properly terminated
                    self.errors.append(LintError(
                        line_number=operation_start_line,
                        error_type="operation_not_terminated",
                        message="Operation block not properly terminated with blank line before next operation",
                        content=lines[operation_start_line-1] if operation_start_line > 0 else line
                    ))
                
                parsing_state = 'operation_block'
                operation_start_line = line_num
                continue
            
            # Detect heading blocks
            elif re.match(r'^#+ ', line):
                if parsing_state == 'operation_block':
                    # Operation terminated by heading - this is valid
                    parsing_state = 'normal'
                    operation_start_line = None
                continue
            
            # Check for blank lines
            elif stripped_line == '':
                if parsing_state == 'operation_block':
                    # Check if this is the end of the file
                    if idx == len(lines) - 1:
                        parsing_state = 'normal'
                        operation_start_line = None
                    else:
                        # Check what comes after the blank line
                        next_non_empty_idx = self._find_next_non_empty_line(lines, idx + 1)
                        if next_non_empty_idx is not None:
                            next_line = lines[next_non_empty_idx]
                            # Valid if next content is heading or operation
                            if re.match(r'^#+ ', next_line) or re.match(r'^@[a-zA-Z]+', next_line):
                                parsing_state = 'normal'
                                operation_start_line = None
                            # Also valid if we've reached the end
                            elif next_non_empty_idx == len(lines) - 1:
                                parsing_state = 'normal' 
                                operation_start_line = None
                        else:
                            # No more content after blank line - valid termination
                            parsing_state = 'normal'
                            operation_start_line = None
                continue
            
            # Non-blank, non-heading, non-operation content
            else:
                if parsing_state == 'operation_block':
                    # This is YAML content within operation - continue
                    continue
                elif parsing_state == 'normal':
                    # Check if this looks like orphaned content after an operation
                    # Look backwards to see if we just had an operation + blank lines
                    prev_operation_line = self._find_previous_operation(lines, idx)
                    if prev_operation_line is not None:
                        # Check if there are only blank lines between operation and this content
                        has_blank_gap = self._has_blank_line_gap(lines, prev_operation_line, idx)
                        if has_blank_gap:
                            self.errors.append(LintError(
                                line_number=line_num,
                                error_type="orphaned_content",
                                message="Content found after operation block that is not a proper heading or operation",
                                content=line,
                                severity="warning"
                            ))
    
    def _validate_operation_blocks(self, lines: List[str]) -> None:
        """
        Validate operation blocks for YAML syntax and schema compliance.
        """
        operation_blocks = self._extract_operation_blocks(lines)
        
        for block in operation_blocks:
            self._validate_single_operation(block, lines)
    
    def _extract_operation_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract operation blocks from document lines."""
        blocks = []
        parsing_state = 'normal'
        current_block = None
        
        for idx, line in enumerate(lines):
            line_num = idx + 1
            
            # Detect operation start
            if re.match(r'^@[a-zA-Z]+', line):
                if current_block is not None:
                    # Save previous block
                    blocks.append(current_block)
                
                operation_name = line.strip('@').strip()
                current_block = {
                    'operation': operation_name,
                    'start_line': line_num,
                    'content': [line],
                    'yaml_start_line': line_num + 1
                }
                parsing_state = 'operation_block'
                continue
            
            # Collect operation content
            elif parsing_state == 'operation_block':
                if line.strip() == '':
                    # Check if this is truly the end
                    next_non_empty_idx = self._find_next_non_empty_line(lines, idx + 1)
                    if next_non_empty_idx is None or \
                       re.match(r'^#+ ', lines[next_non_empty_idx]) or \
                       re.match(r'^@[a-zA-Z]+', lines[next_non_empty_idx]):
                        # End of operation block
                        current_block['end_line'] = line_num - 1
                        blocks.append(current_block)
                        current_block = None
                        parsing_state = 'normal'
                    else:
                        # Blank line within operation (this might be an error)
                        current_block['content'].append(line)
                else:
                    current_block['content'].append(line)
        
        # Handle operation at end of file
        if current_block is not None:
            current_block['end_line'] = len(lines)
            blocks.append(current_block)
        
        return blocks
    
    def _validate_single_operation(self, block: Dict[str, Any], lines: List[str]) -> None:
        """Validate a single operation block."""
        operation_name = block['operation']
        start_line = block['start_line']
        
        # Check if operation is known
        if operation_name not in self.operations_schema:
            self.errors.append(LintError(
                line_number=start_line,
                error_type="unknown_operation",
                message=f"Unknown operation '@{operation_name}'",
                content=block['content'][0] if block['content'] else f"@{operation_name}"
            ))
            return
        
        # Extract YAML content (everything except the first @operation line)
        yaml_lines = block['content'][1:]
        yaml_content = '\n'.join(yaml_lines)
        yaml_start_line = block['yaml_start_line']
        
        # Parse YAML
        try:
            params = yaml.safe_load(yaml_content) if yaml_content.strip() else {}
            if params is None:
                params = {}
        except yaml.YAMLError as e:
            # Calculate error line within YAML
            yaml_error_line = yaml_start_line
            if hasattr(e, 'problem_mark') and e.problem_mark:
                yaml_error_line += e.problem_mark.line
            
            self.errors.append(LintError(
                line_number=yaml_error_line,
                error_type="yaml_syntax_error",
                message=f"YAML syntax error in @{operation_name}: {str(e)}",
                content=yaml_content
            ))
            return
        
        # Validate against schema
        try:
            schema = self.operations_schema[operation_name]
            jsonschema.validate(instance=params, schema=schema)
        except jsonschema.ValidationError as e:
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            self.errors.append(LintError(
                line_number=yaml_start_line,
                error_type="schema_validation_error", 
                message=f"Schema validation error in @{operation_name} at {error_path}: {e.message}",
                content=yaml_content
            ))
            return
        
        # Check for empty lines within YAML that might indicate parsing issues
        self._check_yaml_structure(yaml_lines, yaml_start_line, operation_name)
    
    def _check_yaml_structure(self, yaml_lines: List[str], start_line: int, operation_name: str) -> None:
        """Check for structural issues in YAML that might cause parsing problems."""
        in_multiline = False
        multiline_indicator = None
        
        for idx, line in enumerate(yaml_lines):
            line_num = start_line + idx
            stripped = line.strip()
            
            # Skip empty lines at start or end
            if not stripped and (idx == 0 or idx == len(yaml_lines) - 1):
                continue
            
            # Detect multiline indicators
            if re.search(r':\s*[|>]', line):
                in_multiline = True
                multiline_indicator = line_num
                continue
            
            # Check for empty lines in the middle of YAML
            if not stripped and not in_multiline:
                # Empty line in middle of YAML parameters
                self.errors.append(LintError(
                    line_number=line_num,
                    error_type="yaml_structure_warning",
                    message=f"Empty line within @{operation_name} YAML parameters may cause parsing issues",
                    severity="warning"
                ))
            
            # Reset multiline mode on unindented line
            if stripped and not line.startswith('  ') and not line.startswith('\t'):
                if in_multiline and not re.search(r':\s*[|>]', line):
                    in_multiline = False
                    multiline_indicator = None
    
    def _find_next_non_empty_line(self, lines: List[str], start_idx: int) -> Optional[int]:
        """Find the next non-empty line starting from start_idx."""
        for idx in range(start_idx, len(lines)):
            if lines[idx].strip():
                return idx
        return None
    
    def _find_previous_operation(self, lines: List[str], current_idx: int) -> Optional[int]:
        """Find the most recent operation line before current_idx."""
        for idx in range(current_idx - 1, -1, -1):
            if re.match(r'^@[a-zA-Z]+', lines[idx]):
                return idx
        return None
    
    def _has_blank_line_gap(self, lines: List[str], operation_line: int, content_line: int) -> bool:
        """Check if there's a gap of blank lines between operation and content."""
        has_blank = False
        for idx in range(operation_line + 1, content_line):
            if lines[idx].strip() == '':
                has_blank = True
            elif lines[idx].strip():
                # Non-blank content in between - this might be YAML
                return False
        return has_blank
    
    def print_errors(self, file_path: str = "") -> None:
        """Print linting errors in a formatted way."""
        if not self.errors:
            self.console.print(f"[green]✓[/green] No linting errors found{' in ' + file_path if file_path else ''}")
            return
        
        title = f"Linting Errors{' in ' + file_path if file_path else ''}"
        self.console.print(Panel(f"Found {len(self.errors)} linting issues", title=title, border_style="red"))
        
        for error in self.errors:
            severity_color = {
                "error": "red",
                "warning": "yellow", 
                "info": "blue"
            }.get(error.severity, "red")
            
            self.console.print(f"\n[{severity_color}]{error.severity.upper()}[/{severity_color}] Line {error.line_number}: {error.message}")
            
            if error.content:
                self.console.print(
                    Syntax(
                        error.content.strip(),
                        "yaml" if "yaml" in error.error_type else "markdown",
                        line_numbers=True,
                        theme="monokai",
                        word_wrap=True,
                        background_color="default"
                    )
                )
    
    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return len([e for e in self.errors if e.severity == "error"]) > 0

    def get_errors_as_text(self, file_path: str = "") -> str:
        """Get linting errors formatted as plain text for inclusion in context files."""
        if not self.errors:
            return f"✓ No linting errors found{' in ' + file_path if file_path else ''}"
        
        output = []
        output.append(f"Found {len(self.errors)} linting issues{' in ' + file_path if file_path else ''}:")
        output.append("=" * 60)
        
        for error in self.errors:
            output.append(f"\n{error.severity.upper()} Line {error.line_number}: {error.message}")
            
            if error.content:
                # Add the content with line numbers
                lines = error.content.strip().split('\n')
                for i, line in enumerate(lines, 1):
                    output.append(f"  {i:2d} {line}")
        
        return "\n".join(output)
    
    def has_warnings(self) -> bool:
        """Check if any warnings were found.""" 
        return len([e for e in self.errors if e.severity == "warning"]) > 0


def lint_fractalic_file(file_path: str, schema_text: str, print_results: bool = False) -> List[LintError]:
    """
    Lint a Fractalic file and optionally print results.
    
    Args:
        file_path: Path to the file to lint
        schema_text: Schema text for operation validation
        print_results: Whether to print results to console
        
    Returns:
        List of LintError objects
        
    Raises:
        FractalicLintError: If critical linting errors are found
    """
    linter = FractalicLinter(schema_text)
    errors = linter.lint_file(file_path)
    
    if print_results:
        linter.print_errors(file_path)
    
    if linter.has_errors():
        error_count = len([e for e in errors if e.severity == "error"])
        # Create enhanced error with formatted text as an attribute
        formatted_errors = linter.get_errors_as_text(file_path)
        error = FractalicLintError(f"Found {error_count} linting errors in {file_path}")
        error.formatted_errors = formatted_errors
        raise error
    
    return errors


class FractalicLintError(Exception):
    """Exception raised when linting errors are found."""
    def __init__(self, message: str):
        super().__init__(message)
        self.formatted_errors = ""


class FractalicLintError(Exception):
    """Exception raised when linting errors are found."""
    pass
