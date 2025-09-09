# Fractalic Simple JSON Tools - Technical Specification Document

## Table of Contents

1. [Overview](#overview)
2. [Simple JSON Convention](#simple-json-convention)
3. [Implementation Requirements](#implementation-requirements)
4. [Schema Format Specification](#schema-format-specification)
5. [Implementation Examples](#implementation-examples)
6. [Best Practices](#best-practices)
7. [Testing and Validation](#testing-and-validation)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Fractalic uses **Simple JSON Schema Discovery** for automatic tool registration. This is the ONLY supported approach for autodiscoverable tools.

### Key Benefits
- **Zero-configuration deployment**: Drop a script in `tools/`, restart Fractalic
- **Fast discovery**: Tools must respond within 200ms
- **Self-documenting**: Tools expose their own schema and parameter definitions
- **Error-resistant**: Only valid tools are registered, problematic files are skipped
- **JSON-native**: Clean integration with LLM function calling

### Discovery Process
```mermaid
graph TD
    A[Script in tools/] --> B{Has .yaml?}
    B -->|Yes| C[Use YAML manifest]
    B -->|No| D[Test: {"__test__": true}]
    D --> E{JSON response < 200ms?}
    E -->|Yes| F[Register as Simple JSON tool]
    E -->|No| G[Skip - not a valid tool]
    F --> H[Tool available to LLM]
```

---

## Simple JSON Convention

### Core Principle
Tools accept JSON input as a single command-line argument and return JSON output to stdout. This provides a clean, language-agnostic interface that works well with LLM function calling.

### Discovery Test
Every tool must respond to the discovery test:
```bash
python3 my_tool.py '{"__test__": true}'
```

**Expected response** (any valid JSON):
```json
{"success": true}
```

---

## Implementation Requirements

### Mandatory Requirements

1. **Fast Response**: Must respond to discovery test within 200ms
2. **JSON Input**: Accept single JSON string argument via `sys.argv[1]`
3. **JSON Output**: Print JSON response to stdout using `json.dumps()`
4. **Error Handling**: Return errors as JSON: `{"error": "message"}`
5. **Clean Exit**: Exit with code 0 for success, 1 for errors

### Optional Enhancements

1. **Schema Dump**: Support `--fractalic-dump-schema` for rich parameter definitions
2. **UTF-8 Support**: Use `ensure_ascii=False` in `json.dumps()`
3. **Input Validation**: Validate JSON input against expected schema

---

## Schema Format Specification

### Basic Schema Structure
```json
{
    "description": "Tool description for LLM context",
    "parameters": {
        "type": "object",
        "properties": {
            "param_name": {
                "type": "string|number|boolean|array|object",
                "description": "Parameter description",
                "enum": ["optional", "values"],
                "default": "optional_default"
            }
        },
        "required": ["required_param_names"],
        "additionalProperties": false
    }
}
```

### Supported Parameter Types
- **string**: Text values
- **number**: Numeric values (integer or float)
- **boolean**: true/false values
- **array**: Lists of values
- **object**: Nested JSON objects

---

## Implementation Examples

### Basic Tool Template
```python
#!/usr/bin/env python3
"""Brief description of the tool functionality."""
import json
import sys

def process_data(data):
    """Main processing function."""
    action = data.get("action")
    
    if action == "example":
        param = data.get("param", "default")
        return {"result": f"Processed: {param}"}
    elif action == "calculate":
        x = data.get("x", 0)
        y = data.get("y", 0)
        return {"result": x + y}
    else:
        return {"error": f"Unknown action: {action}"}

def main():
    # Discovery test - must respond quickly
    if len(sys.argv) == 2 and sys.argv[1] == '{"__test__": true}':
        print(json.dumps({"success": True}))
        return
    
    # Optional: Rich schema for better LLM integration
    if len(sys.argv) == 2 and sys.argv[1] == "--fractalic-dump-schema":
        schema = {
            "description": "Example tool demonstrating Simple JSON convention",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["example", "calculate"],
                        "description": "Action to perform"
                    },
                    "param": {
                        "type": "string",
                        "description": "Parameter for example action"
                    },
                    "x": {
                        "type": "number",
                        "description": "First number for calculate action"
                    },
                    "y": {
                        "type": "number", 
                        "description": "Second number for calculate action"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        }
        print(json.dumps(schema, ensure_ascii=False))
        return
    
    # Process JSON input
    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected exactly one JSON argument")
        
        params = json.loads(sys.argv[1])
        result = process_data(params)
        print(json.dumps(result, ensure_ascii=False))
        
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {str(e)}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Best Practices

### Performance
- **Fast startup**: Avoid heavy imports at module level
- **Quick response**: Respond to discovery test in <200ms
- **Lazy loading**: Import dependencies only when needed

### Error Handling
- **JSON errors**: Always return errors as JSON objects
- **Descriptive messages**: Include helpful error descriptions
- **Graceful failure**: Handle edge cases and invalid inputs

### Code Organization
```python
# Good: Fast startup, lazy imports
def process_data(data):
    if data.get("action") == "heavy_operation":
        import heavy_library  # Import only when needed
        return heavy_library.process(data)

# Bad: Slow startup
import heavy_library  # This slows down discovery
```

---

## Testing and Validation

### Manual Testing
```bash
# Test discovery
python3 my_tool.py '{"__test__": true}'

# Test functionality
python3 my_tool.py '{"action": "example", "param": "test"}'
```

### Integration Testing
```python
from core.plugins.tool_registry import ToolRegistry

# Test tool registration
registry = ToolRegistry(tools_dir="./tools")
assert "my_tool" in registry
```

---

## Troubleshooting

### Common Issues

#### Tool Not Discovered
- **Check response time**: Tool must respond to discovery test in <200ms
- **Verify JSON output**: Must return valid JSON to stdout
- **Check file permissions**: Ensure script is executable

#### Runtime Errors
- **JSON format**: Ensure all output is valid JSON
- **Error handling**: Wrap all operations in try/catch
- **Exit codes**: Use exit(0) for success, exit(1) for errors
For scripts exposing multiple tools, return an array of schema objects:

```json
[
  {
    "name": "tool_one",
    "description": "First tool function",
    "parameters": { /* ... */ }
  },
  {
    "name": "tool_two", 
    "description": "Second tool function",
    "parameters": { /* ... */ }
  }
]
```

---

## Implementation Requirements

### Mandatory Requirements

#### 1. Schema Dump Support
**All autodiscoverable tools MUST implement schema dumping:**

```python
# Python implementation
if args.fractalic_dump_schema:
    schema = {
        "name": "my_tool",
        "description": "Tool description",
        "parameters": {
            "type": "object",
            "properties": {
                # ... parameter definitions
            },
            "required": ["required_params"]
        }
    }
    print(json.dumps(schema, indent=2))
    return
```

#### 2. Help Text Contract
**Tools MUST provide comprehensive help text:**
- First non-blank line becomes tool description
- All parameters must have help text
- Must exit with code 0 on `--help`

#### 3. Argument Parsing
**Python tools should use argparse pattern:**

```python
parser = argparse.ArgumentParser(description="Tool description")
parser.add_argument("--param", required=True, help="Parameter description")
parser.add_argument("--fractalic-dump-schema", action="store_true", help=argparse.SUPPRESS)
args = parser.parse_args()
```

#### 4. JSON Output (Recommended)
**Tools should output structured JSON for better LLM integration:**

```python
result = {"status": "success", "data": processed_data}
print(json.dumps(result))
```

### Optional Enhancements

#### Multi-Tool Schema Support
Implement `--fractalic-dump-multi-schema` for multiple tool exposure:

```python
if args.fractalic_dump_multi_schema:
    schemas = [
        {"name": "tool_a", "description": "...", "parameters": {...}},
        {"name": "tool_b", "description": "...", "parameters": {...}}
    ]
    print(json.dumps(schemas, indent=2))
    return
```

---

## Single-Tool vs Multi-Tool Patterns

### Simple JSON Pattern (Recommended)
**Use when**: You want minimal boilerplate and flexible functionality

```python
#!/usr/bin/env python3
"""Weather data fetcher with multiple actions"""
import json, sys, requests

def process_data(data):
    action = data.get("action")
    
    if action == "current_weather":
        lat, lon = data.get("latitude"), data.get("longitude")
        if not lat or not lon:
            return {"error": "latitude and longitude required"}
        
        # Mock API call
        return {
            "location": {"lat": lat, "lon": lon},
            "temperature": 22,
            "condition": "sunny"
        }
    
    elif action == "forecast":
        location = data.get("location")
        days = data.get("days", 3)
        # Mock forecast
        return {
            "location": location,
            "forecast": [{"day": i+1, "temp": 20+i} for i in range(days)]
        }
    
    return {"error": f"Unknown action: {action}"}

def main():
    # Test mode (REQUIRED)
    if len(sys.argv) == 2 and sys.argv[1] == '{"__test__": true}':
        print(json.dumps({"success": True, "_simple": True}))
        return
    
    # Optional: Rich schema
    if len(sys.argv) == 2 and sys.argv[1] == "--fractalic-dump-schema":
        schema = {
            "description": "Weather data fetcher with multiple actions",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["current_weather", "forecast"],
                        "description": "Weather action to perform"
                    },
                    "latitude": {"type": "number", "description": "Latitude (-90 to 90)"},
                    "longitude": {"type": "number", "description": "Longitude (-180 to 180)"},
                    "location": {"type": "string", "description": "Location name for forecast"},
                    "days": {"type": "integer", "description": "Forecast days (default: 3)"}
                },
                "required": ["action"]
            }
        }
        print(json.dumps(schema, ensure_ascii=False))
        return
    
    # Process JSON input
    try:
        params = json.loads(sys.argv[1])
        result = process_data(params)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__": main()
```

### Legacy Single-Tool Pattern
**Use when**: Script performs one primary function with variations via parameters

```python
#!/usr/bin/env python3
"""Weather data fetcher for specific locations"""

import argparse
import json
import requests

def get_tool_schema():
    return {
        "name": "weather_fetch",
        "description": "Fetch current weather data for a geographic location",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude coordinate (-90 to 90)"
                },
                "longitude": {
                    "type": "number", 
                    "description": "Longitude coordinate (-180 to 180)"
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "default": "metric",
                    "description": "Temperature units"
                }
            },
            "required": ["latitude", "longitude"]
        }
    }

# Legacy argparse implementation
parser = argparse.ArgumentParser(description="Weather data fetcher")
parser.add_argument("--latitude", type=float, required=True, help="Latitude (-90 to 90)")
parser.add_argument("--longitude", type=float, required=True, help="Longitude (-180 to 180)")
parser.add_argument("--units", choices=["metric", "imperial"], default="metric", help="Temperature units")
parser.add_argument("--fractalic-dump-schema", action="store_true", help=argparse.SUPPRESS)
args = parser.parse_args()

if args.fractalic_dump_schema:
    print(json.dumps(get_tool_schema(), indent=2))
    exit(0)

# Tool logic...
```

### Legacy Multi-Tool Pattern

def main():
    parser = argparse.ArgumentParser(description="Fetch current weather data for a geographic location")
    parser.add_argument("--latitude", type=float, required=True, help="Latitude coordinate (-90 to 90)")
    parser.add_argument("--longitude", type=float, required=True, help="Longitude coordinate (-180 to 180)")
    parser.add_argument("--units", choices=["metric", "imperial"], default="metric", help="Temperature units")
    parser.add_argument("--fractalic-dump-schema", action="store_true", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    if args.fractalic_dump_schema:
        print(json.dumps(get_tool_schema(), indent=2))
        return
    
    # Implementation logic
    weather_data = fetch_weather(args.latitude, args.longitude, args.units)
    print(json.dumps(weather_data))

if __name__ == "__main__":
    main()
```

### Multi-Tool Pattern
**Use when**: Script provides multiple related but distinct functions

Based on the fractalic_generator example:

```python
#!/usr/bin/env python3
"""Fractalic operation generator with multiple tool functions"""

import argparse
import json

class OperationGenerator:
    def get_multi_tool_schema(self):
        return [
            {
                "name": "generate_import_op",
                "description": "Generate fractalic import operation syntax",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "file": {"type": "string", "description": "Source file path"},
                        "block": {"type": "string", "description": "Source block path"},
                        "mode": {"type": "string", "enum": ["append", "prepend", "replace"]}
                    },
                    "required": ["file"]
                }
            },
            {
                "name": "generate_llm_op",
                "description": "Generate fractalic LLM operation syntax",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "LLM prompt text"},
                        "model": {"type": "string", "description": "LLM model name"},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["prompt"]
                }
            }
        ]

def main():
    parser = argparse.ArgumentParser(description="Generate fractalic operation syntax blocks")
    parser.add_argument("operation", nargs="?", help="Operation type to generate")
    parser.add_argument("--fractalic-dump-multi-schema", action="store_true", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    generator = OperationGenerator()
    
    if args.fractalic_dump_multi_schema:
        print(json.dumps(generator.get_multi_tool_schema(), indent=2))
        return
    
    # Handle individual operations based on first argument
    # Implementation details...
```

---

## Parameter Types and Validation

### Supported Parameter Types

#### String Parameters
```json
{
  "param_name": {
    "type": "string",
    "description": "Text input parameter",
    "minLength": 1,          // optional
    "maxLength": 100,        // optional
    "pattern": "^[a-z]+$"    // optional regex
  }
}
```

#### Numeric Parameters
```json
{
  "count": {
    "type": "integer",
    "description": "Number of items to process",
    "minimum": 1,
    "maximum": 1000,
    "default": 10
  },
  "percentage": {
    "type": "number",
    "description": "Percentage value",
    "minimum": 0.0,
    "maximum": 100.0
  }
}
```

#### Boolean Parameters
```json
{
  "verbose": {
    "type": "boolean",
    "description": "Enable verbose output",
    "default": false
  }
}
```

#### Enum Parameters
```json
{
  "format": {
    "type": "string",
    "enum": ["json", "yaml", "xml"],
    "description": "Output format",
    "default": "json"
  }
}
```

#### Array Parameters
```json
{
  "files": {
    "type": "array",
    "items": {"type": "string"},
    "description": "List of file paths to process",
    "minItems": 1,
    "maxItems": 10
  }
}
```

### Parameter Validation Best Practices

1. **Always provide descriptions**: Essential for LLM understanding
2. **Use appropriate constraints**: `minimum`, `maximum`, `minLength`, etc.
3. **Provide sensible defaults**: Reduces LLM decision complexity
4. **Use enums for limited choices**: Prevents invalid values
5. **Mark required parameters**: Clear dependency specification

---

## Implementation Examples

### Example 1: File Processing Tool

```python
#!/usr/bin/env python3
"""File processor with various operation modes"""

import argparse
import json
import os
from pathlib import Path

def get_tool_schema():
    return {
        "name": "file_processor",
        "description": "Process files with various operations like counting lines, words, or characters",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to process"
                },
                "operation": {
                    "type": "string",
                    "enum": ["count_lines", "count_words", "count_chars", "file_size"],
                    "description": "Type of operation to perform",
                    "default": "count_lines"
                },
                "include_empty": {
                    "type": "boolean", 
                    "description": "Include empty lines in line count",
                    "default": true
                }
            },
            "required": ["file_path"]
        }
    }

def process_file(file_path, operation, include_empty=True):
    """Process file based on operation type"""
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if operation == "file_size":
            return {"file_size": path.stat().st_size, "file_path": file_path}
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if operation == "count_lines":
            lines = content.splitlines()
            if not include_empty:
                lines = [line for line in lines if line.strip()]
            return {"line_count": len(lines), "file_path": file_path}
        
        elif operation == "count_words":
            words = content.split()
            return {"word_count": len(words), "file_path": file_path}
        
        elif operation == "count_chars":
            return {"char_count": len(content), "file_path": file_path}
        
        else:
            return {"error": f"Unknown operation: {operation}"}
            
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Process files with various operations like counting lines, words, or characters")
    parser.add_argument("--file-path", required=True, help="Path to the file to process")
    parser.add_argument("--operation", choices=["count_lines", "count_words", "count_chars", "file_size"], 
                       default="count_lines", help="Type of operation to perform")
    parser.add_argument("--include-empty", action="store_true", default=True, 
                       help="Include empty lines in line count")
    parser.add_argument("--fractalic-dump-schema", action="store_true", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    if args.fractalic_dump_schema:
        print(json.dumps(get_tool_schema(), indent=2))
        return
    
    result = process_file(args.file_path, args.operation, args.include_empty)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
```

### Example 2: Multi-Tool Text Utilities

```python
#!/usr/bin/env python3
"""Text processing utilities - multiple tools in one script"""

import argparse
import json
import re
from typing import List, Dict, Any

class TextUtilities:
    def get_multi_tool_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "text_statistics",
                "description": "Calculate statistics for a text string (word count, character count, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text content to analyze"
                        },
                        "include_spaces": {
                            "type": "boolean",
                            "description": "Include spaces in character count",
                            "default": true
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "text_transform",
                "description": "Transform text using various operations (uppercase, lowercase, title case, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text content to transform"
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["uppercase", "lowercase", "title_case", "reverse", "remove_spaces"],
                            "description": "Transformation operation to apply"
                        }
                    },
                    "required": ["text", "operation"]
                }
            },
            {
                "name": "text_search",
                "description": "Search for patterns in text using regular expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text content to search in"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Regular expression pattern to search for"
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether search should be case sensitive",
                            "default": false
                        }
                    },
                    "required": ["text", "pattern"]
                }
            }
        ]
    
    def text_statistics(self, text: str, include_spaces: bool = True) -> Dict[str, Any]:
        """Calculate text statistics"""
        words = text.split()
        lines = text.splitlines()
        
        char_count = len(text) if include_spaces else len(text.replace(' ', ''))
        
        return {
            "word_count": len(words),
            "character_count": char_count,
            "line_count": len(lines),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    def text_transform(self, text: str, operation: str) -> Dict[str, Any]:
        """Transform text based on operation"""
        transformations = {
            "uppercase": text.upper(),
            "lowercase": text.lower(),
            "title_case": text.title(),
            "reverse": text[::-1],
            "remove_spaces": text.replace(' ', '')
        }
        
        if operation not in transformations:
            return {"error": f"Unknown operation: {operation}"}
        
        return {
            "original_text": text,
            "transformed_text": transformations[operation],
            "operation": operation
        }
    
    def text_search(self, text: str, pattern: str, case_sensitive: bool = False) -> Dict[str, Any]:
        """Search for pattern in text"""
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            matches = re.finditer(pattern, text, flags)
            match_results = []
            
            for match in matches:
                match_results.append({
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "line_number": text[:match.start()].count('\n') + 1
                })
            
            return {
                "pattern": pattern,
                "total_matches": len(match_results),
                "matches": match_results,
                "case_sensitive": case_sensitive
            }
        except re.error as e:
            return {"error": f"Invalid regex pattern: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="Text processing utilities")
    
    # Tool selection (first positional argument)
    parser.add_argument("tool", nargs="?", 
                       choices=["text_statistics", "text_transform", "text_search"],
                       help="Tool function to execute")
    
    # Common parameters
    parser.add_argument("--text", required=False, help="Text content to process")
    
    # text_statistics parameters
    parser.add_argument("--include-spaces", action="store_true", default=True,
                       help="Include spaces in character count")
    
    # text_transform parameters  
    parser.add_argument("--operation", 
                       choices=["uppercase", "lowercase", "title_case", "reverse", "remove_spaces"],
                       help="Transformation operation")
    
    # text_search parameters
    parser.add_argument("--pattern", help="Regular expression pattern")
    parser.add_argument("--case-sensitive", action="store_true", 
                       help="Case sensitive search")
    
    # Schema dump support
    parser.add_argument("--fractalic-dump-multi-schema", action="store_true", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    utils = TextUtilities()
    
    # Handle schema dump
    if args.fractalic_dump_multi_schema:
        print(json.dumps(utils.get_multi_tool_schema(), indent=2))
        return
    
    # Validate tool selection
    if not args.tool:
        parser.error("Tool name is required")
    
    if not args.text:
        parser.error("--text parameter is required")
    
    # Execute the selected tool
    if args.tool == "text_statistics":
        result = utils.text_statistics(args.text, args.include_spaces)
    elif args.tool == "text_transform":
        if not args.operation:
            parser.error("--operation is required for text_transform")
        result = utils.text_transform(args.text, args.operation)
    elif args.tool == "text_search":
        if not args.pattern:
            parser.error("--pattern is required for text_search")
        result = utils.text_search(args.text, args.pattern, args.case_sensitive)
    else:
        result = {"error": f"Unknown tool: {args.tool}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

---

## Best Practices

### 1. Schema Design
- **Descriptive names**: Use clear, descriptive tool and parameter names
- **Comprehensive descriptions**: Provide detailed descriptions for LLM understanding
- **Logical grouping**: Group related parameters and use sensible defaults
- **Validation constraints**: Use appropriate type constraints and validation rules

### 2. Error Handling
```python
def safe_operation(param):
    try:
        result = process(param)
        return {"status": "success", "data": result}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {str(e)}"}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error": f"Unexpected error: {str(e)}"}
```

### 3. Output Consistency
- **Always return JSON**: Structured data is easier for LLMs to process
- **Consistent error format**: Use standard error response structure
- **Include metadata**: Add context like timestamps, version info, etc.

### 4. Performance Considerations
- **Timeout handling**: Implement timeouts for long-running operations
- **Resource limits**: Validate input sizes and resource usage
- **Caching**: Cache expensive computations when appropriate

### 5. Documentation
```python
"""
Tool Title: Clear, descriptive title

Description: Detailed description of what the tool does, when to use it,
and any important limitations or requirements.

Usage Examples:
  python tool.py --param1 value1 --param2 value2
  python tool.py --fractalic-dump-schema

Output Format:
  {
    "status": "success|error",
    "data": {...},
    "error": "error message if applicable"
  }
"""
```

---

## Testing and Validation

### 1. Schema Validation
Test schema dump functionality:

```bash
# Test single-tool schema dump
python my_tool.py --fractalic-dump-schema

# Test multi-tool schema dump  
python my_tool.py --fractalic-dump-multi-schema

# Validate JSON structure
python my_tool.py --fractalic-dump-schema | jq .
```

### 2. Parameter Testing
Test all parameter combinations:

```python
#!/usr/bin/env python3
"""Test script for tool validation"""

import subprocess
import json
import sys

def test_tool_schema(script_path):
    """Test tool schema generation"""
    try:
        result = subprocess.run(
            [sys.executable, script_path, "--fractalic-dump-schema"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            print(f"Schema dump failed: {result.stderr}")
            return False
        
        schema = json.loads(result.stdout)
        
        # Validate required fields
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in schema:
                print(f"Missing required field: {field}")
                return False
        
        print("Schema validation passed")
        return True
        
    except Exception as e:
        print(f"Schema test error: {e}")
        return False

def test_tool_execution(script_path, test_args):
    """Test tool execution with sample arguments"""
    try:
        cmd = [sys.executable, script_path] + test_args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"Execution failed: {result.stderr}")
            return False
        
        # Try to parse JSON output
        try:
            output = json.loads(result.stdout)
            print("Execution test passed")
            return True
        except json.JSONDecodeError:
            print("Warning: Output is not valid JSON")
            return True  # May be valid for some tools
            
    except Exception as e:
        print(f"Execution test error: {e}")
        return False

if __name__ == "__main__":
    script = sys.argv[1] if len(sys.argv) > 1 else "my_tool.py"
    
    print(f"Testing tool: {script}")
    
    # Test schema generation
    if not test_tool_schema(script):
        sys.exit(1)
    
    # Test execution (customize test_args for your tool)
    test_args = ["--help"]  # Safe test
    if not test_tool_execution(script, test_args):
        sys.exit(1)
    
    print("All tests passed!")
```

### 3. Integration Testing
Test with the actual tool registry:

```python
from core.plugins.tool_registry import ToolRegistry

# Test tool discovery
registry = ToolRegistry(tools_dir="test_tools")
print("Discovered tools:", list(registry.keys()))

# Test schema generation
schema = registry.generate_schema()
print("Generated schema:", json.dumps(schema, indent=2))

# Test tool execution
if "my_tool" in registry:
    result = registry["my_tool"](param1="value1", param2="value2")
    print("Execution result:", result)
```

---

## Troubleshooting

### Common Issues

#### 1. Schema Dump Not Working
**Symptoms**: Tool not discovered, empty schema
**Causes**: 
- Missing `--fractalic-dump-schema` argument handling
- Script crashes during schema dump
- Invalid JSON output

**Solutions**:
```python
# Add proper error handling
if args.fractalic_dump_schema:
    try:
        schema = get_tool_schema()
        print(json.dumps(schema, indent=2))
        sys.exit(0)  # Explicit exit
    except Exception as e:
        print(f"Schema generation error: {e}", file=sys.stderr)
        sys.exit(1)
```

#### 2. Invalid Parameter Types
**Symptoms**: LLM receives wrong parameter types
**Causes**:
- Mismatched argparse and schema types
- Missing type conversion

**Solutions**:
```python
# Ensure type consistency
parser.add_argument("--count", type=int, help="Number of items")
# Schema should match:
"count": {"type": "integer", "description": "Number of items"}
```

#### 3. Multi-Tool Registration Issues
**Symptoms**: Only first tool registered, tools overwrite each other
**Causes**:
- Incorrect multi-tool schema format
- Missing tool name in execution

**Solutions**:
```python
# Proper multi-tool execution handling
if args.tool_name:  # First argument should be tool name
    if args.tool_name == "tool_a":
        result = execute_tool_a(args)
    elif args.tool_name == "tool_b":
        result = execute_tool_b(args)
    else:
        result = {"error": f"Unknown tool: {args.tool_name}"}
```

#### 4. Help Text Parsing Fallback Issues
**Symptoms**: Incomplete parameter detection in fallback mode
**Causes**:
- Inconsistent help text format
- Missing parameter descriptions

**Solutions**:
```python
# Ensure consistent help format
parser.add_argument("--input-file", required=True, 
                   help="Path to input file for processing")
# Not: help="input file" (too brief)
# Not: help="" (empty)
```

### Debugging Tools

#### 1. Test Schema Generation
```bash
# Test schema output
python tool.py --fractalic-dump-schema | jq .

# Test multi-tool schema
python tool.py --fractalic-dump-multi-schema | jq .

# Validate schema structure
python -c "
import json, sys
schema = json.load(sys.stdin)
required = ['name', 'description', 'parameters']
missing = [f for f in required if f not in schema]
if missing:
    print(f'Missing fields: {missing}')
    sys.exit(1)
print('Schema valid')
" < schema.json
```

#### 2. Test Registry Integration
```python
#!/usr/bin/env python3
"""Debug tool registration"""

import sys
sys.path.append('.')  # Add project root to path

from core.plugins.tool_registry import ToolRegistry
from core.plugins.cli_introspect import sniff

# Test individual tool introspection
tool_path = "tools/my_tool.py"
schema, desc, runner = sniff(tool_path, "python-cli")
print(f"Schema: {schema}")
print(f"Description: {desc}")
print(f"Runner: {runner}")

# Test full registry
registry = ToolRegistry(tools_dir="tools")
print(f"Registered tools: {list(registry.keys())}")

# Test specific tool
if "my_tool" in registry:
    print("Tool found in registry")
    try:
        result = registry["my_tool"](test_param="test_value")
        print(f"Test execution result: {result}")
    except Exception as e:
        print(f"Execution error: {e}")
```

#### 3. Log Analysis
Enable debug logging to trace discovery process:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Tool registry will output debug information
registry = ToolRegistry(tools_dir="tools")
```

---

## Conclusion

Building autodiscoverable fractalic tools requires adherence to specific technical contracts and patterns. The key success factors are:

1. **Proper schema implementation**: Support for `--fractalic-dump-schema` with valid JSON output
2. **Consistent parameter handling**: Match argparse definitions with schema specifications  
3. **Structured output**: Return JSON for better LLM integration
4. **Comprehensive testing**: Validate both schema generation and execution paths
5. **Error handling**: Graceful failure modes with informative error messages

Following these specifications enables seamless tool integration with the fractalic workflow system, providing powerful extensibility while maintaining simplicity for tool developers.

For additional examples and patterns, refer to the `fractalic_generator.py` implementation in the tools directory, which demonstrates advanced multi-tool patterns and comprehensive parameter handling.
