#!/usr/bin/env python3
"""
Standalone test script for the Fractalic linter.

This script tests the linter functionality without running full Fractalic execution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.linter import FractalicLinter, FractalicLintError
from core.ast_md.parser import schema_text

def test_linter():
    """Test the linter with various test files."""
    
    test_files = [
        ("test_linter_issues.md", True),      # Should have errors
        ("test_valid_syntax.md", False),     # Should pass
        ("test_yaml_edge_cases.md", False),  # Should pass
    ]
    
    base_dir = os.path.dirname(__file__)
    linter = FractalicLinter(schema_text)
    
    print("=== Fractalic Linter Test Suite ===\n")
    
    for test_file, should_have_errors in test_files:
        file_path = os.path.join(base_dir, test_file)
        print(f"Testing: {test_file}")
        print("-" * 50)
        
        try:
            errors = linter.lint_file(file_path)
            
            if should_have_errors:
                if linter.has_errors():
                    print(f"✓ PASS: Found expected errors ({len(errors)} total)")
                    linter.print_errors(test_file)
                else:
                    print(f"✗ FAIL: Expected errors but none found")
            else:
                if linter.has_errors():
                    print(f"✗ FAIL: Unexpected errors found")
                    linter.print_errors(test_file)
                else:
                    print(f"✓ PASS: No errors found as expected")
                    if linter.has_warnings():
                        print(f"  (Found {len([e for e in errors if e.severity == 'warning'])} warnings)")
        
        except Exception as e:
            print(f"✗ ERROR: Exception during linting: {e}")
        
        print("\n")

if __name__ == "__main__":
    test_linter()
