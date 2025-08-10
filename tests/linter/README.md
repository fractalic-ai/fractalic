# Fractalic Linter Tests

This directory contains test cases for the Fractalic linter validation system.

## Test Files

### `test_linter_issues.md`
1. **test_linter_issues.md**: Contains various syntax errors that should be caught by the linter
   - Unknown operation types
   - YAML syntax errors
   - Missing blank lines between operations
   - Orphaned content after operations

### `test_valid_syntax.md`
2. **test_valid_syntax.md**: Contains valid Fractalic syntax that should pass all linting checks
   - Proper YAML formatting
   - Valid operation types (@llm, @shell, @goto, @return)
   - Correct document structure
   - Appropriate blank line spacing
   - Valid block references and flow control
   - Self-contained operations without external dependencies

### `test_yaml_edge_cases.md`
3. **test_yaml_edge_cases.md**: Tests edge cases in YAML parsing
   - Complex nested YAML structures
   - Special characters and escaping
   - Multi-line strings with various formats

## Running Tests

From the Fractalic root directory:

```bash
# Run all tests including linter tests
python -m tests.run_tests

# Run only linter tests
python -m tests.run_tests linter

# Run linter test directly
python tests/linter/test_linter.py

# Test individual files with fractalic
python fractalic.py tests/linter/test_linter_issues.md   # Should show errors
python fractalic.py tests/linter/test_valid_syntax.md    # Should pass
```
