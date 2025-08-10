# Linter Tests

This folder contains test files for the Fractalic linter functionality.

## Test Files

### `test_linter_issues.md`
- Tests various linting error scenarios
- Contains intentional errors to verify linter catches them
- Includes: unknown operations, YAML syntax errors, orphaned content, missing blank lines

### `test_valid_syntax.md`
- Contains valid Fractalic syntax that should pass linting
- Useful for regression testing to ensure linter doesn't reject valid content

### `test_yaml_edge_cases.md`
- Tests edge cases in YAML parsing
- Multiline strings, complex structures, empty parameters

## Running Tests

To test the linter on these files:

```bash
# This should fail with linting errors
python fractalic.py tutorials/linter-tests/test_linter_issues.md

# This should pass without linting errors
python fractalic.py tutorials/linter-tests/test_valid_syntax.md
```

## Expected Behavior

The linter should:
1. Catch YAML syntax errors in operation blocks
2. Detect unknown operations 
3. Identify structural issues (missing blank lines, orphaned content)
4. Validate operation parameters against schema
5. Fail execution if critical errors are found
6. Display helpful error messages with line numbers and context
