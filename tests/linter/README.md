# Fractalic Linter Tests

This directory contains Fractalic document test cases for validating the linter system. All tests are `.md` files written in Fractalic syntax that demonstrate different validation scenarios.

## Test Files

### `test_linter_issues.md`
**Purpose**: Contains various syntax errors that should be caught by the linter
- Unknown operation types (`@unknown_operation`)
- YAML syntax errors (malformed parameters)
- Missing blank lines between operations
- Orphaned content after operations

**Expected Result**: Should fail linting with 4 specific errors

### `test_valid_syntax.md`
**Purpose**: Contains valid Fractalic syntax that should pass all linting checks
- Proper YAML formatting in operation parameters
- Valid operation types (`@llm`, `@shell`, `@goto`, `@return`)
- Correct document structure and flow
- Appropriate blank line spacing
- Valid block references and IDs
- Self-contained operations without external dependencies

**Expected Result**: Should pass linting and execute successfully

### `test_yaml_edge_cases.md`
**Purpose**: Tests edge cases in YAML parsing and validation
- Complex nested YAML structures
- Special characters and escaping
- Multi-line strings with various formats
- Parameter combinations and edge cases

**Expected Result**: Should pass linting validation

## Running Linter Tests

All linter tests are Fractalic documents that can be executed directly:

```bash
# Test invalid syntax (should show 4 linting errors)
python fractalic.py tests/linter/test_linter_issues.md

# Test valid syntax (should pass linting and execute)
python fractalic.py tests/linter/test_valid_syntax.md

# Test YAML edge cases (should pass linting)
python fractalic.py tests/linter/test_yaml_edge_cases.md
```

## Test Philosophy

- **Pure Fractalic**: Tests are written as Fractalic documents, not Python scripts
- **Real-world validation**: Tests use actual Fractalic syntax and operations
- **Clear expectations**: Each test has a documented expected behavior
- **Self-documenting**: The `.md` files themselves show what's being tested

## Adding New Linter Tests

1. Create a new `.md` file with Fractalic syntax
2. Document the test purpose and expected result in this README
3. Test by running: `python fractalic.py tests/linter/your_test.md`
