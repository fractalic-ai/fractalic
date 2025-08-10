# Fractalic Test Suite

This directory contains all internal tests for the Fractalic interpreter. Tests are organized by type and purpose to maintain clarity and separation between user-facing examples and internal validation.

## Directory Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions  
├── linter/         # Linter validation tests and test cases
├── __init__.py     # Package initialization
└── run_tests.py    # Main test runner
```

## Test Categories

### Unit Tests (`unit/`)
Tests for individual components and functions:
- `test_enhanced_tracking.py` - Token tracking system tests
- `test_tool_filtering.py` - Tool filtering mechanism tests

### Integration Tests (`integration/`)
Tests for component interactions and end-to-end functionality:
- (Future integration tests will be added here)

### Linter Tests (`linter/`)
Validation tests for the Fractalic linter system:
- `test_linter.py` - Main linter test runner
- `test_linter_issues.md` - Document with intentional syntax errors
- `test_valid_syntax.md` - Document with valid Fractalic syntax
- `test_yaml_edge_cases.md` - YAML edge case testing
- `README.md` - Detailed linter test documentation

## Running Tests

### Run All Tests
```bash
python -m tests.run_tests
```

### Run Specific Test Types
```bash
python -m tests.run_tests unit         # Unit tests only
python -m tests.run_tests linter       # Linter tests only  
python -m tests.run_tests integration  # Integration tests only
```

### Run Individual Tests
```bash
# Run specific unit test
python -m pytest tests/unit/test_enhanced_tracking.py -v

# Run linter tests directly
python tests/linter/test_linter.py

# Test linter with individual files
python fractalic.py tests/linter/test_linter_issues.md
```

## Test Requirements

- **pytest** for unit and integration tests
- **Fractalic core modules** for linter tests
- Tests should be self-contained and not depend on external services
- Test artifacts (*.ctx, *.trc, call_tree.json) are automatically excluded from git

## Adding New Tests

1. **Unit tests**: Add to `tests/unit/` with `test_*.py` naming
2. **Integration tests**: Add to `tests/integration/` with `test_*.py` naming  
3. **Linter tests**: Add markdown files to `tests/linter/` and update `test_linter.py`

## Design Principles

- **Separation of Concerns**: Internal tests vs user tutorials
- **Clear Organization**: Tests grouped by functionality and scope
- **Easy Execution**: Simple commands to run any subset of tests
- **Clean Repository**: Test artifacts excluded from version control
- **Documentation**: Each test type has clear documentation and examples
