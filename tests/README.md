# Fractalic Test Suite

This directory contains test cases for the Fractalic interpreter. All tests are written as Fractalic documents (`.md` files) to ensure they validate real-world usage patterns.

## Directory Structure

```
tests/
├── linter/         # Linter validation tests using Fractalic documents
└── README.md       # This documentation
```

## Test Categories

### Linter Tests (`linter/`)
Validation tests for the Fractalic linter system using pure Fractalic documents:
- `test_linter_issues.md` - Document with intentional syntax errors
- `test_valid_syntax.md` - Document with valid Fractalic syntax  
- `test_yaml_edge_cases.md` - YAML edge case testing
- `README.md` - Detailed linter test documentation

**Philosophy**: Linter tests are Fractalic documents themselves, not Python scripts. This ensures tests validate real-world usage patterns.

## Running Tests

### Linter Tests
```bash
# Test invalid syntax (should show 4 linting errors)
python fractalic.py tests/linter/test_linter_issues.md

# Test valid syntax (should pass linting and execute)
python fractalic.py tests/linter/test_valid_syntax.md

# Test YAML edge cases (should pass linting)
python fractalic.py tests/linter/test_yaml_edge_cases.md
```

## Test Requirements

- Tests should be self-contained Fractalic documents
- No external dependencies or Python test frameworks
- Test artifacts (*.ctx, *.trc, call_tree.json) are automatically excluded from git
- Each test should have clear expected behavior documented

## Adding New Tests

1. **Linter tests**: Add markdown files to `tests/linter/` and update the linter README
2. **Other tests**: Create new subdirectories as needed, following the pattern of using Fractalic documents

## Design Principles

- **Pure Fractalic**: All tests are written in Fractalic syntax
- **Real-world Validation**: Tests use actual Fractalic operations and patterns
- **Self-documenting**: Test files show exactly what's being tested
- **Simple Execution**: Just run `python fractalic.py test_file.md`
- **Clean Repository**: Test artifacts excluded from version control
