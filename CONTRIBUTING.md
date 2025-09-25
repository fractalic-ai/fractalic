# Contributing to Fractalic

Thank you for your interest in contributing to Fractalic! This guide will help you get started with contributing to the Agentic Development Environment that transforms documents into AI-native applications.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Workflow](#contributing-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Submitting Changes](#submitting-changes)
- [Community and Support](#community-and-support)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [security@fractalic.ai](mailto:security@fractalic.ai).

## Getting Started

### What can I contribute?

- **Bug fixes**: Help us identify and fix issues
- **Feature development**: Implement new functionality
- **Documentation**: Improve our docs, tutorials, and examples
- **Testing**: Add test coverage and improve test infrastructure
- **MCP integrations**: Build new Model Context Protocol servers and tools
- **Examples**: Create sample workflows and use cases

### Before You Start

1. Check our [existing issues](https://github.com/fractalic-ai/fractalic/issues) to see what needs help
2. Join our [Discord community](https://discord.gg/DHbnvxAT) to discuss ideas

## Development Setup

### Prerequisites

- **Python 3.11+** (required)
- **Node.js 16+** (for frontend development)
- **Git**
- **Docker** (optional, for containerized development)

### Quick Setup

The fastest way to get started is using our automated setup script:

```bash
git clone https://github.com/fractalic-ai/fractalic.git
cd fractalic
./local-dev-setup.sh
```

This script will:
- Set up a Python virtual environment
- Install all backend dependencies
- Clone and set up the Fractalic-UI frontend
- Start both backend and frontend servers

### Manual Setup

If you prefer manual setup or need more control:

#### Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy configuration files
cp settings.toml.sample settings.toml
cp mcp_servers.json.sample mcp_servers.json

# Run backend server
python fractalic.py
```

#### Frontend Setup (Optional)

```bash
# Clone the UI repository
git clone https://github.com/fractalic-ai/fractalic-ui.git ../fractalic-ui
cd ../fractalic-ui

# Install dependencies
npm install

# Start development server
npm run dev
```

### Verify Installation

Test your setup by running:

```bash
# Run basic tests
pytest

# Test CLI functionality
python fractalic.py --help

# Run a simple example
python fractalic.py tutorials/01_Basics/hello-world/hello_world.md
```

## Contributing Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/fractalic.git
cd fractalic
git remote add upstream https://github.com/fractalic-ai/fractalic.git
```

### 2. Create a Branch

```bash
# Create a descriptive branch name
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

- Write clear, focused commits
- Follow our coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run the full test suite
pytest

# Run specific tests
pytest tests/test_specific_feature.py

# Run linting
black --check .
flake8 .
mypy .
```

### 5. Commit and Push

```bash
# Add your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add support for new LLM provider"

# Push to your fork
git push origin feature/your-feature-name
```

### 6. Submit a Pull Request

1. Go to GitHub and create a pull request
2. Describe your changes clearly
3. Link any related issues
4. Follow our review process

## Coding Standards

### Python Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **Flake8**: Linting
- **MyPy**: Type checking

Configuration is in `pyproject.toml`. Run formatting before committing:

```bash
# Auto-format code
black .

# Check formatting
black --check .

# Run linter
flake8 .

# Run type checker
mypy .
```

### Code Organization

- **Core logic**: `/core` directory
- **LLM providers**: `/core/llm/providers`
- **Operations**: `/core/operations`
- **Plugins**: `/core/plugins`
- **Tests**: `/tests` directory

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Files/modules**: `snake_case.py`

### Documentation Strings

Use Google-style docstrings:

```python
def process_markdown(content: str, config: Dict[str, Any]) -> ProcessedDocument:
    """Process a Markdown document into a Fractalic workflow.
    
    Args:
        content: Raw markdown content to process
        config: Configuration parameters for processing
        
    Returns:
        Processed document ready for execution
        
    Raises:
        ValidationError: If the document format is invalid
    """
```

## Testing Guidelines

### Test Structure

We use pytest for testing. Tests are organized by component:

```
tests/
├── ast/              # AST parsing tests
├── linter/           # Linter validation tests (as Fractalic docs)
├── mcp-manager/      # MCP integration tests
├── oauth/            # OAuth flow tests
└── test_*.py         # Unit tests
```

### Writing Tests

1. **Unit tests**: Test individual functions/classes
2. **Integration tests**: Test component interactions
3. **End-to-end tests**: Test complete workflows
4. **Fractalic document tests**: Real-world usage validation

Example test:

```python
import pytest
from core.operations import LLMOperation

def test_llm_operation_basic():
    """Test basic LLM operation functionality."""
    op = LLMOperation(prompt="Test prompt", model="gpt-3.5-turbo")
    result = op.execute(context={})
    
    assert result.success
    assert isinstance(result.content, str)
    assert len(result.content) > 0

@pytest.mark.asyncio
async def test_async_operation():
    """Test asynchronous operation execution."""
    # Test async functionality
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core

# Run specific test categories
pytest tests/ast/
pytest tests/linter/

# Run tests matching pattern
pytest -k "test_llm"

# Run tests with verbose output
pytest -v
```

## Documentation Guidelines

We follow strict documentation standards to ensure consistency and clarity across all our documentation.

### Key Principles

- **Plain, precise language**: Avoid marketing speak
- **Imperative tone**: "Add an ID" not "You could add an ID"
- **Present tense**: "Function returns" not "Function will return"
- **Active voice**: Subject performs the action

### Documentation Types

1. **API documentation**: In-code docstrings
2. **User guides**: `/docs` directory
3. **Tutorials**: Step-by-step workflows
4. **Reference**: Complete feature documentation

### Markdown Standards

- Use proper heading hierarchy (H1 → H2 → H3)
- Include code blocks with syntax highlighting
- Add internal links with anchors
- Use bullet points for lists
- Include examples for complex concepts

### Documentation Structure

For substantial docs (>8 sections):

1. Title & Purpose
2. Table of Contents
3. Core Concepts
4. How It Works
5. Usage Patterns & Examples
6. Edge Cases / Pitfalls
7. Performance / Cost Control
8. Safety & Constraints
9. Quick Reference
10. Cross References

## Submitting Changes

### Pull Request Guidelines

#### PR Title Format

Use conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

#### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests and linting
2. **Code review**: Maintainers review your changes
3. **Testing**: Additional testing on different environments
4. **Merge**: Once approved, changes are merged

### After Your PR is Merged

1. Delete your feature branch
2. Update your local repository:
   ```bash
   git checkout main
   git pull upstream main
   ```
3. Consider contributing to related areas!

## Community and Support

### Getting Help

- **Discord**: Join our [community server](https://discord.gg/DHbnvxAT)
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Check our [docs](https://fractalic.ai/docs)

### Reporting Bugs

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Fractalic version
- Operating system
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- Problem description
- Proposed solution
- Alternative solutions considered
- Additional context

### Security Issues

For security vulnerabilities, email [security@fractalic.ai](mailto:security@fractalic.ai) instead of opening a public issue.

## Recognition

Contributors are recognized in:
- Release notes
- Contributors section in README
- Project documentation
- Community Discord

Thank you for contributing to Fractalic! Together, we're building the future of AI-native application development.

---

**Questions?** Join our [Discord](https://discord.gg/DHbnvxAT) or open an [Issue](https://github.com/fractalic-ai/fractalic/issues) for discussion.
