# OAuth Tests

This directory contains tests for the OAuth 2.0 implementation used in Fractalic's MCP integration.

## Test Files

- `test_auto_refresh.py` - Tests automatic token refresh functionality
- `test_existing_tokens.py` - Tests loading and using existing OAuth tokens  
- `test_oauth_tokens.py` - General OAuth token testing
- `test_replicate_direct.py` - Direct Bearer token authentication testing
- `test_current_client_refresh.py` - Current client token refresh testing
- `test_token_refresh.py` - Token refresh mechanism testing

## Running Tests

From the project root directory:

```bash
# Run all OAuth tests
python -m pytest tests/oauth/

# Run specific test
python tests/oauth/test_auto_refresh.py

# Run with verbose output
python -m pytest tests/oauth/ -v
```

## Test Coverage

These tests cover:
- OAuth 2.0 authorization flow
- Automatic token refresh (seamless, no user intervention)
- Token storage and retrieval
- MCP service integration with OAuth
- Error handling and fallback mechanisms
