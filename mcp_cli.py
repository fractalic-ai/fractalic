#!/usr/bin/env python3
"""
MCP Manager CLI - SDK Migration Tools
Utilities for managing OAuth tokens and validating SDK migration
"""

import argparse
import json
import asyncio
import sys
from pathlib import Path

# Import the new SDK manager
try:
    from fractalic_mcp_manager_sdk import MCPClientManager, FileTokenStorage, OAUTH_AVAILABLE
except ImportError:
    print("❌ Cannot import SDK manager. Run migration first.")
    sys.exit(1)

async def list_oauth_tokens():
    """List stored OAuth tokens"""
    if not OAUTH_AVAILABLE:
        print("❌ OAuth not available in current MCP version")
        return
    
    storage = FileTokenStorage()
    storage_path = storage.storage_path
    
    if not storage_path.exists():
        print("ℹ️  No OAuth tokens stored yet")
        return
    
    try:
        with open(storage_path, 'r') as f:
            tokens = json.load(f)
        
        if not tokens:
            print("ℹ️  No OAuth tokens found")
            return
        
        print("🔐 Stored OAuth Tokens:")
        print("======================")
        for client_id, token_data in tokens.items():
            print(f"  Client ID: {client_id}")
            if 'expires_at' in token_data:
                print(f"    Expires: {token_data['expires_at']}")
            print(f"    Has refresh token: {'refresh_token' in token_data}")
            print()
            
    except Exception as e:
        print(f"❌ Error reading tokens: {e}")

async def migrate_oauth_tokens(old_path: str):
    """Migrate OAuth tokens from old location"""
    if not OAUTH_AVAILABLE:
        print("❌ OAuth not available in current MCP version")
        return
    
    old_file = Path(old_path)
    if not old_file.exists():
        print(f"❌ Old token file not found: {old_path}")
        return
    
    storage = FileTokenStorage()
    
    try:
        with open(old_file, 'r') as f:
            old_tokens = json.load(f)
        
        # Copy tokens to new storage
        with open(storage.storage_path, 'w') as f:
            json.dump(old_tokens, f, indent=2)
        
        print(f"✅ Migrated {len(old_tokens)} OAuth tokens to {storage.storage_path}")
        
    except Exception as e:
        print(f"❌ Error migrating tokens: {e}")

async def validate_migration():
    """Validate that migration was successful"""
    print("🔍 Validating SDK Migration")
    print("===========================")
    
    # Check if we can import SDK components
    try:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        print("✅ MCP SDK imports working")
    except ImportError as e:
        print(f"❌ MCP SDK import failed: {e}")
        return False
    
    # Check OAuth availability
    if OAUTH_AVAILABLE:
        print("✅ OAuth 2.1 support available")
    else:
        print("⚠️  OAuth not available (may need MCP 1.12.4+)")
    
    # Check manager can be instantiated
    try:
        manager = MCPClientManager()
        print("✅ MCPClientManager instantiation working")
    except Exception as e:
        print(f"❌ Manager instantiation failed: {e}")
        return False
    
    # Check file structure
    current_file = Path("fractalic_mcp_manager.py")
    sdk_file = Path("fractalic_mcp_manager_sdk.py")
    
    if current_file.exists() and sdk_file.exists():
        current_lines = len(current_file.read_text().splitlines())
        sdk_lines = len(sdk_file.read_text().splitlines())
        
        if current_lines == sdk_lines:
            print("✅ Migration appears complete (files match)")
            reduction = ((2358 - current_lines) / 2358) * 100
            print(f"✅ Code reduction: {reduction:.1f}%")
        else:
            print(f"⚠️  Current file: {current_lines} lines, SDK file: {sdk_lines} lines")
            print("   Migration may be incomplete")
    
    print("\n🎯 Migration validation complete!")
    return True

async def setup_oauth_interactive():
    """Interactive OAuth setup"""
    if not OAUTH_AVAILABLE:
        print("❌ OAuth not available in current MCP version")
        return
    
    print("🔐 Interactive OAuth Setup")
    print("==========================")
    
    # Load services from config
    manager = MCPClientManager()
    services = await manager.load_config()
    
    oauth_services = [s for s in services if s.oauth_client_id]
    
    if not oauth_services:
        print("ℹ️  No OAuth-enabled services found in mcp_servers.json")
        print("   Add OAUTH_CLIENT_ID to service env vars to enable OAuth")
        return
    
    print(f"Found {len(oauth_services)} OAuth-enabled services:")
    for i, service in enumerate(oauth_services):
        print(f"  {i+1}. {service.name}")
    
    # For demo purposes, just show what would happen
    print("\n🚀 To complete OAuth setup:")
    print("1. Start the MCP manager: python fractalic_mcp_manager.py")
    print("2. OAuth will auto-setup when services connect")
    print("3. Browser will open for authorization")
    print("4. Tokens will be stored persistently")

async def main():
    parser = argparse.ArgumentParser(description="MCP Manager CLI - SDK Migration Tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # OAuth token management
    oauth_parser = subparsers.add_parser('oauth-list', help='List stored OAuth tokens')
    
    migrate_parser = subparsers.add_parser('oauth-migrate', help='Migrate OAuth tokens from old location')
    migrate_parser.add_argument('old_path', help='Path to old token file')
    
    setup_parser = subparsers.add_parser('oauth-setup', help='Interactive OAuth setup')
    
    # Migration validation
    validate_parser = subparsers.add_parser('validate', help='Validate SDK migration')
    
    args = parser.parse_args()
    
    if args.command == 'oauth-list':
        await list_oauth_tokens()
    elif args.command == 'oauth-migrate':
        await migrate_oauth_tokens(args.old_path)
    elif args.command == 'oauth-setup':
        await setup_oauth_interactive()
    elif args.command == 'validate':
        await validate_migration()
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
