#!/usr/bin/env python3

import asyncio
import json
import time
from fractalic_mcp_manager_sdk_v2 import MCPSupervisorV2

async def test_token_refresh():
    """Test the token refresh functionality directly."""
    print("🔄 Testing token refresh functionality...")
    
    # Check current token status
    with open('oauth_tokens.json', 'r') as f:
        tokens = json.load(f)
    
    replicate_token = tokens.get('replicate', {})
    obtained_at = replicate_token.get('obtained_at', 0)
    expires_in = replicate_token.get('expires_in', 0)
    current_time = time.time()
    elapsed = current_time - obtained_at
    remaining = expires_in - elapsed
    
    print(f"📊 Current token status:")
    print(f"   Obtained at: {time.ctime(obtained_at)}")
    print(f"   Elapsed: {elapsed:.1f}s")
    print(f"   Remaining: {remaining:.1f}s")
    print(f"   Status: {'EXPIRED' if remaining <= 0 else 'VALID'}")
    
    if 'refresh_token' in replicate_token:
        refresh_prefix = replicate_token['refresh_token'][:20] + '...'
        print(f"   Has refresh token: {refresh_prefix}")
    else:
        print("   ❌ No refresh token available")
        return
    
    # Create MCP manager and attempt token refresh
    manager = MCPSupervisorV2()
    
    print(f"\n🔄 Attempting to refresh tokens...")
    try:
        # Call the refresh method directly
        await manager._maybe_refresh_tokens('replicate')
        
        # Check if tokens were updated
        with open('oauth_tokens.json', 'r') as f:
            updated_tokens = json.load(f)
        
        updated_replicate = updated_tokens.get('replicate', {})
        new_obtained_at = updated_replicate.get('obtained_at', 0)
        
        if new_obtained_at > obtained_at:
            print(f"✅ Token refresh succeeded!")
            print(f"   New obtained_at: {time.ctime(new_obtained_at)}")
            new_elapsed = time.time() - new_obtained_at
            new_remaining = updated_replicate.get('expires_in', 0) - new_elapsed
            print(f"   New remaining: {new_remaining:.1f}s")
        else:
            print(f"❌ Token refresh failed - no new tokens")
    
    except Exception as e:
        print(f"❌ Token refresh error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_token_refresh())
