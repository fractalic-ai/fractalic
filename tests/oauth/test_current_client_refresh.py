#!/usr/bin/env python3

import asyncio
import json
import httpx
from fractalic_mcp_manager_sdk_v2 import MCPSupervisorV2

async def test_refresh_with_current_client():
    """Test refresh using the current manager's client credentials instead of stored ones."""
    print("🔄 Testing token refresh with current manager client credentials...")
    
    manager = MCPSupervisorV2()
    provider = manager.oauth_providers.get('replicate')
    
    if not provider or not hasattr(provider, 'context'):
        print("❌ No provider or context found")
        return
        
    # Get the current client metadata from the provider
    if not hasattr(provider.context, 'client_metadata'):
        print("❌ No client metadata in provider context")
        return
        
    client_metadata = provider.context.client_metadata
    current_client_id = getattr(client_metadata, 'client_id', None)
    
    print(f"✅ Current manager client_id: {current_client_id}")
    
    # Load tokens
    storage = manager.token_storages.get('replicate')
    tokens = await storage.get_tokens()
    
    if not tokens or not tokens.refresh_token:
        print("❌ No refresh token available")
        return
        
    # Get discovery info
    server_url = provider.context.server_url
    discovery_url = server_url.rstrip('/') + '/.well-known/oauth-authorization-server'
    
    token_endpoint = None
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(discovery_url)
            if resp.status_code == 200:
                discovery_data = resp.json()
                token_endpoint = discovery_data.get('token_endpoint')
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            
    if not token_endpoint:
        token_endpoint = server_url.rstrip('/') + '/token'
    
    print(f"✅ Using token endpoint: {token_endpoint}")
    
    # Try refresh with CURRENT client ID (not stored one)
    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': tokens.refresh_token,
        'client_id': current_client_id,
    }
    
    print(f"🔄 Attempting refresh with current client_id...")
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(token_endpoint, data=payload)
            
            print(f"📡 Response status: {resp.status_code}")
            
            if resp.status_code == 200:
                jd = resp.json()
                print(f"✅ Refresh successful with current client_id!")
                
                new_access = jd.get('access_token')
                new_expires = jd.get('expires_in')
                
                if new_access:
                    print(f"   New access token: {new_access[:20]}...")
                    print(f"   Expires in: {new_expires}s")
                    
                    # Update the tokens
                    await storage.replace_access_token(new_access, new_expires, jd.get('scope'))
                    print(f"✅ Tokens updated successfully!")
                    
                    # Also update the client_info in the token file to match current client
                    with open('oauth_tokens.json', 'r') as f:
                        data = json.load(f)
                    
                    if 'replicate' in data:
                        # Update client_info to current client credentials
                        if hasattr(client_metadata, 'client_secret'):
                            client_secret = getattr(client_metadata, 'client_secret', None)
                        else:
                            client_secret = None
                            
                        data['replicate']['client_info']['client_id'] = current_client_id
                        if client_secret:
                            data['replicate']['client_info']['client_secret'] = client_secret
                        
                        with open('oauth_tokens.json', 'w') as f:
                            json.dump(data, f, indent=2)
                            
                        print(f"✅ Updated stored client_info to match current client")
                    
                    return True
                else:
                    print(f"❌ No access_token in response: {jd}")
            else:
                print(f"❌ Refresh failed: {resp.status_code}")
                try:
                    error_data = resp.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error text: {resp.text}")
                    
    except Exception as e:
        print(f"❌ Refresh request failed: {e}")
        
    return False

if __name__ == "__main__":
    asyncio.run(test_refresh_with_current_client())
