#!/usr/bin/env python3
"""
Test Replicate Bearer token authentication directly with httpx
"""

import asyncio
import httpx
import json

async def test_bearer_token():
    """Test Bearer token authentication with Replicate SSE endpoint"""
    
    # Load token
    with open("oauth_tokens.json", "r") as f:
        tokens = json.load(f)
    
    access_token = tokens["replicate"]["access_token"]
    print(f"Testing Bearer token: {access_token[:20]}...")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("https://mcp.replicate.com/sse", headers=headers, timeout=10.0)
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            if response.status_code != 200:
                print(f"Error response: {response.text}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_bearer_token())
