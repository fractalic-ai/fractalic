#!/usr/bin/env python3
"""
Test OAuth flow with extended timeout for replicate service
"""

import asyncio
import aiohttp
import json
import sys

async def test_oauth_flow():
    """Test the OAuth flow with replicate service"""
    try:
        print("🔄 Testing OAuth flow for replicate service...")
        print("📋 This will:")
        print("   1. Request tools from replicate")
        print("   2. Trigger OAuth if needed (browser will open)")
        print("   3. Wait up to 5 minutes for authorization")
        print("   4. Show available tools after authorization")
        print()
        
        async with aiohttp.ClientSession() as session:
            print("🚀 Requesting replicate tools...")
            
            # Request tools - this should trigger OAuth if needed
            async with session.get('http://localhost:5859/tools/replicate') as response:
                if response.status == 200:
                    tools = await response.json()
                    print(f"✅ Success! Got {len(tools)} tools from replicate")
                    
                    # Show first few tools as confirmation
                    for i, tool in enumerate(tools[:3]):
                        print(f"   🔧 Tool {i+1}: {tool.get('name', 'unknown')}")
                    
                    if len(tools) > 3:
                        print(f"   ... and {len(tools) - 3} more tools")
                        
                    return True
                else:
                    print(f"❌ Failed: HTTP {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
                    return False
                    
    except asyncio.TimeoutError:
        print("⏰ Timeout - OAuth authorization took longer than expected")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 OAuth Timeout Test for Replicate Service")
    print("=" * 50)
    
    # Test if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:5859/services') as response:
                if response.status != 200:
                    print("❌ MCP SDK Manager not running on port 5859")
                    print("   Start it with: python fractalic_mcp_manager_sdk.py")
                    return False
    except:
        print("❌ Cannot connect to MCP SDK Manager on port 5859")
        print("   Start it with: python fractalic_mcp_manager_sdk.py")
        return False
    
    print("✅ MCP SDK Manager is running")
    print()
    
    # Run OAuth test
    success = await test_oauth_flow()
    
    if success:
        print("\n🎉 OAuth flow test completed successfully!")
        print("   The 5-minute timeout should provide enough time for authorization")
    else:
        print("\n❌ OAuth flow test failed")
        print("   Check the server logs for more details")
    
    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
