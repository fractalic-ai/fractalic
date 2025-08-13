#!/usr/bin/env python3
"""
MCP Manager Compatibility Test
Tests that the SDK version maintains compatibility with existing Fractalic ecosystem
"""

import asyncio
import aiohttp
import json
import subprocess
import time
import sys
from pathlib import Path

class CompatibilityTest:
    def __init__(self, port=5859):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        
    async def test_cli_interface(self):
        """Test CLI commands work"""
        print("🧪 Testing CLI Interface...")
        
        # Test CLI help
        result = subprocess.run([
            sys.executable, "fractalic_mcp_manager_sdk.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CLI help works")
            # Check for expected commands
            expected_commands = ['serve', 'status', 'tools', 'start', 'stop', 'restart']
            help_text = result.stdout
            missing_commands = [cmd for cmd in expected_commands if cmd not in help_text]
            
            if not missing_commands:
                print("✅ All expected CLI commands present")
            else:
                print(f"❌ Missing CLI commands: {missing_commands}")
                return False
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
        
        return True
    
    async def test_api_endpoints(self):
        """Test REST API endpoints"""
        print("🧪 Testing API Endpoints...")
        
        expected_endpoints = [
            "/status",
            "/tools", 
            "/list_tools",
            "/call_tool"
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in expected_endpoints:
                try:
                    if endpoint == "/call_tool":
                        # POST endpoint
                        async with session.post(f"{self.base_url}{endpoint}", 
                                              json={"service": "test", "tool": "test"}) as resp:
                            if resp.status in [200, 400]:  # 400 is expected for test data
                                print(f"✅ {endpoint} responds")
                            else:
                                print(f"❌ {endpoint} unexpected status: {resp.status}")
                                return False
                    else:
                        # GET endpoint
                        async with session.get(f"{self.base_url}{endpoint}") as resp:
                            if resp.status == 200:
                                print(f"✅ {endpoint} responds")
                                
                                # Check response format
                                data = await resp.json()
                                if endpoint == "/status":
                                    if "services" in data and "total_services" in data:
                                        print(f"✅ {endpoint} has expected format")
                                    else:
                                        print(f"❌ {endpoint} missing expected fields")
                                        return False
                                        
                                elif endpoint in ["/tools", "/list_tools"]:
                                    if isinstance(data, dict):
                                        print(f"✅ {endpoint} has expected format")
                                    else:
                                        print(f"❌ {endpoint} wrong format")
                                        return False
                            else:
                                print(f"❌ {endpoint} status: {resp.status}")
                                return False
                                
                except Exception as e:
                    print(f"❌ {endpoint} failed: {e}")
                    return False
        
        return True
    
    async def test_service_management(self):
        """Test service start/stop/restart"""
        print("🧪 Testing Service Management...")
        
        async with aiohttp.ClientSession() as session:
            # Get list of services first
            async with session.get(f"{self.base_url}/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    services = list(data.get('services', {}).keys())
                    
                    if not services:
                        print("ℹ️  No services configured for testing")
                        return True
                    
                    # Test service management with first service
                    test_service = services[0]
                    print(f"Testing with service: {test_service}")
                    
                    # Test start
                    async with session.post(f"{self.base_url}/start/{test_service}") as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            print(f"✅ Start endpoint responds: {result}")
                        else:
                            print(f"❌ Start endpoint failed: {resp.status}")
                            return False
                    
                    # Test stop
                    async with session.post(f"{self.base_url}/stop/{test_service}") as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            print(f"✅ Stop endpoint responds: {result}")
                        else:
                            print(f"❌ Stop endpoint failed: {resp.status}")
                            return False
                    
                    # Test restart
                    async with session.post(f"{self.base_url}/restart/{test_service}") as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            print(f"✅ Restart endpoint responds: {result}")
                        else:
                            print(f"❌ Restart endpoint failed: {resp.status}")
                            return False
                    
                else:
                    print(f"❌ Status endpoint failed: {resp.status}")
                    return False
        
        return True
    
    async def test_port_compatibility(self):
        """Test default port is 5859"""
        print("🧪 Testing Port Compatibility...")
        
        if self.port == 5859:
            print("✅ Using default port 5859 (Fractalic standard)")
            return True
        else:
            print(f"❌ Using port {self.port}, expected 5859")
            return False
    
    async def test_error_handling(self):
        """Test error responses are properly formatted"""
        print("🧪 Testing Error Handling...")
        
        async with aiohttp.ClientSession() as session:
            # Test invalid tool call
            async with session.post(f"{self.base_url}/call_tool", 
                                  json={"invalid": "data"}) as resp:
                if resp.status in [400, 500]:
                    data = await resp.json()
                    if "error" in data and "success" in data:
                        print("✅ Error responses have expected format")
                        return True
                    else:
                        print("❌ Error response missing expected fields")
                        return False
                else:
                    print(f"❌ Expected error status, got: {resp.status}")
                    return False
    
    async def run_all_tests(self):
        """Run all compatibility tests"""
        print("🧪 MCP Manager SDK Compatibility Test")
        print("=====================================")
        print()
        
        tests = [
            ("CLI Interface", self.test_cli_interface),
            ("API Endpoints", self.test_api_endpoints),
            ("Service Management", self.test_service_management),
            ("Port Compatibility", self.test_port_compatibility),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if await test_func():
                    passed += 1
                print()
            except Exception as e:
                print(f"❌ {test_name} test failed with exception: {e}")
                print()
        
        print("📊 Test Results")
        print("===============")
        print(f"Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("🎉 All compatibility tests passed!")
            print("✅ SDK version is compatible with Fractalic ecosystem")
            return True
        else:
            print("⚠️  Some compatibility tests failed")
            print("❌ SDK version needs fixes before deployment")
            return False

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MCP Manager SDK compatibility')
    parser.add_argument('--port', '-p', type=int, default=5859, 
                        help='Port to test against')
    parser.add_argument('--check-running', action='store_true',
                        help='Check if manager is already running')
    
    args = parser.parse_args()
    
    if args.check_running:
        # Test if manager is already running
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{args.port}/status") as resp:
                    if resp.status == 200:
                        print(f"✅ Manager is running on port {args.port}")
                    else:
                        print(f"❌ Manager returned status {resp.status}")
                        sys.exit(1)
        except Exception as e:
            print(f"❌ Manager not running or not accessible: {e}")
            sys.exit(1)
    else:
        # Run compatibility tests
        tester = CompatibilityTest(args.port)
        success = await tester.run_all_tests()
        
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
