#!/usr/bin/env python3
"""
Fractalic Test Runner

Run all Fractalic internal tests.

Usage:
    python -m tests.run_tests           # Run all tests
    python -m tests.run_tests unit      # Run unit tests only  
    python -m tests.run_tests linter    # Run linter tests only
    python -m tests.run_tests integration # Run integration tests only
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests(test_type="all"):
    """Run tests based on type specified"""
    test_dir = Path(__file__).parent
    
    if test_type == "all":
        # Run all test types
        print("🧪 Running all Fractalic tests...")
        run_unit_tests()
        run_linter_tests()
        run_integration_tests()
    elif test_type == "unit":
        run_unit_tests()
    elif test_type == "linter":
        run_linter_tests()
    elif test_type == "integration":
        run_integration_tests()
    else:
        print(f"❌ Unknown test type: {test_type}")
        print("Available types: all, unit, linter, integration")
        sys.exit(1)

def run_unit_tests():
    """Run unit tests using pytest"""
    print("\n📋 Running unit tests...")
    unit_dir = Path(__file__).parent / "unit"
    if unit_dir.exists():
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(unit_dir), "-v"
        ])
        if result.returncode != 0:
            print("❌ Unit tests failed")
        else:
            print("✅ Unit tests passed")

def run_linter_tests():
    """Run linter tests using Fractalic documents"""
    print("\n🔍 Running linter tests...")
    linter_dir = Path(__file__).parent / "linter"
    
    # Test files and their expected outcomes
    test_cases = [
        ("test_linter_issues.md", True),      # Should have errors
        ("test_valid_syntax.md", False),     # Should pass
        ("test_yaml_edge_cases.md", False),  # Should pass
    ]
    
    all_passed = True
    
    for test_file, should_fail in test_cases:
        test_path = linter_dir / test_file
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue
            
        print(f"  Testing: {test_file}")
        
        # Run fractalic on the test file
        result = subprocess.run([
            sys.executable, "fractalic.py", str(test_path)
        ], capture_output=True, text=True)
        
        # Check if the result matches expectations
        if should_fail:
            if result.returncode != 0 and "linting errors" in result.stderr:
                print(f"    ✅ PASS: Found expected linting errors")
            else:
                print(f"    ❌ FAIL: Expected linting errors but test passed")
                all_passed = False
        else:
            if result.returncode == 0 or "No linting errors found" in result.stderr:
                print(f"    ✅ PASS: Valid syntax passed as expected")
            else:
                print(f"    ❌ FAIL: Valid syntax failed unexpectedly")
                print(f"    Error: {result.stderr[:200]}...")
                all_passed = False
    
    if all_passed:
        print("✅ All linter tests passed")
    else:
        print("❌ Some linter tests failed")

def run_integration_tests():
    """Run integration tests"""
    print("\n🔗 Running integration tests...")
    integration_dir = Path(__file__).parent / "integration"
    if integration_dir.exists() and any(integration_dir.iterdir()):
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(integration_dir), "-v"
        ])
        if result.returncode != 0:
            print("❌ Integration tests failed")
        else:
            print("✅ Integration tests passed")
    else:
        print("📝 No integration tests found")

if __name__ == "__main__":
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_tests(test_type)
