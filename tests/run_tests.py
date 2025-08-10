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
    """Run linter tests"""
    print("\n🔍 Running linter tests...")
    linter_dir = Path(__file__).parent / "linter"
    
    # Run the Python linter test if it exists
    linter_test_py = linter_dir / "test_linter.py"
    if linter_test_py.exists():
        result = subprocess.run([
            sys.executable, str(linter_test_py)
        ])
        if result.returncode != 0:
            print("❌ Linter tests failed")
        else:
            print("✅ Linter tests passed")

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
