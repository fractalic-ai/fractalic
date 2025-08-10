#!/usr/bin/env python3
"""Test script to verify tool filtering and token calculation accuracy"""

import sys
sys.path.append('/Users/marina/llexem-jan-25-deploy/llexem_deploy_2025/fractalic')

from core.llm.llm_client import LLMClient
from core.token_stats import token_stats
from core.config import Config
import toml

def test_tool_filtering():
    """Test that token calculation only considers actually specified tools"""
    print("Testing Tool Filtering and Token Calculation...")
    
    # Load settings from file
    try:
        with open('settings.toml', 'r') as f:
            settings = toml.load(f)
    except FileNotFoundError:
        print("settings.toml not found.")
        return
    
    # Setup Config
    provider = 'openai'
    api_key = settings.get('settings', {}).get('openai', {}).get('apiKey', None)
    model = settings.get('settings', {}).get('openai', {}).get('model', 'gpt-4o')
    
    if not api_key:
        print("No API key found in settings.toml")
        return
    
    # Configure global config
    Config.TOML_SETTINGS = settings
    Config.LLM_PROVIDER = provider
    Config.API_KEY = api_key
    Config.MODEL = model
    
    # Test 1: Call with ALL tools
    print("\n=== TEST 1: LLM call with ALL tools ===")
    token_stats.reset()  # Clear previous stats
    
    llm_client = LLMClient(model=model, provider=provider)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    
    response1 = llm_client.llm_call(
        prompt_text="",
        messages=messages,
        operation_params={'tools': 'all', '_source_file': 'test_all_tools.py'}
    )
    print("Response received for ALL tools test")
    token_stats.print_session_summary()
    
    # Test 2: Call with SPECIFIC tool only  
    print("\n=== TEST 2: LLM call with SPECIFIC tool (fractalic_run) ===")
    token_stats.reset()  # Clear previous stats
    
    response2 = llm_client.llm_call(
        prompt_text="",
        messages=messages,
        operation_params={'tools': 'fractalic_run', '_source_file': 'test_specific_tool.py'}
    )
    print("Response received for SPECIFIC tool test")
    token_stats.print_session_summary()
    
    # Test 3: Call with NO tools
    print("\n=== TEST 3: LLM call with NO tools ===")
    token_stats.reset()  # Clear previous stats
    
    response3 = llm_client.llm_call(
        prompt_text="",
        messages=messages,
        operation_params={'tools': 'none', '_source_file': 'test_no_tools.py'}
    )
    print("Response received for NO tools test")
    token_stats.print_session_summary()

if __name__ == "__main__":
    test_tool_filtering()
