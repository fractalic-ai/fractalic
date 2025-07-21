#!/usr/bin/env python3
"""Test script for enhanced token tracking"""

import sys
sys.path.append('/Users/marina/llexem-jan-25-deploy/llexem_deploy_2025/fractalic')

from core.llm.llm_client import LLMClient
from core.token_stats import token_stats
from core.config import Config
import toml

def test_enhanced_tracking():
    """Test the enhanced token tracking with a real LLM call"""
    print("Testing Enhanced Token Tracking...")
    
    # Load settings from file
    try:
        with open('settings.toml', 'r') as f:
            settings = toml.load(f)
    except FileNotFoundError:
        print("settings.toml not found.")
        return
    
    # Setup Config like fractalic.py does
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
    
    # Initialize LLM client
    llm_client = LLMClient(
        model=model,
        provider=provider
    )
    
    # Test message that might use available tools
    messages = [
        {"role": "user", "content": "What is 2+2? Use any available calculation tools if you have them."}
    ]
    
    # Make LLM call
    print("Making LLM call with tools...")
    
    # Use llm_call with messages and operation_params
    operation_params = {
        'tools': 'all',  # Use all available tools from registry
        '_source_file': 'test_enhanced_tracking.py'
    }
    
    response = llm_client.llm_call(
        prompt_text="",  # Empty since we're using messages
        messages=messages,
        operation_params=operation_params
    )
    
    print(f"Response: {response}")
    
    # Print token usage summary
    print("\n" + "="*50)
    print("ENHANCED TOKEN TRACKING RESULTS:")
    token_stats.print_session_summary()
    
    return response

if __name__ == "__main__":
    test_enhanced_tracking()
