#!/usr/bin/env python3
"""
Debug OAuth token loading and authentication flow
"""

import asyncio
import logging
import json
from fractalic_mcp_manager_sdk_v2 import FileTokenStorage

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_tokens():
    """Debug token loading and format"""
    logger.info("Testing token storage...")
    
    # Test token loading
    storage = FileTokenStorage("oauth_tokens.json", "replicate")
    tokens = await storage.get_tokens()
    
    if tokens:
        logger.info(f"✅ Found tokens: {tokens}")
        
        # Check token format
        if 'created_at' in tokens:
            logger.info("✅ Enhanced token format detected")
        else:
            logger.warning("⚠️ Legacy token format - may need conversion")
            
    else:
        logger.warning("❌ No tokens found")
        
    # Check raw file content
    try:
        with open("oauth_tokens.json", "r") as f:
            raw_content = json.load(f)
            logger.info(f"Raw file content: {raw_content}")
    except Exception as e:
        logger.error(f"Failed to read raw file: {e}")

if __name__ == "__main__":
    asyncio.run(debug_tokens())
