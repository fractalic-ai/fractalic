#!/usr/bin/env python3
"""
Test OAuth flow for Replicate integration to verify our enhancements work correctly.
"""

import asyncio
import logging
from fractalic_mcp_manager_sdk_v2 import MCPSupervisorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_replicate_oauth():
    """Test that OAuth flow works for Replicate"""
    supervisor = MCPSupervisorV2()
    
    logger.info("Testing Replicate OAuth flow...")
    
    try:
        # This should trigger OAuth flow if authentication is required
        tools = await supervisor._get_tools_for_service_direct("replicate")
        
        if tools:
            logger.info(f"✅ SUCCESS: Got {len(tools)} tools from Replicate:")
            for i, tool in enumerate(tools[:3]):  # Show first 3 tools
                logger.info(f"  {i+1}. {tool['name']}: {tool['description'][:100]}...")
        else:
            logger.warning("❌ FAILED: No tools received from Replicate")
            
    except Exception as e:
        logger.error(f"❌ ERROR: Failed to get tools from Replicate: {e}")

if __name__ == "__main__":
    asyncio.run(test_replicate_oauth())
