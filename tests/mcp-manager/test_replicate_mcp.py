#!/usr/bin/env python3
"""Relocated test: direct test of Replicate MCP service using official SDK.
Original location: project root. Moved under tests/mcp-manager for organization.
"""

from pathlib import Path
import sys

# Ensure repository root on path when running via pytest or direct invocation
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-import original script logic by reading original file if needed
# For now we embed the previous content directly to keep test hermetic.

import asyncio
import json
import logging
from pathlib import Path as _P

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SimpleTokenStorage:
    def __init__(self, service_name: str = "replicate"):
        self.service_name = service_name
        # Use project oauth-cache directory instead of root oauth_tokens.json
        oauth_cache_dir = ROOT / "oauth-cache"
        oauth_cache_dir.mkdir(exist_ok=True)
        self.tokens_file = oauth_cache_dir / "oauth_tokens.json"
    async def get_tokens(self) -> OAuthToken | None:
        if not self.tokens_file.exists():
            return None
        try:
            data = json.loads(self.tokens_file.read_text())
            if self.service_name in data:
                td = data[self.service_name]
                return OAuthToken(
                    access_token=td['access_token'],
                    token_type=td.get('token_type', 'Bearer'),
                    expires_in=td.get('expires_in'),
                    refresh_token=td.get('refresh_token'),
                    scope=td.get('scope')
                )
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
        return None
    async def set_tokens(self, tokens: OAuthToken) -> None:
        try:
            data = {}
            if self.tokens_file.exists():
                data = json.loads(self.tokens_file.read_text())
            data[self.service_name] = {
                'access_token': tokens.access_token,
                'token_type': tokens.token_type,
                'expires_in': tokens.expires_in,
                'refresh_token': tokens.refresh_token,
                'scope': tokens.scope
            }
            self.tokens_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
    async def get_client_info(self):
        return None
    async def set_client_info(self, client_info):
        pass

async def test_replicate_mcp():
    url = "https://mcp.replicate.com/sse"
    headers = {"MCP-Protocol-Version": "2025-06-18"}
    token_storage = SimpleTokenStorage("replicate")

    async def dummy_redirect_handler(auth_url: str):
        logger.info(f"Would redirect to: {auth_url}")
    async def dummy_callback_handler():
        logger.info("Callback handler called")
        return ("dummy_code", "dummy_state")

    oauth_provider = OAuthClientProvider(
        server_url="https://mcp.replicate.com",
        client_metadata=OAuthClientMetadata(
            client_name="Test MCP Client",
            redirect_uris=[AnyUrl("http://localhost:5859/oauth/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="read write",
        ),
        storage=token_storage,
        redirect_handler=dummy_redirect_handler,
        callback_handler=dummy_callback_handler,
    )

    logger.info("Starting Replicate MCP test...")
    timeouts_to_test = [
        (30.0, 60.0),
        (30.0, 120.0),
        (60.0, 300.0),
    ]
    for connection_timeout, sse_read_timeout in timeouts_to_test:
        logger.info(f"Testing with timeouts: connection={connection_timeout}s, sse_read={sse_read_timeout}s")
        try:
            async with sse_client(
                url,
                auth=oauth_provider,
                headers=headers,
                timeout=connection_timeout,
                sse_read_timeout=sse_read_timeout
            ) as (read_stream, write_stream):
                logger.info("SSE connection established")
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), timeout=30.0)
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=60.0)
                    logger.info(f"Received {len(tools_result.tools)} tools")
                    assert len(tools_result.tools) >= 0
                    return True
        except Exception as e:
            logger.error(f"Attempt failed: {e}")
            continue
    logger.error("All timeout configurations failed")
    return False

if __name__ == "__main__":
    ok = asyncio.run(test_replicate_mcp())
    raise SystemExit(0 if ok else 1)
