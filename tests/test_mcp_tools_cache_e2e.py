import asyncio
import time
import json
import contextlib
import sys
from pathlib import Path
from typing import Dict, Any

import aiohttp
import pytest

# Ensure repository root (containing fractalic_mcp_manager_sdk_v2.py) is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the serve coroutine directly from the MCP manager v2 file
from fractalic_mcp_manager_sdk_v2 import serve

CACHE_PORT = 5987  # Use a non-default port to avoid collisions
BASE_URL = f"http://127.0.0.1:{CACHE_PORT}"

@pytest.mark.asyncio
async def test_tools_cache_end_to_end():
    """End-to-end test:
    1. Start server (serve coroutine) on an alternate port.
    2. Wait for /status to report services.
    3. For each service, fetch /tools/{name} once (cold) and record latency & count.
    4. Fetch again (warm) and assert:
       - Tool count is stable.
       - Warm latency <= cold latency + 50ms tolerance.
    5. Ensure at least one service shows a clear speedup (cache benefit).
    NOTE: Replicate service is skipped until OAuth flow succeeds (currently 401 loops).
    """
    server_task = asyncio.create_task(serve(port=CACHE_PORT, host='127.0.0.1', disable_signals=True))
    try:
        async with aiohttp.ClientSession() as sess:
            # Wait for status readiness
            status_data: Dict[str, Any] = {}
            for _ in range(40):  # up to ~8s
                await asyncio.sleep(0.2)
                try:
                    async with sess.get(f"{BASE_URL}/status?include_tools_info=true") as r:
                        if r.status != 200:
                            continue
                        status_data = await r.json()
                    if status_data.get('services'):
                        break
                except Exception:
                    continue
            assert status_data.get('services'), "Status endpoint did not return services in time"

            services = list(status_data['services'].keys())
            # Skip disabled services & replicate (OAuth pending)
            services = [s for s in services if s.lower() != 'replicate']

            timings_first: Dict[str, float] = {}
            timings_second: Dict[str, float] = {}
            tool_counts: Dict[str, int] = {}

            # First (cold) fetch
            for svc in services:
                t0 = time.perf_counter()
                async with sess.get(f"{BASE_URL}/tools/{svc}") as r:
                    body = await r.json()
                dt_ms = (time.perf_counter() - t0) * 1000
                count = len(body.get('tools', [])) if isinstance(body, dict) and 'tools' in body else 0
                timings_first[svc] = dt_ms
                tool_counts[svc] = count

            # Second (warm) fetch
            for svc in services:
                t0 = time.perf_counter()
                async with sess.get(f"{BASE_URL}/tools/{svc}") as r:
                    body = await r.json()
                dt_ms = (time.perf_counter() - t0) * 1000
                timings_second[svc] = dt_ms
                # Assertions: counts match & latency improved or within tolerance
                assert len(body.get('tools', [])) == tool_counts[svc], f"Tool count changed for {svc}"
                assert dt_ms <= timings_first[svc] + 50.0, (
                    f"Cache did not reduce latency for {svc}: cold={timings_first[svc]:.1f}ms warm={dt_ms:.1f}ms"
                )

            # At least one service should show a clear improvement (>20% faster or >50ms saved if cold >100ms)
            improved = False
            for svc in services:
                cold = timings_first[svc]
                warm = timings_second[svc]
                if cold > 100 and (cold - warm) > 50:
                    improved = True
                    break
                if warm < cold * 0.8:
                    improved = True
                    break
            assert improved, f"No clear cache speedup observed. First: {timings_first} Second: {timings_second}"

            # Emit diagnostic summary for visibility
            summary = {
                'timings_first_ms': timings_first,
                'timings_second_ms': timings_second,
                'tool_counts': tool_counts,
            }
            print("\nCACHE_E2E_SUMMARY:\n" + json.dumps(summary, indent=2))
    finally:
        server_task.cancel()
        # Explicitly suppress asyncio.CancelledError (may subclass BaseException depending on Python version)
        import asyncio as _asyncio
        with contextlib.suppress(Exception, _asyncio.CancelledError):
            await server_task
