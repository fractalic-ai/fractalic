#!/usr/bin/env python3
"""
HTTP API Handlers for FastMCP Manager
Maintains exact API schemas for frontend compatibility
"""

import json
import logging
import time
from aiohttp import web

from .fastmcp_manager import FastMCPManager

logger = logging.getLogger(__name__)

# Global manager instance
manager: FastMCPManager = None

def init_manager():
    """Initialize global manager instance"""
    global manager
    if manager is None:
        manager = FastMCPManager()

# ===== CORE MCP OPERATIONS (9 endpoints) =====

async def health_handler(request):
    """GET /health - Fast health check"""
    init_manager()
    return web.json_response({"status": "ok", "timestamp": time.time()})

async def status_handler(request):
    """GET /status - Simple health check (no service polling)"""
    init_manager()
    try:
        # Simple health check - just return manager is running
        # Frontend expects this to be instant
        import os
        status = {
            "status": "running",
            "api_responsive": True,
            "exit_code": None,
            "last_pid": os.getpid()
        }
        return web.json_response(status)
    except Exception as e:
        logger.error(f"Status error: {e}")
        return web.json_response({
            "status": "error",
            "api_responsive": False,
            "error": str(e)
        }, status=500)

async def complete_status_handler(request):
    """GET /status/complete - Complete MCP state with all data embedded"""
    init_manager()
    try:
        # Get or compute complete status with all embedded data
        complete_status = await manager.get_complete_status()
        
        # Add global metadata
        complete_status["complete_data_included"] = True
        complete_status["mcp_version"] = "2.0.1"
        complete_status["oauth_enabled"] = any(
            s.get("has_oauth", False) 
            for s in complete_status.get("services", {}).values()
        )
        
        return web.json_response(complete_status)
    except Exception as e:
        logger.error(f"Complete status error: {e}")
        # If JSON serialization error, try to convert problematic objects
        try:
            import json
            # Try to serialize and catch specific error
            json.dumps(complete_status)
        except TypeError as json_err:
            logger.error(f"JSON serialization error: {json_err}")
            # Return error without trying to serialize problematic data
            return web.json_response({
                "error": f"JSON serialization error: {str(json_err)}",
                "hint": "Some objects cannot be serialized to JSON"
            }, status=500)
        return web.json_response({"error": str(e)}, status=500)

async def list_tools_handler(request):
    """GET /list_tools - Fractalic compatibility endpoint (flat array format)"""
    init_manager()
    try:
        all_tools = await manager.get_all_tools()
        
        # Convert nested services format to flat array format for Fractalic compatibility
        flat_tools = []
        total_token_count = 0
        enabled_services = 0
        
        for service_name, service_data in all_tools.items():
            if 'error' in service_data:
                continue
                
            tools = service_data.get('tools', [])
            enabled_services += 1
            
            for tool in tools:
                # Create tool with service prefix as expected by Fractalic
                tool_with_service = {
                    **tool,
                    "name": f"{service_name}.{tool['name']}",
                    "service": service_name,
                    "original_name": tool['name']
                }
                flat_tools.append(tool_with_service)
            
            # Add service token count if available
            total_token_count += service_data.get('token_count', 0)
        
        # Return format matching original fractalic_mcp_manager.py
        response = {
            'tools': flat_tools,
            'count': len(flat_tools),
            'total_token_count': total_token_count,
            'services_count': enabled_services
        }
        return web.json_response(response)
    except Exception as e:
        logger.error(f"List tools error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def toggle_service_handler(request):
    """POST /toggle/{name} - Toggle service enabled/disabled"""
    init_manager()
    service_name = request.match_info['name']
    
    try:
        result = await manager.toggle_service(service_name)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Toggle service error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def get_all_tools_handler(request):
    """GET /tools - Get all tools from all services"""
    init_manager()
    try:
        all_tools = await manager.get_all_tools()
        return web.json_response(all_tools)
    except Exception as e:
        logger.error(f"Get all tools error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def get_tools_handler(request):
    """GET /tools/{name} - Get tools for specific service"""
    init_manager()
    service_name = request.match_info['name']
    
    try:
        tools = await manager.get_tools_for_service(service_name)
        return web.json_response({
            "service": service_name,
            "tools": tools,
            "count": len(tools)
        })
    except Exception as e:
        logger.error(f"Get tools error for {service_name}: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def get_capabilities_handler(request):
    """GET /capabilities/{name} - Get capabilities for service"""
    init_manager()
    service_name = request.match_info['name']
    
    try:
        capabilities = await manager.get_capabilities_for_service(service_name)
        return web.json_response({
            "service": service_name,
            "capabilities": capabilities
        })
    except Exception as e:
        logger.error(f"Get capabilities error for {service_name}: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def call_tool_handler(request):
    """POST /call/{service}/{tool} - Call a tool"""
    init_manager()
    service_name = request.match_info['service']
    tool_name = request.match_info['tool']
    
    try:
        data = await request.json()
        arguments = data.get('arguments', {})
        
        result = await manager.call_tool_for_service(service_name, tool_name, arguments)
        
        # Handle non-serializable objects in result
        try:
            return web.json_response(result)
        except TypeError as e:
            if "not JSON serializable" in str(e):
                # Convert non-serializable objects to strings
                safe_result = {
                    "success": result.get("success", False),
                    "result": {
                        "content": str(result.get("result", {}).get("content", "")),
                        "isError": result.get("result", {}).get("isError", False)
                    }
                }
                return web.json_response(safe_result)
            else:
                raise
    except Exception as e:
        logger.error(f"Call tool error: {e}")
        return web.json_response({"error": str(e)}, status=500)

# ===== PROMPTS & RESOURCES (4 endpoints) =====

async def list_prompts_handler(request):
    """GET /list_prompts - Get all prompts from all enabled services"""
    init_manager()
    try:
        import asyncio
        
        # Collect tasks for parallel execution
        tasks = []
        service_names = []
        
        for service_name, config in manager.service_configs.items():
            if config.enabled:
                tasks.append(manager.get_prompts_for_service(service_name))
                service_names.append(service_name)
        
        # Execute all tasks in parallel with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        services_response = {}
        for service_name, result in zip(service_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to get prompts for {service_name}: {result}")
                services_response[service_name] = {"error": str(result), "prompts": []}
            elif result:
                services_response[service_name] = {"prompts": result}
        
        return web.json_response(services_response)
    except Exception as e:
        logger.error(f"List prompts error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def get_prompt_handler(request):
    """POST /prompt/{service}/{prompt} - Get prompt content with arguments"""
    init_manager()
    service_name = request.match_info['service']
    prompt_name = request.match_info['prompt']
    
    try:
        data = await request.json()
        arguments = data.get('arguments', {})
        
        client = manager.create_fastmcp_client(service_name)
        if not client:
            return web.json_response({"error": f"Service {service_name} not available"}, status=404)
        
        async with client as c:
            result = await c.get_prompt(prompt_name, arguments)
            
            return web.json_response({
                "service": service_name,
                "prompt": prompt_name,
                "result": {
                    "messages": result.messages if hasattr(result, 'messages') else [],
                    "description": result.description if hasattr(result, 'description') else ""
                }
            })
            
    except Exception as e:
        logger.error(f"Get prompt error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def list_resources_handler(request):
    """GET /list_resources - Get all resources from all enabled services"""
    init_manager()
    try:
        import asyncio
        
        # Collect tasks for parallel execution
        tasks = []
        service_names = []
        
        for service_name, config in manager.service_configs.items():
            if config.enabled:
                tasks.append(manager.get_resources_for_service(service_name))
                service_names.append(service_name)
        
        # Execute all tasks in parallel with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        services_response = {}
        for service_name, result in zip(service_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to get resources for {service_name}: {result}")
                services_response[service_name] = {"error": str(result), "resources": []}
            elif result:
                services_response[service_name] = {"resources": result}
        
        return web.json_response(services_response)
    except Exception as e:
        logger.error(f"List resources error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def read_resource_handler(request):
    """POST /resource/{service}/read - Read resource content"""
    init_manager()
    service_name = request.match_info['service']
    
    try:
        data = await request.json()
        resource_uri = data.get('uri')
        
        if not resource_uri:
            return web.json_response({"error": "Resource URI is required"}, status=400)
        
        client = manager.create_fastmcp_client(service_name)
        if not client:
            return web.json_response({"error": f"Service {service_name} not available"}, status=404)
        
        async with client as c:
            result = await c.read_resource(resource_uri)
            
            return web.json_response({
                "service": service_name,
                "uri": resource_uri,
                "content": result.contents if hasattr(result, 'contents') else str(result)
            })
            
    except Exception as e:
        logger.error(f"Read resource error: {e}")
        return web.json_response({"error": str(e)}, status=500)

# ===== OAUTH MANAGEMENT (5 endpoints) =====

async def oauth_start_handler(request):
    """POST /oauth/start/{service} - Start OAuth flow"""
    init_manager()
    service_name = request.match_info['service']
    
    try:
        result = await manager.start_oauth_flow(service_name)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"OAuth start error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def oauth_reset_handler(request):
    """POST /oauth/reset/{service} - Reset OAuth tokens"""
    init_manager()
    service_name = request.match_info['service']
    
    try:
        result = await manager.reset_oauth_tokens(service_name)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"OAuth reset error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def oauth_authorize_handler(request):
    """POST /oauth/authorize/{service} - Authorize service"""
    init_manager()
    service_name = request.match_info['service']
    
    # For FastMCP, authorization is handled automatically in start_oauth_flow
    # This endpoint maintains API compatibility
    try:
        result = await manager.start_oauth_flow(service_name)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"OAuth authorize error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def oauth_status_all_handler(request):
    """GET /oauth/status - Get OAuth status for all services"""
    init_manager()
    try:
        status = await manager.get_oauth_status()
        return web.json_response(status)
    except Exception as e:
        logger.error(f"OAuth status all error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def oauth_status_service_handler(request):
    """GET /oauth/status/{service} - Get OAuth status for specific service"""
    init_manager()
    service_name = request.match_info['service']
    
    try:
        status = await manager.get_service_oauth_status(service_name)
        return web.json_response({
            "service": service_name,
            **status
        })
    except Exception as e:
        logger.error(f"OAuth status error for {service_name}: {e}")
        return web.json_response({"error": str(e)}, status=500)

# ===== SERVER MANAGEMENT (3 endpoints) =====

async def add_server_handler(request):
    """POST /add_server - Add new MCP server"""
    init_manager()
    try:
        data = await request.json()
        result = await manager.add_server(data)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Add server error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def delete_server_handler(request):
    """POST /delete_server - Remove MCP server"""
    init_manager()
    try:
        data = await request.json()
        service_name = data.get('name')
        
        if not service_name:
            return web.json_response({"error": "Server name is required"}, status=400)
        
        result = await manager.delete_server(service_name)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Delete server error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def kill_handler(request):
    """POST /kill - Shutdown MCP manager"""
    init_manager()
    try:
        # Graceful shutdown
        logger.info("Shutdown requested via API")
        
        # No cached clients to close - per-request approach
        
        return web.json_response({
            "success": True,
            "message": "MCP Manager shutting down"
        })
    except Exception as e:
        logger.error(f"Kill handler error: {e}")
        return web.json_response({"error": str(e)}, status=500)


def setup_routes(app):
    """Setup all API routes with CORS support"""
    import aiohttp_cors
    import time
    
    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Core MCP Operations
    cors.add(app.router.add_get('/health', health_handler))
    cors.add(app.router.add_get('/status', status_handler))
    cors.add(app.router.add_get('/status/complete', complete_status_handler))
    cors.add(app.router.add_get('/list_tools', list_tools_handler))
    cors.add(app.router.add_post('/toggle/{name}', toggle_service_handler))
    cors.add(app.router.add_get('/tools', get_all_tools_handler))
    cors.add(app.router.add_get('/tools/{name}', get_tools_handler))
    cors.add(app.router.add_get('/capabilities/{name}', get_capabilities_handler))
    cors.add(app.router.add_post('/call/{service}/{tool}', call_tool_handler))
    
    # Prompts & Resources  
    cors.add(app.router.add_get('/list_prompts', list_prompts_handler))
    cors.add(app.router.add_post('/prompt/{service}/{prompt}', get_prompt_handler))
    cors.add(app.router.add_get('/list_resources', list_resources_handler))
    cors.add(app.router.add_post('/resource/{service}/read', read_resource_handler))
    
    # OAuth Management
    cors.add(app.router.add_post('/oauth/start/{service}', oauth_start_handler))
    cors.add(app.router.add_post('/oauth/reset/{service}', oauth_reset_handler))
    cors.add(app.router.add_post('/oauth/authorize/{service}', oauth_authorize_handler))
    cors.add(app.router.add_get('/oauth/status', oauth_status_all_handler))
    cors.add(app.router.add_get('/oauth/status/{service}', oauth_status_service_handler))
    
    # Server Management
    cors.add(app.router.add_post('/add_server', add_server_handler))
    cors.add(app.router.add_post('/delete_server', delete_server_handler))
    cors.add(app.router.add_post('/kill', kill_handler))