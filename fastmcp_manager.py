#!/usr/bin/env python3
"""
FastMCP-based MCP Manager
Replaces 4656-line manual implementation with FastMCP client
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastmcp import Client
from fastmcp.client.auth import OAuth

from mcp_config import MCPConfigLoader, ServiceConfig
from status_cache import StatusCache

logger = logging.getLogger(__name__)

class FastMCPManager:
    """
    FastMCP-based MCP Manager
    Handles all MCP server connections using FastMCP clients
    """
    
    def __init__(self):
        self.config_loader = MCPConfigLoader()
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.service_states: Dict[str, str] = {}  # enabled/disabled
        
        # Initialize status cache with 60 second TTL
        self.cache = StatusCache(default_ttl=60.0)
        
        # Load configurations
        self.service_configs = self.config_loader.load_config()
        
        # Initialize service states
        for name, config in self.service_configs.items():
            self.service_states[name] = "enabled" if config.enabled else "disabled"
    
    
    def create_fastmcp_client(self, service_name: str) -> Optional[Client]:
        """Create FastMCP client for service (per-request, not cached)"""
        try:
            config = self.service_configs.get(service_name)
            if not config or not config.enabled:
                return None
            
            # Create client based on transport type
            if config.transport == 'stdio':
                # For stdio servers - create full MCP config dict
                command = config.spec.get('command')
                if not command:
                    logger.error(f"No command specified for stdio service {service_name}")
                    return None
                
                # Create MCP config dict in standard format
                mcp_config = {
                    "mcpServers": {
                        service_name: {
                            "command": command,
                            "args": config.spec.get('args', []),
                            "env": config.spec.get('env', {})
                        }
                    }
                }
                
                # Create stdio client with MCP config
                client = Client(mcp_config)
                
            else:
                # For HTTP/SSE servers
                url = config.spec.get('url')
                if not url:
                    logger.error(f"No URL specified for service {service_name}")
                    return None
                
                # For HTTP servers - try without OAuth first
                # If server has embedded tokens (long URL paths), no OAuth needed
                import re
                if re.search(r'/[A-Za-z0-9+/=]{50,}', url):
                    # URL contains embedded tokens - no OAuth
                    client = Client(url)
                else:
                    # Clean URLs likely need OAuth
                    client = Client(url, auth='oauth')
            
            # Don't cache stdio clients - return immediately for per-request usage
            return client
            
        except Exception as e:
            logger.error(f"Failed to create client for {service_name}: {e}")
            return None
    
    async def _poll_single_service(self, service_name: str, config: ServiceConfig) -> tuple[str, Dict[str, Any]]:
        """Poll a single service for status (for parallel execution)"""
        service_status = {
            "enabled": config.enabled,
            "transport": config.transport,
            "connected": False,
            "tools_count": 0,
            "error": None
        }
        
        if not config.enabled:
            return (service_name, service_status)
        
        try:
            client = self.create_fastmcp_client(service_name)
            if client:
                async with client as c:
                    # Set timeout for individual service
                    await asyncio.wait_for(c.ping(), timeout=5.0)
                    service_status["connected"] = True
                    
                    # Get tools count
                    tools = await asyncio.wait_for(c.list_tools(), timeout=5.0)
                    tools_list = tools.tools if hasattr(tools, 'tools') else tools
                    service_status["tools_count"] = len(tools_list) if tools_list else 0
                    
                    # Cache the tools while we have them
                    if tools_list:
                        formatted_tools = []
                        for tool in tools_list:
                            formatted_tools.append({
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else None
                            })
                        await self.cache.set_service_tools(service_name, formatted_tools)
                    
        except asyncio.TimeoutError:
            service_status["error"] = "Service timeout (5s)"
            logger.warning(f"Service {service_name} timed out")
        except Exception as e:
            service_status["error"] = str(e)
            logger.error(f"Service {service_name} error: {e}")
        
        return (service_name, service_status)
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services with caching and parallel processing"""
        # Try to get from cache first
        cached_status = await self.cache.get_all_services_status()
        if cached_status:
            logger.debug("Returning cached service status")
            return cached_status
        
        logger.info("Cache miss - polling all services in parallel")
        
        # Create tasks for parallel polling
        tasks = []
        for service_name, config in self.service_configs.items():
            tasks.append(self._poll_single_service(service_name, config))
        
        # Execute all polls in parallel
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        logger.info(f"Parallel polling completed in {elapsed:.2f}s")
        
        # Process results and update cache
        status = {
            "services": {},
            "total_enabled": 0,
            "total_disabled": 0
        }
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Polling error: {result}")
                continue
            
            service_name, service_status = result
            status["services"][service_name] = service_status
            
            # Update counters
            if service_status["enabled"]:
                status["total_enabled"] += 1
            else:
                status["total_disabled"] += 1
            
            # Cache individual service status
            await self.cache.set_service_status(service_name, service_status)
        
        return status
    
    async def get_complete_status(self) -> Dict[str, Any]:
        """Get complete status by combining existing methods with all embedded data"""
        import traceback
        import json
        
        try:
            logger.info("=== Starting get_complete_status() DEBUG ===")
            
            # Get all data in parallel - leverage existing caching
            start_time = time.time()
            
            # For enabled services, get prompts and resources too
            enabled_services = [name for name, config in self.service_configs.items() if config.enabled]
            logger.debug(f"Enabled services: {enabled_services}")
            
            # Gather everything in parallel
            tasks = [
                self.get_service_status(),
                self.get_oauth_status(), 
                self.get_all_tools(),
            ]
            
            # TEMPORARILY DISABLED: Skip prompts and resources for performance testing
            # prompt_tasks = [self._safe_get_prompts(name) for name in enabled_services]
            # resource_tasks = [self._safe_get_resources(name) for name in enabled_services]
            prompt_tasks = []
            resource_tasks = []
            
            logger.debug("Getting basic status, oauth status, all tools...")
            basic_status, oauth_status, all_tools = await asyncio.gather(*tasks)
            logger.debug(f"Basic data gathered - services: {len(basic_status['services'])}, oauth: {len(oauth_status)}, tools: {len(all_tools)}")
            
            # Test JSON serialization of each component
            try:
                json.dumps(basic_status)
                logger.debug("basic_status: JSON serializable ✓")
            except Exception as e:
                logger.error(f"basic_status JSON ERROR: {e}")
                logger.error(f"basic_status traceback:\n{traceback.format_exc()}")
                raise
            
            try:
                json.dumps(oauth_status)
                logger.debug("oauth_status: JSON serializable ✓")
            except Exception as e:
                logger.error(f"oauth_status JSON ERROR: {e}")
                logger.error(f"oauth_status traceback:\n{traceback.format_exc()}")
                raise
                
            try:
                json.dumps(all_tools)
                logger.debug("all_tools: JSON serializable ✓")
            except Exception as e:
                logger.error(f"all_tools JSON ERROR: {e}")
                logger.error(f"all_tools traceback:\n{traceback.format_exc()}")
                raise
            
            # TEMPORARILY DISABLED: Skip prompts and resources gathering
            # if enabled_services:
            #     logger.debug(f"Getting prompts and resources for {len(enabled_services)} enabled services...")
            #     prompt_results, resource_results = await asyncio.gather(
            #         asyncio.gather(*prompt_tasks, return_exceptions=True),
            #         asyncio.gather(*resource_tasks, return_exceptions=True)
            #     )
            #     logger.debug(f"Prompts/resources gathered - prompt results: {len(prompt_results)}, resource results: {len(resource_results)}")
            # else:
            #     prompt_results, resource_results = [], []
            prompt_results, resource_results = [], []
            
            elapsed = time.time() - start_time
            logger.debug(f"Complete status gathered in {elapsed:.2f}s")
            
            # Build response in frontend schema format
            total_prompts = 0
            total_resources = 0
            
            complete_status = {
                "total_services": len(self.service_configs),
                "enabled_services": basic_status["total_enabled"], 
                "total_tools": sum(data.get("count", 0) for data in all_tools.values()),
                "services": {}
            }
            
            # Build service details with embedded data
            logger.debug("Building service details...")
            for service_name, service_info in basic_status["services"].items():
                logger.debug(f"Processing service: {service_name}")
                
                tools_data = all_tools.get(service_name, {})
                tools = tools_data.get("tools", [])
                
                # TEMPORARILY DISABLED: Set prompts/resources to null for performance testing
                prompts = None
                resources = None
                
                # if service_info["enabled"] and service_name in enabled_services:
                #     service_idx = enabled_services.index(service_name)
                #     
                #     # Get prompts result
                #     if (service_idx < len(prompt_results) and 
                #         not isinstance(prompt_results[service_idx], Exception)):
                #         prompts = prompt_results[service_idx] or []
                #     
                #     # Get resources result
                #     if (service_idx < len(resource_results) and 
                #         not isinstance(resource_results[service_idx], Exception)):
                #         resources = resource_results[service_idx] or []
                
                service_state = "connected" if service_info["connected"] else ("disabled" if not service_info["enabled"] else "error")
                complete_service = {
                    "status": service_state,
                    "connected": service_info["connected"],
                    "enabled": service_info["enabled"], 
                    "transport": service_info["transport"],
                    "has_oauth": service_name in oauth_status,
                    "tool_count": len(tools),
                    "prompt_count": 0,
                    "resource_count": 0,
                    "token_count": 0,  # Skip token counting for now
                    "tools": tools,
                    "prompts": prompts,
                    "resources": resources
                }
                
                # TEMPORARILY DISABLED: Skip totals counting
                # total_prompts += len(prompts)
                # total_resources += len(resources)
                
                # Add config info (ensure JSON serializable)
                logger.debug(f"Adding config info for {service_name}")
                config = self.service_configs.get(service_name)
                if config:
                    url = config.spec.get("url")
                    complete_service["url"] = str(url) if url else None
                    complete_service["command"] = config.spec.get("command")
                
                # Add OAuth details if present
                logger.debug(f"Adding OAuth details for {service_name}")
                if service_name in oauth_status:
                    oauth_data = oauth_status[service_name]
                    complete_service["oauth"] = oauth_data
                    
                    # Test JSON serialization of this service's oauth data
                    try:
                        json.dumps(oauth_data)
                        logger.debug(f"{service_name} oauth data: JSON serializable ✓")
                    except Exception as e:
                        logger.error(f"{service_name} oauth JSON ERROR: {e}")
                        logger.error(f"OAuth data type/content: {type(oauth_data)} = {oauth_data}")
                        logger.error(f"OAuth traceback:\n{traceback.format_exc()}")
                        raise
                
                # Test JSON serialization of complete service
                try:
                    json.dumps(complete_service)
                    logger.debug(f"{service_name} complete_service: JSON serializable ✓")
                except Exception as e:
                    logger.error(f"{service_name} complete_service JSON ERROR: {e}")
                    logger.error(f"Complete service traceback:\n{traceback.format_exc()}")
                    
                    # Deep inspection of problematic data
                    for key, value in complete_service.items():
                        try:
                            json.dumps(value)
                        except Exception as field_err:
                            logger.error(f"  Field '{key}' ({type(value)}): {field_err}")
                            logger.error(f"  Value: {value}")
                    raise
                
                complete_status["services"][service_name] = complete_service
                logger.debug(f"Service {service_name} processed successfully")
            
            # Add calculated totals
            complete_status["total_prompts"] = total_prompts
            complete_status["total_resources"] = total_resources
            complete_status["total_tokens"] = 0  # Skip for now
            
            # Final JSON serialization test
            logger.debug("Testing final complete_status JSON serialization...")
            try:
                json.dumps(complete_status)
                logger.debug("Final complete_status: JSON serializable ✓")
            except Exception as e:
                logger.error(f"Final complete_status JSON ERROR: {e}")
                logger.error(f"Final traceback:\n{traceback.format_exc()}")
                raise
            
            logger.info("=== get_complete_status() DEBUG COMPLETE ===")
            return complete_status
            
        except Exception as e:
            logger.error(f"=== COMPLETE STATUS FATAL ERROR ===")
            logger.error(f"Error: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.error(f"=== END FATAL ERROR ===")
            raise
    
    async def _safe_get_prompts(self, service_name: str) -> List[Dict[str, Any]]:
        """Safely get prompts with timeout and error handling"""
        try:
            return await asyncio.wait_for(self.get_prompts_for_service(service_name), timeout=5.0)
        except Exception as e:
            logger.debug(f"Failed to get prompts for {service_name}: {e}")
            return []
    
    async def _safe_get_resources(self, service_name: str) -> List[Dict[str, Any]]:
        """Safely get resources with timeout and error handling"""
        try:
            return await asyncio.wait_for(self.get_resources_for_service(service_name), timeout=5.0)
        except Exception as e:
            logger.debug(f"Failed to get resources for {service_name}: {e}")
            return []
    
    async def get_tools_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get tools for specific service with caching"""
        # Try cache first
        cached_tools = await self.cache.get_service_tools(service_name)
        if cached_tools is not None:
            logger.debug(f"Returning cached tools for {service_name}")
            return cached_tools
        
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                return []
            
            async with client as c:
                tools = await asyncio.wait_for(c.list_tools(), timeout=10.0)
                tools_list = tools.tools if hasattr(tools, 'tools') else tools
                
                if not tools_list:
                    return []
                
                # Format tools for API response
                formatted_tools = []
                for tool in tools_list:
                    formatted_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else None
                    })
                
                # Cache the result
                await self.cache.set_service_tools(service_name, formatted_tools)
                
                return formatted_tools
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting tools for {service_name}")
            raise
        except Exception as e:
            logger.error(f"Failed to get tools for {service_name}: {e}")
            raise
    
    async def get_all_tools(self) -> Dict[str, Any]:
        """Get tools from all enabled services with full collection caching"""
        # Try to get cached full collection first
        cached_all_tools = await self.cache.get_cached_data("all_tools", ttl=30.0)
        if cached_all_tools is not None:
            logger.info("Returning cached all_tools collection")
            return cached_all_tools
        
        logger.info("Building fresh all_tools collection")
        all_tools = {}
        
        # Get enabled services
        enabled_services = [(name, config) for name, config in self.service_configs.items() if config.enabled]
        
        if not enabled_services:
            return all_tools
        
        # Create tasks for parallel execution
        async def get_service_tools_safe(service_name: str) -> tuple[str, Dict[str, Any]]:
            try:
                tools = await self.get_tools_for_service(service_name)
                return service_name, {
                    "tools": tools,
                    "count": len(tools)
                }
            except Exception as e:
                logger.error(f"Error getting tools for {service_name}: {e}")
                return service_name, {
                    "error": str(e),
                    "tools": [],
                    "count": 0
                }
        
        # Execute all service tool fetching in parallel
        logger.info(f"Fetching tools from {len(enabled_services)} services in parallel")
        tasks = [get_service_tools_safe(name) for name, _ in enabled_services]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            service_name, service_data = result
            all_tools[service_name] = service_data
        
        # Cache the full collection for 30 seconds
        await self.cache.set_cached_data("all_tools", all_tools, ttl=30.0)
        logger.info(f"Built all_tools collection with {len(all_tools)} services")
        return all_tools
    
    async def call_tool_for_service(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool for specific service"""
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                raise Exception(f"No client available for service {service_name}")
            
            async with client as c:
                result = await c.call_tool(tool_name, arguments)
                
                # Extract content from FastMCP CallToolResult
                content_parts = []
                
                # Handle content list (TextContent, etc.)
                if result.content:
                    for item in result.content:
                        if hasattr(item, 'text'):
                            content_parts.append(item.text)
                        else:
                            content_parts.append(str(item))
                
                # Handle structured_content if available
                if result.structured_content:
                    content_parts.append(str(result.structured_content))
                
                # Handle data field if available
                if result.data:
                    content_parts.append(str(result.data))
                
                # Join all content parts
                content = "\n".join(content_parts) if content_parts else ""
                
                return {
                    "success": True,
                    "result": {
                        "content": content,
                        "isError": result.is_error
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} for {service_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_capabilities_for_service(self, service_name: str) -> Dict[str, Any]:
        """Get capabilities for specific service"""
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                return {}
            
            async with client as c:
                # FastMCP doesn't have explicit capabilities endpoint
                # Infer capabilities from available operations
                capabilities = {
                    "supportsTools": True,
                    "supportsPrompts": True,
                    "supportsResources": True
                }
                
                try:
                    tools = await c.list_tools()
                    capabilities["toolsCount"] = len(tools.tools) if hasattr(tools, 'tools') and tools.tools else 0
                except:
                    capabilities["supportsTools"] = False
                    capabilities["toolsCount"] = 0
                
                return capabilities
                
        except Exception as e:
            logger.error(f"Failed to get capabilities for {service_name}: {e}")
            return {"error": str(e)}
    
    async def get_prompts_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get prompts for specific service with timeout"""
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                return []
            
            async with client as c:
                prompts = await asyncio.wait_for(c.list_prompts(), timeout=5.0)
                prompts_list = prompts.prompts if hasattr(prompts, 'prompts') else prompts
                
                if not prompts_list:
                    return []
                
                # Format prompts for API response
                formatted_prompts = []
                for prompt in prompts_list:
                    formatted_prompts.append({
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": prompt.arguments if hasattr(prompt, 'arguments') else []
                    })
                
                return formatted_prompts
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting prompts for {service_name}")
            raise
        except Exception as e:
            logger.error(f"Failed to get prompts for {service_name}: {e}")
            raise
    
    async def get_resources_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get resources for specific service with timeout"""
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                return []
            
            async with client as c:
                resources = await asyncio.wait_for(c.list_resources(), timeout=5.0)
                resources_list = resources.resources if hasattr(resources, 'resources') else resources
                
                if not resources_list:
                    return []
                
                # Format resources for API response
                formatted_resources = []
                for resource in resources_list:
                    formatted_resources.append({
                        "uri": str(resource.uri),  # Convert AnyUrl to string for JSON serialization
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType if hasattr(resource, 'mimeType') else None
                    })
                
                return formatted_resources
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting resources for {service_name}")
            raise
        except Exception as e:
            logger.error(f"Failed to get resources for {service_name}: {e}")
            raise
    
    async def toggle_service(self, service_name: str) -> Dict[str, Any]:
        """Toggle service enabled/disabled status with cache update"""
        if service_name not in self.service_configs:
            return {"error": f"Service {service_name} not found"}
        
        # Toggle state
        current_state = self.service_states.get(service_name, "enabled")
        new_state = "disabled" if current_state == "enabled" else "enabled"
        
        self.service_states[service_name] = new_state
        self.service_configs[service_name].enabled = (new_state == "enabled")
        
        # Update config file
        self.config_loader.update_service_enabled(service_name, new_state == "enabled")
        
        # Invalidate only this service's cache
        await self.cache.invalidate_service(service_name)
        
        # If enabling, poll the service to populate cache
        if new_state == "enabled":
            try:
                _, service_status = await self._poll_single_service(
                    service_name, self.service_configs[service_name]
                )
                await self.cache.set_service_status(service_name, service_status)
            except Exception as e:
                logger.warning(f"Failed to poll newly enabled service {service_name}: {e}")
        else:
            # If disabling, just update cache with disabled status
            await self.cache.set_service_status(service_name, {
                "enabled": False,
                "transport": self.service_configs[service_name].transport,
                "connected": False,
                "tools_count": 0,
                "error": None
            })
        
        return {
            "service": service_name,
            "status": new_state,
            "success": True
        }
    
    async def start_oauth_flow(self, service_name: str) -> Dict[str, Any]:
        """Start OAuth flow for service (using FastMCP)"""
        try:
            config = self.service_configs.get(service_name)
            if not config:
                return {"error": f"Service {service_name} not found"}
            
            # Check if service requires OAuth using FastMCP's check
            url = config.spec.get('url')
            if not url:
                return {"error": f"Service {service_name} is not an HTTP server"}
                
            from fastmcp.client.auth.oauth import check_if_auth_required
            requires_auth = await check_if_auth_required(url)
            if not requires_auth:
                return {"error": f"Service {service_name} does not require OAuth"}
            
            # Create client - this will trigger OAuth flow  
            client = self.create_fastmcp_client(service_name)
            if not client:
                return {"error": f"Failed to create client for {service_name}"}
            
            # Test connection to trigger OAuth
            async with client as c:
                await c.ping()
            
            return {
                "success": True,
                "service": service_name,
                "message": "OAuth flow initiated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start OAuth flow for {service_name}: {e}")
            return {"error": str(e)}
    
    async def reset_oauth_tokens(self, service_name: str) -> Dict[str, Any]:
        """Reset OAuth tokens for service"""
        try:
            config = self.service_configs.get(service_name)
            if not config:
                return {"error": f"Service {service_name} not found"}
            
            # Clear FastMCP tokens
            from fastmcp.client.auth.oauth import FileTokenStorage
            
            url = config.spec.get('url')
            if url:
                storage = FileTokenStorage(server_url=url)
                await storage.clear()
            
            # No client cache to clear - per-request approach
            
            return {
                "success": True,
                "service": service_name,
                "message": "OAuth tokens cleared successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to reset OAuth for {service_name}: {e}")
            return {"error": str(e)}
    
    async def _check_oauth_single(self, service_name: str, config: ServiceConfig) -> tuple[str, Optional[Dict[str, Any]]]:
        """Check OAuth status for single service (for parallel execution)"""
        # Check only HTTP servers
        if 'url' not in config.spec:
            return (service_name, None)
        
        url = config.spec.get('url')
        if not url:
            return (service_name, None)
        
        try:
            # Use FastMCP's check_if_auth_required to determine if server needs OAuth
            from fastmcp.client.auth.oauth import check_if_auth_required
            
            requires_auth = await asyncio.wait_for(check_if_auth_required(url), timeout=3.0)
            
            if requires_auth:
                # Server requires OAuth - check token status
                from fastmcp.client.auth.oauth import FileTokenStorage
                
                storage = FileTokenStorage(server_url=url)
                tokens = await storage.get_tokens()
                
                if tokens:
                    return (service_name, {
                        "authenticated": True,
                        "has_token": True,
                        "token_type": getattr(tokens, 'token_type', 'Bearer'),
                        "expires_at": getattr(tokens, 'expires_at', None),
                        "scope": getattr(tokens, 'scope', '')
                    })
                else:
                    return (service_name, {
                        "authenticated": False,
                        "has_token": False
                    })
            # If server doesn't require OAuth, we don't include it
            return (service_name, None)
            
        except asyncio.TimeoutError:
            logger.warning(f"OAuth check timeout for {service_name}")
            return (service_name, None)
        except Exception as e:
            logger.warning(f"Failed to check OAuth status for {service_name}: {e}")
            return (service_name, None)
    
    async def get_oauth_status(self) -> Dict[str, Any]:
        """Get OAuth status for all services with parallel processing"""
        # Create tasks for parallel OAuth checking
        tasks = []
        for service_name, config in self.service_configs.items():
            # Check cache first
            cached_oauth = await self.cache.get_oauth_status(service_name)
            if cached_oauth:
                continue  # Will use cached value
            
            tasks.append(self._check_oauth_single(service_name, config))
        
        # Get cached values
        oauth_status = {}
        for service_name in self.service_configs:
            cached = await self.cache.get_oauth_status(service_name)
            if cached:
                oauth_status[service_name] = cached
        
        # If we have tasks, execute them in parallel
        if tasks:
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start_time
            logger.info(f"OAuth check completed in {elapsed:.2f}s for {len(tasks)} services")
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"OAuth check error: {result}")
                    continue
                
                service_name, oauth_data = result
                if oauth_data:
                    oauth_status[service_name] = oauth_data
                    # Cache the result
                    await self.cache.set_oauth_status(service_name, oauth_data)
        
        return oauth_status
    
    async def get_service_oauth_status(self, service_name: str) -> Dict[str, Any]:
        """Get OAuth status for specific service"""
        all_status = await self.get_oauth_status()
        return all_status.get(service_name, {
            "authenticated": False,
            "has_token": False,
            "error": "Service not found or does not support OAuth"
        })
    
    async def add_server(self, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add new MCP server configuration with cache update"""
        try:
            name = server_config.get('name')
            if not name:
                return {"error": "Server name is required"}
            
            if name in self.service_configs:
                return {"error": f"Service {name} already exists"}
            
            # Create service config
            config = ServiceConfig.from_dict(name, server_config)
            self.service_configs[name] = config
            self.service_states[name] = "enabled" if config.enabled else "disabled"
            
            # Save to mcp_servers.json
            self._save_server_config(name, server_config)
            
            # If service is enabled, poll it and add to cache
            if config.enabled:
                try:
                    _, service_status = await self._poll_single_service(name, config)
                    await self.cache.set_service_status(name, service_status)
                except Exception as e:
                    logger.warning(f"Failed to poll new service {name}: {e}")
                    # Still add to cache with error status
                    await self.cache.set_service_status(name, {
                        "enabled": True,
                        "transport": config.transport,
                        "connected": False,
                        "tools_count": 0,
                        "error": str(e)
                    })
            else:
                # Add disabled service to cache
                await self.cache.set_service_status(name, {
                    "enabled": False,
                    "transport": config.transport,
                    "connected": False,
                    "tools_count": 0,
                    "error": None
                })
            
            return {
                "success": True,
                "service": name,
                "message": "Server added successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to add server: {e}")
            return {"error": str(e)}
    
    async def delete_server(self, service_name: str) -> Dict[str, Any]:
        """Delete MCP server configuration with cache cleanup"""
        try:
            if service_name not in self.service_configs:
                return {"error": f"Service {service_name} not found"}
            
            # Remove from internal state
            del self.service_configs[service_name]
            if service_name in self.service_states:
                del self.service_states[service_name]
            
            # Remove from config file
            self._remove_server_config(service_name)
            
            # Remove from cache (will update counters)
            await self.cache.remove_service(service_name)
            
            return {
                "success": True,
                "service": service_name,
                "message": "Server deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete server {service_name}: {e}")
            return {"error": str(e)}
    
    def _save_server_config(self, name: str, config: Dict[str, Any]):
        """Save server configuration using standard MCP format"""
        self.config_loader.add_server(name, config)
    
    def _remove_server_config(self, name: str):
        """Remove server configuration using standard MCP format"""
        self.config_loader.remove_server(name)