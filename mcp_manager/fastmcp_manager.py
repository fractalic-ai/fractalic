#!/usr/bin/env python3
"""
Fractalic MCP Manager


TODO: Token counting feature
- Scope: Only tool schemas (not prompts/resources) 
- Method: 3-tier fallback (LiteLLM→TikToken→char/4 approximation)
- Format: Convert MCP tools to OpenAI function format for LiteLLM
- Model: Use gpt-4o tokenizer for accuracy
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastmcp import Client
from fastmcp.client.auth import OAuth

from .mcp_config import MCPConfigLoader, ServiceConfig
from .status_cache import StatusCache
from .oauth_helper import get_oauth_cache_dir, create_custom_oauth_client, create_custom_token_storage

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
    
    
    def _get_or_create_oauth(self, url: str) -> OAuth:
        """Create OAuth instance for a URL (no caching - FastMCP handles token persistence)."""
        # FastMCP automatically loads existing tokens from storage
        return create_custom_oauth_client(url)
    
    def create_fastmcp_client(self, service_name: str, context: Dict[str, Any] | None = None) -> Optional[Client]:
        """Create FastMCP client for service (per-request, not cached)"""
        try:
            config = self.service_configs.get(service_name)
            if not config or not config.enabled:
                return None
            
            # Determine desired working directory (default from context)
            desired_cwd = None
            if context and isinstance(context, dict):
                desired_cwd = context.get('current_directory')
            
            if config.transport == 'stdio':
                command = config.spec.get('command')
                if not command:
                    logger.error(f"No command specified for stdio service {service_name}")
                    return None
                
                # Use env from config as-is (do not inject FRACTALIC_* vars)
                env = config.spec.get('env', {})
                
                mcp_config = {
                    "mcpServers": {
                        service_name: {
                            "command": command,
                            "args": config.spec.get('args', []),
                            "env": env
                        }
                    }
                }
                
                # Set working directory via MCP JSON (supported by clients)
                if desired_cwd:
                    mcp_config["mcpServers"][service_name]["cwd"] = desired_cwd
                
                client = Client(mcp_config)
            else:
                url = config.spec.get('url')
                if not url:
                    logger.error(f"No URL specified for service {service_name}")
                    return None
                
                import re
                if re.search(r'/[A-Za-z0-9+/=]{50,}', url):
                    client = Client(url)
                else:
                    oauth_client = self._get_or_create_oauth(url)
                    client = Client(url, auth=oauth_client)
            
            return client
        except Exception as e:
            logger.exception(f"Error creating FastMCP client for {service_name}: {e}")
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
                    # FastMCP automatically handles transport detection and parallel/sequential execution
                    # Let FastMCP determine the best approach based on the actual transport
                    ping_task = asyncio.wait_for(c.ping(), timeout=5.0)
                    tools_task = asyncio.wait_for(c.list_tools(), timeout=5.0)
                    ping_result, tools_result = await asyncio.gather(ping_task, tools_task, return_exceptions=True)
                    
                    # Process ping result
                    if not isinstance(ping_result, Exception):
                        service_status["connected"] = True
                    else:
                        logger.warning(f"Ping failed for {service_name}: {ping_result}")
                    
                    # Process tools result
                    if not isinstance(tools_result, Exception):
                        tools_list = tools_result.tools if hasattr(tools_result, 'tools') else tools_result
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
                    else:
                        logger.warning(f"Tools fetch failed for {service_name}: {tools_result}")
                        service_status["tools_count"] = 0
                    
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
        
        # Check for cached complete status first
        cached_complete_status = await self.cache.get_cached_data("complete_status", ttl=30.0)
        if cached_complete_status is not None:
            logger.info("Returning cached complete status")
            return cached_complete_status
        
        try:
            logger.info("=== Starting get_complete_status() with SINGLE CLIENT architecture ===")
            
            start_time = time.time()
            
            # Get all services (enabled and disabled)
            all_services = list(self.service_configs.keys())
            enabled_services = [name for name, config in self.service_configs.items() if config.enabled]
            disabled_services = [name for name, config in self.service_configs.items() if not config.enabled]
            logger.info(f"All services: {all_services}")
            logger.info(f"Enabled services for data fetch: {enabled_services}")
            logger.info(f"Disabled services (config only): {disabled_services}")
            
            # UNIFIED APPROACH: Only get OAuth status separately, extract service status from unified service data
            all_tasks = [
                self.get_oauth_status(), 
            ]
            
            # Add single task per ENABLED service that gets ALL data (status+tools+prompts+resources) with one client  
            service_data_tasks = []
            if enabled_services:
                service_data_tasks = [self.get_all_service_data(name) for name in enabled_services]
                all_tasks.extend(service_data_tasks)
            
            logger.info(f"Running {len(all_tasks)} total tasks ({len(service_data_tasks)} unified single-client service tasks)")
            
            # Execute all operations in parallel
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Unpack results
            oauth_status = results[0]
            
            # Build service status from unified service data results (no duplicate clients!)
            basic_status = {
                "services": {},
                "total_enabled": len(enabled_services),
                "total_disabled": len(disabled_services)
            }
            all_tools = {}
            all_prompts = {}
            all_resources = {}
            
            # Process enabled services with data fetch
            if enabled_services:
                for i, service_name in enumerate(enabled_services):
                    service_data = results[1 + i]  # Skip oauth_status
                    if isinstance(service_data, dict) and 'tools' in service_data:
                        # Extract status info from unified data
                        status_info = service_data.get("status_info", {})
                        basic_status["services"][service_name] = {
                            "enabled": status_info.get("enabled", True),
                            "transport": status_info.get("transport", "unknown"),
                            "connected": status_info.get("connected", False),
                            "tools_count": status_info.get("tools_count", len(service_data["tools"])),
                            "error": status_info.get("error")
                        }
                        
                        # Process tools, prompts, resources from unified data
                        all_tools[service_name] = {
                            "tools": service_data["tools"],
                            "count": len(service_data["tools"])
                        }
                        all_prompts[service_name] = service_data["prompts"]
                        all_resources[service_name] = service_data["resources"]
                    else:
                        logger.warning(f"Invalid service data for {service_name}: {service_data}")
                        # Add to basic_status with error state
                        basic_status["services"][service_name] = {
                            "enabled": True,
                            "transport": "unknown", 
                            "connected": False,
                            "tools_count": 0,
                            "error": f"Invalid service data: {service_data}"
                        }
                        basic_status["total_enabled"] -= 1
                        basic_status["total_disabled"] += 1
                        
                        all_tools[service_name] = {"tools": [], "count": 0}
                        all_prompts[service_name] = []
                        all_resources[service_name] = []
            
            # Process disabled services (config only, no data fetch)
            for service_name in disabled_services:
                config = self.service_configs.get(service_name)
                basic_status["services"][service_name] = {
                    "enabled": False,
                    "transport": config.spec.get("transport", "stdio") if config else "unknown",
                    "connected": False,
                    "tools_count": 0,
                    "error": None
                }
                
                # Empty data for disabled services
                all_tools[service_name] = {"tools": [], "count": 0}
                all_prompts[service_name] = []
                all_resources[service_name] = []
            
            elapsed = time.time() - start_time
            logger.info(f"ALL data gathered with SINGLE CLIENT architecture in {elapsed:.2f}s")
            
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
            
            
            # Build response in frontend schema format
            total_prompts = sum(len(prompts) for prompts in all_prompts.values())
            total_resources = sum(len(resources) for resources in all_resources.values())
            
            complete_status = {
                "total_services": len(all_services),
                "enabled_services": basic_status["total_enabled"], 
                "total_tools": sum(data.get("count", 0) for data in all_tools.values()),
                "oauth_enabled": len(oauth_status) > 0,
                "services": {}
            }
            
            # Build service details with embedded data
            logger.debug("Building service details...")
            for service_name in all_services:  # Process ALL services (enabled and disabled)
                logger.debug(f"Processing service: {service_name}")
                
                # Get service info from basic_status (guaranteed to exist for all services)
                service_info = basic_status["services"][service_name]
                
                tools_data = all_tools.get(service_name, {})
                tools = tools_data.get("tools", [])
                
                # Get prompts and resources from single-client results
                prompts = all_prompts.get(service_name, [])
                resources = all_resources.get(service_name, [])
                
                service_state = "connected" if service_info["connected"] else ("disabled" if not service_info["enabled"] else "error")
                complete_service = {
                    "status": service_state,
                    "connected": service_info["connected"],
                    "enabled": service_info["enabled"], 
                    "transport": service_info["transport"],
                    "has_oauth": service_name in oauth_status,
                    "tool_count": len(tools),
                    "prompt_count": len(prompts) if prompts else 0,
                    "resource_count": len(resources) if resources else 0,
                    "token_count": 0,  # TODO: Implement token counting - convert tools to OpenAI format, use LiteLLM→TikToken→approx fallback
                    "tools": tools,
                    "prompts": prompts,
                    "resources": resources
                }
                
                # Totals already calculated above
                
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
            complete_status["total_tokens"] = 0  # TODO: Sum token_count from all services (only tools schemas)
            
            # Final JSON serialization test
            logger.debug("Testing final complete_status JSON serialization...")
            try:
                json.dumps(complete_status)
                logger.debug("Final complete_status: JSON serializable ✓")
            except Exception as e:
                logger.error(f"Final complete_status JSON ERROR: {e}")
                logger.error(f"Final traceback:\n{traceback.format_exc()}")
                raise
            
            # Cache the complete result for 30 seconds
            await self.cache.set_cached_data("complete_status", complete_status, ttl=30.0)
            
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
    
    async def get_all_service_data(self, service_name: str) -> Dict[str, Any]:
        """Get ALL data (status, tools, prompts, resources) for a service using single client - UNIFIED APPROACH"""
        logger.debug(f"Getting unified data for service {service_name} with single client")
        
        # Check caches first
        cached_status = await self.cache.get_service_status(service_name)
        cached_tools = await self.cache.get_service_tools(service_name)
        cached_prompts = await self.cache.get_cached_data(f"prompts_{service_name}", ttl=60.0)
        cached_resources = await self.cache.get_cached_data(f"resources_{service_name}", ttl=60.0)
        
        # If all are cached, return cached data with status
        if (cached_status is not None and cached_tools is not None and 
            cached_prompts is not None and cached_resources is not None):
            logger.debug(f"All unified data cached for {service_name}")
            return {
                "status_info": cached_status,
                "tools": cached_tools,
                "prompts": cached_prompts,
                "resources": cached_resources
            }
        
        # Need to fetch some or all data - use SINGLE CLIENT for everything
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                return {"tools": [], "prompts": [], "resources": []}
            
            # SINGLE CLIENT SESSION WITH PARALLEL OPERATIONS - UNIFIED STATUS + DATA
            config = self.service_configs.get(service_name)
            service_status = {
                "enabled": config.enabled if config else False,
                "transport": config.transport if config else "unknown",
                "connected": False,
                "tools_count": 0,
                "error": None
            }
            
            async with client as c:
                logger.debug(f"Using unified single client session for {service_name} - status + data")
                
                # Create tasks for parallel execution within single client session
                tasks_to_run = []
                
                # Add status tasks if not cached (ping + tools for count)
                if cached_status is None:
                    async def get_status():
                        try:
                            # FastMCP automatically handles transport detection and parallel/sequential execution
                            # Let FastMCP determine the best approach based on the actual transport
                            ping_task = asyncio.wait_for(c.ping(), timeout=5.0)
                            tools_task = asyncio.wait_for(c.list_tools(), timeout=5.0)
                            ping_result, tools_result = await asyncio.gather(ping_task, tools_task, return_exceptions=True)
                            
                            # Process status results
                            if not isinstance(ping_result, Exception):
                                service_status["connected"] = True
                            if not isinstance(tools_result, Exception):
                                tools_list = tools_result.tools if hasattr(tools_result, 'tools') else tools_result
                                service_status["tools_count"] = len(tools_list) if tools_list else 0
                            
                            await self.cache.set_service_status(service_name, service_status)
                            return service_status
                        except Exception as e:
                            logger.warning(f"Status check failed for {service_name}: {e}")
                            service_status["error"] = str(e)
                            return service_status
                    
                    tasks_to_run.append(("status", get_status()))
                
                # Add tools task if not cached (or if we need fresh data for status)
                if cached_tools is None:
                    async def get_tools():
                        try:
                            tools = await asyncio.wait_for(c.list_tools(), timeout=10.0)
                            tools_list = tools.tools if hasattr(tools, 'tools') and tools.tools else tools
                            
                            if tools_list:
                                formatted_tools = []
                                for tool in tools_list:
                                    formatted_tools.append({
                                        "name": tool.name,
                                        "description": tool.description,
                                        "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else None
                                    })
                                await self.cache.set_service_tools(service_name, formatted_tools)
                                return formatted_tools
                            else:
                                await self.cache.set_service_tools(service_name, [])
                                return []
                        except Exception as e:
                            logger.warning(f"Failed to get tools for {service_name}: {e}")
                            return []
                    
                    tasks_to_run.append(("tools", get_tools()))
                
                # Add prompts task if not cached
                if cached_prompts is None:
                    async def get_prompts():
                        try:
                            prompts = await asyncio.wait_for(c.list_prompts(), timeout=5.0)
                            prompts_list = prompts.prompts if hasattr(prompts, 'prompts') else prompts
                            
                            if prompts_list:
                                formatted_prompts = []
                                for prompt in prompts_list:
                                    formatted_prompts.append({
                                        "name": prompt.name,
                                        "description": prompt.description,
                                        "arguments": prompt.arguments if hasattr(prompt, 'arguments') else []
                                    })
                                await self.cache.set_cached_data(f"prompts_{service_name}", formatted_prompts, ttl=60.0)
                                return formatted_prompts
                            else:
                                await self.cache.set_cached_data(f"prompts_{service_name}", [], ttl=60.0)
                                return []
                        except Exception as e:
                            logger.warning(f"Failed to get prompts for {service_name}: {e}")
                            await self.cache.set_cached_data(f"prompts_{service_name}", [], ttl=60.0)
                            return []
                    
                    tasks_to_run.append(("prompts", get_prompts()))
                
                # Add resources task if not cached
                if cached_resources is None:
                    async def get_resources():
                        try:
                            resources = await asyncio.wait_for(c.list_resources(), timeout=5.0)
                            resources_list = resources.resources if hasattr(resources, 'resources') else resources
                            
                            if resources_list:
                                formatted_resources = []
                                for resource in resources_list:
                                    formatted_resources.append({
                                        "uri": str(resource.uri),
                                        "name": resource.name,
                                        "description": resource.description,
                                        "mimeType": resource.mimeType if hasattr(resource, 'mimeType') else None
                                    })
                                await self.cache.set_cached_data(f"resources_{service_name}", formatted_resources, ttl=60.0)
                                return formatted_resources
                            else:
                                await self.cache.set_cached_data(f"resources_{service_name}", [], ttl=60.0)
                                return []
                        except Exception as e:
                            logger.warning(f"Failed to get resources for {service_name}: {e}")
                            await self.cache.set_cached_data(f"resources_{service_name}", [], ttl=60.0)
                            return []
                    
                    tasks_to_run.append(("resources", get_resources()))
                
                # Run all needed operations in parallel within the SAME client session
                if tasks_to_run:
                    logger.debug(f"Running {len(tasks_to_run)} operations in parallel for {service_name}")
                    results = await asyncio.gather(*[task for _, task in tasks_to_run], return_exceptions=True)
                    
                    # Process results
                    for i, (task_name, _) in enumerate(tasks_to_run):
                        result = results[i]
                        if task_name == "status" and cached_status is None:
                            cached_status = result if not isinstance(result, Exception) else service_status
                        elif task_name == "tools" and cached_tools is None:
                            cached_tools = result if not isinstance(result, Exception) else []
                        elif task_name == "prompts" and cached_prompts is None:
                            cached_prompts = result if not isinstance(result, Exception) else []
                        elif task_name == "resources" and cached_resources is None:
                            cached_resources = result if not isinstance(result, Exception) else []
            
            # Ensure we have status info (from cache or fresh)
            final_status = cached_status if cached_status is not None else service_status
            
            logger.debug(f"Unified single client session complete for {service_name}")
            return {
                "status_info": final_status,
                "tools": cached_tools if cached_tools is not None else [],
                "prompts": cached_prompts if cached_prompts is not None else [], 
                "resources": cached_resources if cached_resources is not None else []
            }
                
        except Exception as e:
            logger.error(f"Failed to get unified data for {service_name} with single client: {e}")
            # Return error status + empty data
            error_status = {
                "enabled": config.enabled if config else False,
                "transport": config.transport if config else "unknown", 
                "connected": False,
                "tools_count": 0,
                "error": str(e)
            }
            return {
                "status_info": error_status,
                "tools": [], 
                "prompts": [], 
                "resources": []
            }

    async def get_tools_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get tools for specific service - delegates to single client method"""
        all_data = await self.get_all_service_data(service_name)
        return all_data["tools"]
    
    async def get_all_tools(self) -> Dict[str, Any]:
        """Get tools from all services (enabled with data, disabled with empty data) with full collection caching"""
        # Try to get cached full collection first
        cached_all_tools = await self.cache.get_cached_data("all_tools", ttl=30.0)
        if cached_all_tools is not None:
            logger.info("Returning cached all_tools collection")
            return cached_all_tools
        
        logger.info("Building fresh all_tools collection")
        all_tools = {}
        
        # Get enabled and disabled services
        enabled_services = [(name, config) for name, config in self.service_configs.items() if config.enabled]
        disabled_services = [(name, config) for name, config in self.service_configs.items() if not config.enabled]
        
        # Add disabled services with empty data (for frontend management)
        for name, config in disabled_services:
            all_tools[name] = {
                "tools": [],
                "count": 0,
                "enabled": False,
                "error": "Service is disabled"
            }
        
        if not enabled_services:
            return all_tools
        
        # Create tasks for parallel execution (only for enabled services)
        async def get_service_tools_safe(service_name: str) -> tuple[str, Dict[str, Any]]:
            try:
                tools = await self.get_tools_for_service(service_name)
                return service_name, {
                    "tools": tools,
                    "count": len(tools),
                    "enabled": True
                }
            except Exception as e:
                logger.error(f"Error getting tools for {service_name}: {e}")
                return service_name, {
                    "error": str(e),
                    "tools": [],
                    "count": 0,
                    "enabled": True
                }
        
        # Execute all service tool fetching in parallel (only enabled services)
        logger.info(f"Fetching tools from {len(enabled_services)} enabled services in parallel")
        tasks = [get_service_tools_safe(name) for name, _ in enabled_services]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary (add enabled services data)
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
    
    async def call_tool_for_service(self, service_name: str, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Call tool for specific service"""
        try:
            client = self.create_fastmcp_client(service_name, context)
            if not client:
                raise Exception(f"No client available for service {service_name}")
            
            # Do not mutate arguments; context is used only for client setup (e.g., cwd)
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
                
                output = "\n".join([p for p in content_parts if p]) if content_parts else ""
                
                return {
                    "success": True,
                    "result": {
                        "content": output,
                        "raw": getattr(result, 'raw', None)
                    },
                    "isError": False
                }
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name} for service {service_name}: {e}")
            return {"success": False, "isError": True, "error": str(e)}
    
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
        """Get prompts for specific service - delegates to single client method"""
        all_data = await self.get_all_service_data(service_name)
        return all_data["prompts"]
    
    async def get_resources_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get resources for specific service - delegates to single client method"""
        all_data = await self.get_all_service_data(service_name)
        return all_data["resources"]
    
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
        
        # Invalidate this service's cache
        await self.cache.invalidate_service(service_name)
        
        # Invalidate complete status cache since service state changed
        await self.cache.invalidate_cached_data("complete_status")
        
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
        """Start OAuth flow for service (using FastMCP).
        Primary path: use our custom OAuth (fixed callback_port, cached per URL).
        Fallback: auth="oauth" if custom OAuth fails.
        Treat initial 401/403 as initiation (not fatal) and invalidate caches.
        """
        try:
            config = self.service_configs.get(service_name)
            if not config:
                return {"error": f"Service {service_name} not found"}

            url = config.spec.get('url')
            if not url:
                return {"error": f"Service {service_name} is not an HTTP server"}

            # Primary path: custom OAuth with fixed callback_port (cached per URL)
            try:
                oauth_client = self._get_or_create_oauth(url)
                client = Client(url, auth=oauth_client)

                try:
                    async with client as c:
                        await c.ping()
                    await self.cache.invalidate_service(service_name)
                    return {
                        "success": True,
                        "service": service_name,
                        "message": "OAuth flow completed or not required"
                    }
                except Exception as e:
                    msg = str(e)
                    if ("401" in msg) or ("Unauthorized" in msg) or ("403" in msg) or ("Forbidden" in msg):
                        await self.cache.invalidate_service(service_name)
                        return {
                            "success": True,
                            "service": service_name,
                            "message": "OAuth flow initiated; complete auth in browser"
                        }
                    # Otherwise try fallback path below
                    raise

            except Exception:
                # Fallback: default library behavior (random port) to avoid UX break
                client_fb = Client(url, auth="oauth")
                try:
                    async with client_fb as c:
                        await c.ping()
                    await self.cache.invalidate_service(service_name)
                    return {
                        "success": True,
                        "service": service_name,
                        "message": "OAuth flow completed or not required (fallback)"
                    }
                except Exception as e2:
                    msg2 = str(e2)
                    if ("401" in msg2) or ("Unauthorized" in msg2) or ("403" in msg2) or ("Forbidden" in msg2):
                        await self.cache.invalidate_service(service_name)
                        return {
                            "success": True,
                            "service": service_name,
                            "message": "OAuth flow initiated; complete auth in browser (fallback)"
                        }
                    return {"error": f"Failed to create or ping client for {service_name}: {e2}"}

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
                storage = create_custom_token_storage(url)
                storage.clear()
            
            # No client cache to clear - per-request approach
            
            return {
                "success": True,
                "service": service_name,
                "message": "OAuth tokens cleared successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to reset OAuth for {service_name}: {e}")
            return {"error": str(e)}
    
    async def _check_if_auth_required_with_redirects(self, url: str) -> bool:
        """
        Check if MCP endpoint requires OAuth by following redirects.
        This is a workaround for FastMCP's check_if_auth_required not following redirects.
        """
        import httpx
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=3.0) as client:
                response = await client.get(url)
                # Check for authentication requirements
                if response.status_code == 401:
                    return True
                if 'www-authenticate' in response.headers:
                    return True
                # Check for OAuth-related headers that indicate auth is required
                auth_headers = ['x-clerk-auth-status', 'authorization', 'www-authenticate']
                for header in auth_headers:
                    if header in response.headers:
                        value = response.headers[header].lower()
                        if any(keyword in value for keyword in ['signed-out', 'bearer', 'oauth']):
                            return True
                return False
        except Exception:
            # If we can't determine, fallback to FastMCP's check_if_auth_required
            try:
                from fastmcp.client.auth.oauth import check_if_auth_required
                return await check_if_auth_required(url)
            except Exception:
                return False

    async def _check_oauth_single(self, service_name: str, config: ServiceConfig) -> tuple[str, Optional[Dict[str, Any]]]:
        """Check OAuth status for single service (for parallel execution)"""
        # Check only HTTP servers
        if 'url' not in config.spec:
            return (service_name, None)
        
        url = config.spec.get('url')
        if not url:
            return (service_name, None)
        
        try:
            # First check if we have tokens - if we do, assume server requires auth
            from fastmcp.client.auth.oauth import FileTokenStorage
            
            storage = create_custom_token_storage(url)
            tokens = await storage.get_tokens()
            
            if tokens:
                # We have tokens - server requires OAuth, return token status
                
                if tokens:
                    # FastMCP uses token file modification time + expires_in for expiry calculation
                    import time as _time
                    from pathlib import Path
                    
                    expires_in = getattr(tokens, 'expires_in', None)
                    expires_at = None
                    remaining_seconds = None
                    refresh_needed = False
                    
                    if expires_in:
                        # Build token file path using FastMCP's naming convention
                        cache_key = storage.get_cache_key()
                        token_file = storage.cache_dir / f'{cache_key}_tokens.json'
                        
                        if token_file.exists():
                            # Use file modification time as token issued time
                            issued_at = token_file.stat().st_mtime
                            expires_at = issued_at + expires_in
                            remaining = expires_at - _time.time()
                            remaining_seconds = max(0, int(remaining))
                            refresh_needed = remaining < 300  # Refresh if less than 5 minutes left
                        else:
                            # Fallback: live validation since we can't calculate locally
                            expires_at = None
                            remaining_seconds = None
                    else:
                        # Token doesn't expire (permanent token)  
                        expires_at = None
                        remaining_seconds = None
                    
                    return (service_name, {
                        "authenticated": True,
                        "has_token": True,
                        "has_access_token": bool(getattr(tokens, 'access_token', None)),
                        "has_refresh_token": bool(getattr(tokens, 'refresh_token', None)),
                        "token_type": getattr(tokens, 'token_type', 'Bearer'),
                        "expires_at": expires_at,
                        "access_token_remaining_seconds": remaining_seconds,
                        "refresh_needed": refresh_needed,
                        "scope": getattr(tokens, 'scope', ''),
                        "client_configured": True,  # FastMCP handles OAuth internally
                        "provider_configured": True  # We have tokens, so server requires OAuth
                    })
            
            # No tokens - check if server requires OAuth
            requires_auth = await self._check_if_auth_required_with_redirects(url)
            
            if requires_auth:
                return (service_name, {
                    "authenticated": False,
                    "has_token": False,
                    "has_access_token": False,
                    "has_refresh_token": False,
                    "client_configured": True,  # FastMCP handles OAuth internally
                    "provider_configured": True  # Server requires OAuth
                })
            
            # Server doesn't require OAuth
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
            
            # Invalidate complete status cache since server list changed
            await self.cache.invalidate_cached_data("complete_status")
            
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
            
            # Invalidate complete status cache since server list changed
            await self.cache.invalidate_cached_data("complete_status")
            
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