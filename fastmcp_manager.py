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
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {
            "services": {},
            "total_enabled": 0,
            "total_disabled": 0
        }
        
        for service_name, config in self.service_configs.items():
            service_status = {
                "enabled": config.enabled,
                "transport": config.transport,
                "has_oauth": config.has_oauth,
                "connected": False,
                "tools_count": 0,
                "error": None
            }
            
            if config.enabled:
                status["total_enabled"] += 1
                
                # Test connection
                try:
                    client = self.create_fastmcp_client(service_name)
                    if client:
                        async with client as c:
                            await c.ping()
                            service_status["connected"] = True
                            
                            # Get tools count
                            tools = await c.list_tools()
                            tools_list = tools.tools if hasattr(tools, 'tools') else tools
                            service_status["tools_count"] = len(tools_list) if tools_list else 0
                            
                except Exception as e:
                    service_status["error"] = str(e)
            else:
                status["total_disabled"] += 1
            
            status["services"][service_name] = service_status
        
        return status
    
    async def get_tools_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get tools for specific service"""
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                return []
            
            async with client as c:
                tools = await c.list_tools()
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
                
                return formatted_tools
                
        except Exception as e:
            logger.error(f"Failed to get tools for {service_name}: {e}")
            raise
    
    async def get_all_tools(self) -> Dict[str, Any]:
        """Get tools from all enabled services"""
        all_tools = {}
        
        for service_name, config in self.service_configs.items():
            if not config.enabled:
                continue
            
            try:
                tools = await self.get_tools_for_service(service_name)
                all_tools[service_name] = {
                    "tools": tools,
                    "count": len(tools)
                }
            except Exception as e:
                all_tools[service_name] = {
                    "error": str(e),
                    "tools": [],
                    "count": 0
                }
        
        return all_tools
    
    async def call_tool_for_service(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool for specific service"""
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                raise Exception(f"No client available for service {service_name}")
            
            async with client as c:
                result = await c.call_tool(tool_name, arguments)
                
                # Format result for API response
                return {
                    "success": True,
                    "result": {
                        "content": result.content if hasattr(result, 'content') else str(result),
                        "isError": getattr(result, 'is_error', False)
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
        """Get prompts for specific service"""
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                return []
            
            async with client as c:
                prompts = await c.list_prompts()
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
                
        except Exception as e:
            logger.error(f"Failed to get prompts for {service_name}: {e}")
            raise
    
    async def get_resources_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get resources for specific service"""
        try:
            client = self.create_fastmcp_client(service_name)
            if not client:
                return []
            
            async with client as c:
                resources = await c.list_resources()
                resources_list = resources.resources if hasattr(resources, 'resources') else resources
                
                if not resources_list:
                    return []
                
                # Format resources for API response
                formatted_resources = []
                for resource in resources_list:
                    formatted_resources.append({
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType if hasattr(resource, 'mimeType') else None
                    })
                
                return formatted_resources
                
        except Exception as e:
            logger.error(f"Failed to get resources for {service_name}: {e}")
            raise
    
    def toggle_service(self, service_name: str) -> Dict[str, Any]:
        """Toggle service enabled/disabled status"""
        if service_name not in self.service_configs:
            return {"error": f"Service {service_name} not found"}
        
        # Toggle state
        current_state = self.service_states.get(service_name, "enabled")
        new_state = "disabled" if current_state == "enabled" else "enabled"
        
        self.service_states[service_name] = new_state
        self.service_configs[service_name].enabled = (new_state == "enabled")
        
        # No client cache to clear - per-request approach
        
        # Update config file
        self.config_loader.update_service_enabled(service_name, new_state == "enabled")
        
        return {
            "service": service_name,
            "status": new_state,
            "success": True
        }
    
    async def start_oauth_flow(self, service_name: str) -> Dict[str, Any]:
        """Start OAuth flow for service (using FastMCP)"""
        try:
            config = self.service_configs.get(service_name)
            if not config or not config.has_oauth:
                return {"error": f"Service {service_name} does not support OAuth"}
            
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
    
    async def get_oauth_status(self) -> Dict[str, Any]:
        """Get OAuth status for all services"""
        oauth_status = {}
        
        for service_name, config in self.service_configs.items():
            if config.has_oauth:
                try:
                    # Check FastMCP token status
                    from fastmcp.client.auth.oauth import FileTokenStorage
                    
                    url = config.spec.get('url')
                    if url:
                        storage = FileTokenStorage(server_url=url)
                        tokens = await storage.get_tokens()
                        
                        if tokens:
                            oauth_status[service_name] = {
                                "authenticated": True,
                                "has_token": True,
                                "token_type": getattr(tokens, 'token_type', 'Bearer'),
                                "expires_at": getattr(tokens, 'expires_at', None),
                                "scope": getattr(tokens, 'scope', '')
                            }
                        else:
                            oauth_status[service_name] = {
                                "authenticated": False,
                                "has_token": False
                            }
                    else:
                        oauth_status[service_name] = {"error": "No URL configured"}
                        
                except Exception as e:
                    oauth_status[service_name] = {"error": str(e)}
        
        return oauth_status
    
    async def get_service_oauth_status(self, service_name: str) -> Dict[str, Any]:
        """Get OAuth status for specific service"""
        all_status = await self.get_oauth_status()
        return all_status.get(service_name, {
            "authenticated": False,
            "has_token": False,
            "error": "Service not found or does not support OAuth"
        })
    
    def add_server(self, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add new MCP server configuration"""
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
            
            return {
                "success": True,
                "service": name,
                "message": "Server added successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to add server: {e}")
            return {"error": str(e)}
    
    def delete_server(self, service_name: str) -> Dict[str, Any]:
        """Delete MCP server configuration"""
        try:
            if service_name not in self.service_configs:
                return {"error": f"Service {service_name} not found"}
            
            # Remove from internal state
            del self.service_configs[service_name]
            if service_name in self.service_states:
                del self.service_states[service_name]
            
            # Remove from config file
            self._remove_server_config(service_name)
            
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