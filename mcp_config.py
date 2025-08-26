#!/usr/bin/env python3
"""
MCP Service Configuration Module
Handles loading and management of MCP server configurations
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent

@dataclass
class ServiceConfig:
    """Service configuration for MCP servers"""
    name: str
    transport: str  # stdio, sse, streamable-http
    spec: Dict[str, Any]
    has_oauth: bool = False
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> 'ServiceConfig':
        """Create ServiceConfig from dictionary"""
        # Auto-detect transport if not specified
        transport = config.get('transport', 'stdio')
        
        if 'url' in config and transport == 'stdio':
            url_val = config['url']
            path = urlparse(url_val).path.rstrip('/').lower()
            
            # Detect transport type from URL path
            if path.endswith('/sse') or '/sse/' in path or path.endswith('/mcp/sse'):
                transport = 'sse'
            else:
                transport = 'streamable-http'
        
        # OAuth only when explicitly configured by user
        has_oauth = config.get('oauth', False)
        
        return cls(
            name=name,
            transport=transport,
            spec=config,
            has_oauth=has_oauth,
            enabled=config.get('enabled', True)
        )
    
    
    def to_fastmcp_config(self) -> Dict[str, Any]:
        """Convert to FastMCP client configuration format"""
        if self.transport == 'stdio':
            return {
                "command": self.spec.get('command'),
                "args": self.spec.get('args', []),
                "env": self.spec.get('env', {})
            }
        else:
            config = {"url": self.spec.get('url')}
            if self.has_oauth:
                config["auth"] = "oauth"
            return config


class MCPConfigLoader:
    """Loads and manages MCP server configurations"""
    
    def __init__(self, config_path: str = "mcp_servers.json"):
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = ROOT_DIR / self.config_path
        
        self._config_cache: Optional[Dict[str, ServiceConfig]] = None
    
    def load_config(self) -> Dict[str, ServiceConfig]:
        """Load MCP server configurations from JSON file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
            
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            configs = {}
            mcp_servers = data.get('mcpServers', {})
            
            for name, spec in mcp_servers.items():
                try:
                    config = ServiceConfig.from_dict(name, spec)
                    configs[name] = config
                    logger.debug(f"Loaded config for {name}: transport={config.transport}, oauth={config.has_oauth}")
                except Exception as e:
                    logger.error(f"Failed to load config for {name}: {e}")
                    continue
            
            self._config_cache = configs
            logger.info(f"Loaded {len(configs)} service configurations")
            return configs
            
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {}
    
    def get_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get configuration for specific service"""
        if self._config_cache is None:
            self.load_config()
        return self._config_cache.get(service_name)
    
    def get_oauth_services(self) -> Dict[str, ServiceConfig]:
        """Get all services that require OAuth"""
        if self._config_cache is None:
            self.load_config()
        return {name: config for name, config in self._config_cache.items() 
                if config.has_oauth and config.enabled}
    
    def get_enabled_services(self) -> Dict[str, ServiceConfig]:
        """Get all enabled services"""
        if self._config_cache is None:
            self.load_config()
        return {name: config for name, config in self._config_cache.items() 
                if config.enabled}
    
    def update_service_enabled(self, service_name: str, enabled: bool):
        """Update service enabled status"""
        if self._config_cache is None:
            self.load_config()
        
        if service_name in self._config_cache:
            self._config_cache[service_name].enabled = enabled
            # Optionally save back to file
            self._save_config()
    
    def _save_config(self):
        """Save current configuration back to file"""
        try:
            if self._config_cache is None:
                return
            
            # Load original data to preserve structure
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            # Update enabled status
            for name, config in self._config_cache.items():
                if name in data.get('mcpServers', {}):
                    data['mcpServers'][name]['enabled'] = config.enabled
            
            # Save back
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def add_server(self, name: str, config: Dict[str, Any]):
        """Add new server to configuration file"""
        try:
            # Load existing config
            data = {"mcpServers": {}}
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
            
            # Add new server
            data["mcpServers"][name] = config
            
            # Save back
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Update cache
            if self._config_cache is not None:
                service_config = ServiceConfig.from_dict(name, config)
                self._config_cache[name] = service_config
                
        except Exception as e:
            logger.error(f"Failed to add server {name}: {e}")
            raise
    
    def remove_server(self, name: str):
        """Remove server from configuration file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                # Remove server
                if name in data.get("mcpServers", {}):
                    del data["mcpServers"][name]
                    
                    # Save back
                    with open(self.config_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    # Update cache
                    if self._config_cache is not None and name in self._config_cache:
                        del self._config_cache[name]
                        
        except Exception as e:
            logger.error(f"Failed to remove server {name}: {e}")
            raise
    
    def to_fastmcp_multi_config(self) -> Dict[str, Dict[str, Any]]:
        """Convert all configurations to FastMCP multi-server format"""
        if self._config_cache is None:
            self.load_config()
        
        fastmcp_config = {}
        for name, config in self._config_cache.items():
            if config.enabled:
                fastmcp_config[name] = config.to_fastmcp_config()
        
        return {"mcpServers": fastmcp_config}