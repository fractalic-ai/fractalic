#!/usr/bin/env python3
"""
FastMCP-based MCP Manager - Main Entry Point
Maintains CLI compatibility with original fractalic_mcp_manager.py
"""

import asyncio
import logging
import argparse
import signal
import sys
from pathlib import Path
from aiohttp import web
import aiohttp_cors

from api_handlers import setup_routes, init_manager
from fastmcp_manager import FastMCPManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FastMCPServer:
    """FastMCP-based MCP Manager Server"""
    
    def __init__(self, port: int = 5859, host: str = "localhost"):
        self.port = port
        self.host = host
        self.app = None
        self.runner = None
        self.site = None
        self._shutdown_event = asyncio.Event()
    
    async def create_app(self):
        """Create aiohttp web application"""
        self.app = web.Application()
        
        # Initialize global manager
        init_manager()
        
        # Setup routes with CORS
        setup_routes(self.app)
        
        # Add middleware for logging
        @web.middleware
        async def logging_middleware(request, handler):
            start_time = asyncio.get_event_loop().time()
            try:
                response = await handler(request)
                process_time = asyncio.get_event_loop().time() - start_time
                logger.info(f"{request.method} {request.path} - {response.status} - {process_time:.3f}s")
                return response
            except Exception as e:
                process_time = asyncio.get_event_loop().time() - start_time
                logger.error(f"{request.method} {request.path} - ERROR: {e} - {process_time:.3f}s")
                raise
        
        self.app.middlewares.append(logging_middleware)
        return self.app
    
    async def start_server(self):
        """Start the HTTP server"""
        try:
            await self.create_app()
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            
            logger.info(f"FastMCP Manager started on http://{self.host}:{self.port}")
            logger.info("Available endpoints:")
            logger.info("  GET /health - Health check")
            logger.info("  GET /status - Service status")
            logger.info("  GET /status/complete - Complete status with OAuth")
            logger.info("  GET /list_tools - All tools (Fractalic compatibility)")
            logger.info("  GET /tools - All tools")
            logger.info("  GET /tools/{name} - Tools for service")
            logger.info("  POST /call/{service}/{tool} - Call tool")
            logger.info("  POST /toggle/{name} - Toggle service")
            logger.info("  GET /oauth/status - OAuth status for all services")
            logger.info("  POST /oauth/start/{service} - Start OAuth flow")
            logger.info("  POST /oauth/reset/{service} - Reset OAuth tokens")
            logger.info("  POST /add_server - Add MCP server")
            logger.info("  POST /delete_server - Delete MCP server")
            logger.info("  POST /kill - Shutdown server")
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the HTTP server"""
        try:
            if self.site:
                await self.site.stop()
                logger.info("Server site stopped")
            
            if self.runner:
                await self.runner.cleanup()
                logger.info("Server runner cleaned up")
                
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
    
    async def run_forever(self):
        """Run server until shutdown signal"""
        await self.start_server()
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        finally:
            logger.info("Shutting down FastMCP Manager...")
            await self.stop_server()

async def test_services():
    """Test connectivity to configured MCP services"""
    logger.info("Testing MCP service connectivity...")
    
    manager = FastMCPManager()
    status = await manager.get_service_status()
    
    print("\n=== MCP Services Status ===")
    print(f"Total Services: {len(status['services'])}")
    print(f"Enabled: {status['total_enabled']}")
    print(f"Disabled: {status['total_disabled']}")
    print()
    
    for service_name, service_info in status['services'].items():
        status_icon = "✅" if service_info['connected'] else "❌"
        # OAuth status now comes separately from get_oauth_status()
        tools_count = f" ({service_info['tools_count']} tools)" if service_info['connected'] else ""
        error_msg = f" - Error: {service_info['error']}" if service_info.get('error') else ""
        
        print(f"{status_icon} {service_name}{tools_count}{error_msg}")
    
    print("\n=== OAuth Status ===")
    oauth_status = await manager.get_oauth_status()
    for service_name, oauth_info in oauth_status.items():
        if oauth_info.get('authenticated'):
            print(f"🔐 {service_name}: Authenticated")
        else:
            print(f"🔓 {service_name}: Not authenticated")

async def list_tools():
    """List all available tools from all services"""
    logger.info("Listing all MCP tools...")
    
    manager = FastMCPManager()
    all_tools = await manager.get_all_tools()
    
    print("\n=== Available MCP Tools ===")
    total_tools = 0
    
    for service_name, service_data in all_tools.items():
        if 'error' in service_data:
            print(f"\n❌ {service_name}: {service_data['error']}")
            continue
        
        tools = service_data.get('tools', [])
        count = len(tools)
        total_tools += count
        
        print(f"\n📦 {service_name} ({count} tools):")
        for tool in tools:
            print(f"  • {tool['name']}: {tool.get('description', 'No description')}")
    
    print(f"\nTotal tools available: {total_tools}")

def main():
    """Main CLI entry point - maintains compatibility with original"""
    parser = argparse.ArgumentParser(
        description="FastMCP-based MCP Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fastmcp_main.py                    # Start server on default port 5859
  python fastmcp_main.py --port 8080        # Start server on custom port
  python fastmcp_main.py --test             # Test service connectivity
  python fastmcp_main.py --list-tools       # List all available tools
  python fastmcp_main.py --verbose          # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5859,
        help='Port to run the server on (default: 5859)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind the server to (default: localhost)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test connectivity to all configured MCP services'
    )
    
    parser.add_argument(
        '--list-tools', '-l',
        action='store_true',
        help='List all available tools from all services'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='mcp_servers.json',
        help='Path to MCP servers configuration file'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Handle different modes
    if args.test:
        asyncio.run(test_services())
        return
    
    if args.list_tools:
        asyncio.run(list_tools())
        return
    
    # Default: Start HTTP server
    logger.info(f"Starting FastMCP Manager (reduced from 4656 to ~900 lines)")
    logger.info(f"Configuration file: {args.config}")
    
    server = FastMCPServer(port=args.port, host=args.host)
    
    try:
        asyncio.run(server.run_forever())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()