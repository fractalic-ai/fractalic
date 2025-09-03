#!/usr/bin/env python3
"""
Fractalic Publisher - Main CLI Interface

A plugin-based publishing system for deploying Fractalic to various cloud platforms.
"""
import argparse
import logging
import sys
import os
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from publisher.plugin_manager import PluginManager
from publisher.models import DeploymentConfig, PluginCapability


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )


def list_plugins(plugin_manager: PluginManager):
    """List all available plugins"""
    plugins = plugin_manager.list_plugins()
    if not plugins:
        print("No plugins found.")
        return
        
    print("📦 Available Publishing Plugins:")
    print()
    
    for plugin_name in sorted(plugins):
        info = plugin_manager.get_plugin_info(plugin_name)
        if info:
            print(f"🔌 {info.display_name} ({plugin_name})")
            print(f"   Description: {info.description}")
            print(f"   Difficulty: {info.setup_difficulty}")
            print(f"   Deploy Time: {info.deploy_time_estimate}")
            print(f"   Pricing: {info.pricing_info}")
            
            # Show capabilities
            capabilities = [cap.value.replace('-', ' ').title() for cap in info.capabilities]
            if capabilities:
                print(f"   Features: {', '.join(capabilities)}")
            print()


def show_plugin_details(plugin_manager: PluginManager, plugin_name: str):
    """Show detailed information about a specific plugin"""
    info = plugin_manager.get_plugin_info(plugin_name)
    if not info:
        print(f"❌ Plugin '{plugin_name}' not found.")
        return
        
    print(f"📋 {info.display_name} Details")
    print("=" * 50)
    print(f"Name: {info.name}")
    print(f"Version: {info.version}")
    print(f"Description: {info.description}")
    print(f"Homepage: {info.homepage_url}")
    print(f"Documentation: {info.documentation_url}")
    print(f"Setup Difficulty: {info.setup_difficulty}")
    print(f"Deploy Time: {info.deploy_time_estimate}")
    print(f"Pricing: {info.pricing_info}")
    print(f"Free Tier: {info.free_tier_limits}")
    print()
    
    # Capabilities
    if info.capabilities:
        print("🚀 Capabilities:")
        for cap in info.capabilities:
            print(f"  • {cap.value.replace('-', ' ').title()}")
        print()
    
    # Deploy button if available
    if info.badge_url or info.deploy_button_url:
        print("🔗 Integration:")
        if info.badge_url:
            print(f"  Badge: {info.badge_url}")
        if info.deploy_button_url:
            print(f"  Deploy Button: {info.deploy_button_url}")


def deploy_application(plugin_manager: PluginManager, args):
    """Deploy application using specified plugin"""
    plugin = plugin_manager.get_plugin(args.plugin)
    if not plugin:
        print(f"❌ Plugin '{args.plugin}' not found.")
        return False
        
    # Build configuration
    config = DeploymentConfig(
        plugin_name=args.plugin,
        container_name=args.name,
        environment_vars=parse_env_vars(args.env or []),
        port_mapping=parse_port_mapping(args.ports or "3000:3000,8000:8000"),
        custom_domain=args.domain,
        plugin_specific={},
        script_name=getattr(args, 'script_name', ''),
        script_folder=getattr(args, 'script_folder', '')
    )
    
    # Validate configuration
    is_valid, error = plugin.validate_config(config)
    if not is_valid:
        print(f"❌ Configuration error: {error}")
        return False
        
    # Progress callback
    def progress_callback(message: str, percentage: int):
        print(f"[{percentage:3d}%] {message}")
    
    print(f"🚀 Deploying '{args.name}' using {args.plugin}...")
    print()
    
    # Deploy
    result = plugin.publish(
        source_path=os.getcwd(),
        config=config,
        progress_callback=progress_callback
    )
    
    print()
    if result.success:
        print("✅ Deployment successful!")
        # Display the detailed message from the plugin
        if hasattr(result, 'message') and result.message:
            print(result.message)
        else:
            # Fallback to basic info if no detailed message
            print(f"🌐 URL: {result.url}")
            if result.admin_url:
                print(f"⚙️  Admin: {result.admin_url}")
            if result.build_time:
                print(f"⏱️  Build time: {result.build_time:.1f}s")
            print(f"📝 Deployment ID: {result.deployment_id}")
    else:
        print("❌ Deployment failed!")
        print(f"Error: {result.error}")
        return False
        
    return True


def generate_readme_badges(plugin_manager: PluginManager):
    """Generate README badges for one-click deployment"""
    one_click_plugins = plugin_manager.get_one_click_plugins()
    
    if not one_click_plugins:
        print("No one-click deployment plugins available.")
        return
        
    print("📝 README Deploy Buttons:")
    print()
    print("Add these to your README.md for one-click deployment:")
    print()
    
    for plugin_name in one_click_plugins:
        plugin = plugin_manager.get_plugin(plugin_name)
        info = plugin_manager.get_plugin_info(plugin_name)
        
        if info and (info.badge_url or info.deploy_button_url):
            # Create sample config for button generation
            config = DeploymentConfig(
                plugin_name=plugin_name,
                container_name="fractalic-app",
                environment_vars={},
                port_mapping={3000: 3000, 8000: 8000}
            )
            
            button = plugin.get_deploy_button_markdown(config)
            if button:
                print(f"## {info.display_name}")
                print(button)
                print()


def parse_env_vars(env_list: list) -> Dict[str, str]:
    """Parse environment variables from KEY=VALUE format"""
    env_vars = {}
    for env_str in env_list:
        if '=' in env_str:
            key, value = env_str.split('=', 1)
            env_vars[key] = value
    return env_vars


def parse_port_mapping(ports_str: str) -> Dict[int, int]:
    """Parse port mapping from HOST:CONTAINER,HOST:CONTAINER format"""
    port_mapping = {}
    for port_pair in ports_str.split(','):
        if ':' in port_pair:
            host_port, container_port = port_pair.split(':', 1)
            port_mapping[int(host_port)] = int(container_port)
    return port_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Fractalic Publisher - Deploy to cloud platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available plugins
  python publisher_cli.py list
  
  # Show plugin details
  python publisher_cli.py info local_docker
  
  # Deploy to local Docker
  python publisher_cli.py deploy local_docker --name my-app
  
  # Deploy with custom ports and environment
  python publisher_cli.py deploy local_docker --name my-app \\
    --ports "3100:3000,8100:8000" --env "API_KEY=secret"
  
  # Generate README badges
  python publisher_cli.py badges
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List plugins
    subparsers.add_parser('list', help='List all available plugins')
    
    # Plugin info
    info_parser = subparsers.add_parser('info', help='Show detailed plugin information')
    info_parser.add_argument('plugin', help='Plugin name')
    
    # Deploy
    deploy_parser = subparsers.add_parser('deploy', help='Deploy to a platform')
    deploy_parser.add_argument('plugin', help='Plugin name')
    deploy_parser.add_argument('--name', '-n', required=True, help='Deployment name')
    deploy_parser.add_argument('--ports', '-p', help='Port mapping (e.g., "3000:3000,8000:8000")')
    deploy_parser.add_argument('--env', '-e', action='append', help='Environment variable (KEY=VALUE)')
    deploy_parser.add_argument('--domain', '-d', help='Custom domain')
    deploy_parser.add_argument('--script-name', help='Script name (required for docker-registry plugin)')
    deploy_parser.add_argument('--script-folder', help='Script folder path (required for docker-registry plugin)')
    
    # Generate badges
    subparsers.add_parser('badges', help='Generate README deploy buttons')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    setup_logging(args.verbose)
    
    # Initialize plugin manager
    plugin_manager = PluginManager()
    loaded_plugins = plugin_manager.load_all_plugins()
    
    if not loaded_plugins:
        print("⚠️  No plugins found. Please check your plugins directory.")
        return
        
    # Handle commands
    if args.command == 'list':
        list_plugins(plugin_manager)
    elif args.command == 'info':
        show_plugin_details(plugin_manager, args.plugin)
    elif args.command == 'deploy':
        deploy_application(plugin_manager, args)
    elif args.command == 'badges':
        generate_readme_badges(plugin_manager)


if __name__ == "__main__":
    main()
