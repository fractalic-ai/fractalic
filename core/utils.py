# Utilities
# - read_file
# - change_working_directory
# - print_ast_nodes
# - get_content_without_header
# - execute_shell_command

import os
import subprocess
import locale
from contextlib import contextmanager
import socket
import time
from pathlib import Path

from core.ast_md.node import NodeType, Node
from core.ast_md.ast import AST

# Import centralized path management
from core.paths import get_fractalic_root, get_session_root

def parse_file(filename: str) -> AST:
    content = read_file(filename)
    
    # Run linter before parsing to catch issues early
    try:
        from core.linter import lint_fractalic_file
        from core.ast_md.parser import schema_text
        lint_fractalic_file(filename, schema_text, print_results=True)
    except ImportError:
        # Fallback if linter is not available
        pass
    except Exception as e:
        # Re-raise linting errors to prevent execution with invalid files
        raise e
    
    return AST(content)

@contextmanager
def change_working_directory(new_path):
    """
    Temporarily change the working directory.
    
    :param new_path: Path to the new working directory
    """
    old_path = os.getcwd()
    os.chdir(new_path)
    try:
        yield
    finally:
        os.chdir(old_path)

def read_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        raise IOError(f"Error reading file '{file_path}': {e}")



def print_ast_nodes(ast: AST) -> None:
    current_node = ast.get_first_node()
    if current_node is None:
        # print("[Debug: print_ast_nodes] AST is empty.")
        return

    while current_node is not None:
        print(f"Key: {current_node.key}, ID: {current_node.id}, Content: {current_node.content.strip()}")
        current_node = current_node.next

    
def get_content_without_header(node: Node) -> str:
    content_lines = node.content.split('\n')
    if node.type == NodeType.HEADING and content_lines:
        content_lines = content_lines[1:]
    content_without_header = '\n'.join(content_lines)
    return content_without_header.strip()


import toml

# Import centralized path management
# (get_fractalic_root and get_session_root imported above)

def load_settings(settings_file=None):
    """Load settings from TOML file with proper error handling.
    
    Uses centralized path management to locate settings.toml correctly.
    
    Args:
        settings_file: Optional explicit path to settings file. If not provided,
                      uses centralized path management to find the correct settings.toml
    """
    if settings_file is None:
        try:
            # Try session_root first, then fractalic_root
            try:
                session_root = get_session_root()
                settings_file = str(session_root / "settings.toml")
                if not os.path.exists(settings_file):
                    # Fallback to fractalic_root
                    settings_file = str(get_fractalic_root() / "settings.toml")
            except:
                # If session_root not available, use fractalic_root
                settings_file = str(get_fractalic_root() / "settings.toml")
        except Exception as e:
            print(f"[WARNING] Could not locate settings file using path management: {e}")
            settings_file = 'settings.toml'  # Fallback to relative path
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for settings file at: {settings_file}")
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            return toml.load(f)
    except FileNotFoundError:
        print(f"[WARNING] Settings file {settings_file} not found. Using defaults.")
        return {}
    except toml.TomlDecodeError as e:
        print(f"[ERROR] Error parsing {settings_file}: {e}")
        return {}

def is_port_available(host='localhost', port=8001):
    """Check if a port is available on the given host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False

def find_available_port(start_port=8001, max_attempts=100):
    """Find the first available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port=port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

def check_docker_container_on_port(port):
    """Check if there's a Docker container using the specified port."""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', 'table {{.Names}}\t{{.Ports}}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if f':{port}->' in line or f':{port}/' in line:
                    container_name = line.split('\t')[0]
                    return container_name
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return None