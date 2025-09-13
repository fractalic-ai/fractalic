"""
Fractalic - The Agentic Development Environment

Turn simple documents into powerful, AI-native applications.
"""

__version__ = "0.1.2"

# Expose main functions without importing dependencies yet
def main():
    """CLI entry point for Fractalic."""
    from .fractalic import main as _main
    return _main()

def run_fractalic(*args, **kwargs):
    """Programmatic entry point for Fractalic."""
    from .fractalic import run_fractalic as _run_fractalic
    return _run_fractalic(*args, **kwargs)

__all__ = ["main", "run_fractalic", "__version__"]
