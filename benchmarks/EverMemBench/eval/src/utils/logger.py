"""
Logging utilities.

Provides Rich console for beautiful output.
"""
from typing import Optional

from rich.console import Console


# Global console instance
_console: Optional[Console] = None


def get_console() -> Console:
    """
    Get global Rich console instance.
    
    Returns:
        Rich Console object
    """
    global _console
    if _console is None:
        _console = Console()
    return _console


def print_header(title: str, width: int = 60):
    """
    Print a formatted header.
    
    Args:
        title: Header title
        width: Header width in characters
    """
    console = get_console()
    console.print(f"\n{'='*width}", style="bold cyan")
    console.print(title, style="bold cyan")
    console.print(f"{'='*width}", style="bold cyan")


def print_success(message: str):
    """Print success message."""
    console = get_console()
    console.print(f"✅ {message}", style="bold green")


def print_error(message: str):
    """Print error message."""
    console = get_console()
    console.print(f"❌ {message}", style="bold red")


def print_warning(message: str):
    """Print warning message."""
    console = get_console()
    console.print(f"⚠️  {message}", style="yellow")


def print_info(message: str):
    """Print info message."""
    console = get_console()
    console.print(f"ℹ️  {message}", style="dim")

