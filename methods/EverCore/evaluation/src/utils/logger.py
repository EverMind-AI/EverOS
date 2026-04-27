"""
Logging utilities.

Provides unified logging functionality.
"""
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    name: str = "evaluation"
) -> logging.Logger:
    """
    Setup logger.
    
    Args:
        log_file: Log file path (optional)
        level: Log level
        name: Logger name
        
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add Rich Console Handler (colored output)
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Add file Handler (if log file is specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_console() -> Console:
    """Get Rich Console instance."""
    return Console()

