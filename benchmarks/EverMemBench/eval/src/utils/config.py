"""
Configuration utilities.

Provides YAML loading with environment variable substitution.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML file with environment variable substitution.
    
    Supports formats:
    - ${VAR_NAME} - Required environment variable
    - ${VAR_NAME:default} - With default value
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Parsed configuration dict
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required env var is missing
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Substitute environment variables
    content = _substitute_env_vars(content)
    
    return yaml.safe_load(content) or {}


def _substitute_env_vars(content: str) -> str:
    """
    Substitute ${VAR_NAME} and ${VAR_NAME:default} patterns.
    
    Args:
        content: YAML content string
        
    Returns:
        Content with substituted values
    """
    # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
    pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}'
    
    def replacer(match):
        var_name = match.group(1)
        default = match.group(2)
        
        value = os.environ.get(var_name)
        
        if value is not None:
            return value
        elif default is not None:
            return default
        else:
            # Keep original for debugging (will likely cause error later)
            return match.group(0)
    
    return re.sub(pattern, replacer, content)


def get_config_path(config_name: str) -> Path:
    """
    Get path to a config file in eval/config/ directory.
    
    Args:
        config_name: Config file name (e.g., "memos.yaml")
        
    Returns:
        Full path to config file
    """
    eval_root = Path(__file__).parent.parent.parent
    return eval_root / "config" / config_name

