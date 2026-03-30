"""
Adapter registry - provide adapter registration and creation.
Uses lazy loading strategy, keeps __init__.py empty.
"""
import importlib
from typing import Dict, Type, List
from evaluation.src.adapters.base import BaseAdapter


_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}

# Adapter module mapping (for lazy loading)
_ADAPTER_MODULES = {
    # Local systems
    "evermemos": "evaluation.src.adapters.evermemos_adapter",
    
    # Online API systems
    "mem0": "evaluation.src.adapters.mem0_adapter",
    "memos": "evaluation.src.adapters.memos_adapter",
    "memu": "evaluation.src.adapters.memu_adapter",
    "zep": "evaluation.src.adapters.zep_adapter",
    "evermemos_api": "evaluation.src.adapters.evermemos_api_adapter",
    "memobase": "evaluation.src.adapters.memobase_adapter",
    "supermemory": "evaluation.src.adapters.supermemory_adapter",
    
    # Future systems:
    # "nemori": "evaluation.src.adapters.nemori_adapter",
}


def register_adapter(name: str):
    """
    Decorator for registering adapters.
    
    Usage:
        @register_adapter("evermemos")
        class EverMemOSAdapter(BaseAdapter):
            ...
    """
    def decorator(cls: Type[BaseAdapter]):
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_adapter_loaded(name: str):
    """
    Ensure specified adapter is loaded (lazy loading strategy).
    
    Trigger @register_adapter decorator execution via dynamic import.
    This keeps __init__.py empty per project convention.
    
    Args:
        name: Adapter name
        
    Raises:
        ValueError: If adapter doesn't exist
        RuntimeError: If module loaded but not registered
    """
    if name in _ADAPTER_REGISTRY:
        return  # Already loaded
    
    if name not in _ADAPTER_MODULES:
        raise ValueError(
            f"Unknown adapter: {name}. "
            f"Available adapters: {list(_ADAPTER_MODULES.keys())}"
        )
    
    # Dynamically import module, trigger @register_adapter execution
    module_path = _ADAPTER_MODULES[name]
    importlib.import_module(module_path)
    
    # Verify registration success
    if name not in _ADAPTER_REGISTRY:
        raise RuntimeError(
            f"Adapter '{name}' module loaded but not registered. "
            f"Check if @register_adapter('{name}') decorator is present."
        )


def create_adapter(name: str, config: dict, output_dir = None) -> BaseAdapter:
    """
    Create adapter instance.
    
    Args:
        name: Adapter name
        config: Config dict
        output_dir: Output directory (for persistence, optional)
        
    Returns:
        Adapter instance
        
    Raises:
        ValueError: If adapter not registered
    """
    # Lazy loading: ensure adapter loaded
    _ensure_adapter_loaded(name)
    
    # Try passing output_dir, fallback if adapter doesn't support it
    try:
        return _ADAPTER_REGISTRY[name](config, output_dir=output_dir)
    except TypeError:
        # Adapter doesn't accept output_dir parameter, use default creation
        return _ADAPTER_REGISTRY[name](config)


def list_adapters() -> List[str]:
    """List all available adapters."""
    return list(_ADAPTER_MODULES.keys())
