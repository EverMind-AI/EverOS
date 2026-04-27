"""
Converter registry - provide converter registration and creation.
Uses lazy loading strategy, keeps __init__.py empty.
"""
import importlib
from typing import Dict, Type, List, Optional
from evaluation.src.converters.base import BaseConverter


_CONVERTER_REGISTRY: Dict[str, Type[BaseConverter]] = {}

# Converter module mapping (for lazy loading)
_CONVERTER_MODULES = {
    "longmemeval": "evaluation.src.converters.longmemeval_converter",
    "personamem": "evaluation.src.converters.personamem_converter",
    # Future converters
}


def register_converter(name: str):
    """
    Decorator for registering converters.
    
    Usage:
        @register_converter("longmemeval")
        class LongMemEvalConverter(BaseConverter):
            ...
    """
    def decorator(cls: Type[BaseConverter]):
        _CONVERTER_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_converter_loaded(name: str):
    """
    Ensure specified converter is loaded (lazy loading strategy).
    
    Args:
        name: Converter name
        
    Raises:
        ValueError: If converter doesn't exist
        RuntimeError: If module loaded but not registered
    """
    if name in _CONVERTER_REGISTRY:
        return  # Already loaded
    
    if name not in _CONVERTER_MODULES:
        raise ValueError(
            f"Unknown converter: {name}. "
            f"Available converters: {list(_CONVERTER_MODULES.keys())}"
        )
    
    # Dynamically import module, trigger @register_converter execution
    module_path = _CONVERTER_MODULES[name]
    importlib.import_module(module_path)
    
    # Verify registration success
    if name not in _CONVERTER_REGISTRY:
        raise RuntimeError(
            f"Converter '{name}' module loaded but not registered. "
            f"Check if @register_converter('{name}') decorator is present."
        )


def get_converter(name: str) -> Optional[BaseConverter]:
    """
    Get converter instance (if exists).
    
    Args:
        name: Converter name
        
    Returns:
        Converter instance, or None if not exists
    """
    if name not in _CONVERTER_MODULES:
        return None  # Dataset doesn't need conversion
    
    # Lazy loading: ensure converter loaded
    _ensure_converter_loaded(name)
    
    return _CONVERTER_REGISTRY[name]()


def list_converters() -> List[str]:
    """List all available converters."""
    return list(_CONVERTER_MODULES.keys())


