"""
Evaluator registry - provide evaluator registration and creation.
Uses lazy loading strategy, keeps __init__.py empty.
"""
import importlib
from typing import Dict, Type, List
from evaluation.src.evaluators.base import BaseEvaluator


_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}

# Evaluator module mapping (for lazy loading)
_EVALUATOR_MODULES = {
    "llm_judge": "evaluation.src.evaluators.llm_judge",
    "exact_match": "evaluation.src.evaluators.exact_match",
    "hybrid": "evaluation.src.evaluators.hybrid",
    # Future evaluators:
    # "bert_score": "evaluation.src.evaluators.bert_score",
}


def register_evaluator(name: str):
    """
    Decorator for registering evaluators.
    
    Usage:
        @register_evaluator("llm_judge")
        class LLMJudge(BaseEvaluator):
            ...
    """
    def decorator(cls: Type[BaseEvaluator]):
        _EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_evaluator_loaded(name: str):
    """
    Ensure specified evaluator is loaded (lazy loading strategy).
    
    Trigger @register_evaluator decorator execution via dynamic import.
    This keeps __init__.py empty per project convention.
    
    Args:
        name: Evaluator name
        
    Raises:
        ValueError: If evaluator doesn't exist
        RuntimeError: If module loaded but not registered
    """
    if name in _EVALUATOR_REGISTRY:
        return  # Already loaded
    
    if name not in _EVALUATOR_MODULES:
        raise ValueError(
            f"Unknown evaluator: {name}. "
            f"Available evaluators: {list(_EVALUATOR_MODULES.keys())}"
        )
    
    # Dynamically import module, trigger @register_evaluator execution
    module_path = _EVALUATOR_MODULES[name]
    importlib.import_module(module_path)
    
    # Verify registration success
    if name not in _EVALUATOR_REGISTRY:
        raise RuntimeError(
            f"Evaluator '{name}' module loaded but not registered. "
            f"Check if @register_evaluator('{name}') decorator is present."
        )


def create_evaluator(name: str, llm_provider=None) -> BaseEvaluator:
    """
    Create evaluator instance.
    
    Args:
        name: Evaluator name
        llm_provider: LLM provider (required by some evaluators)
        
    Returns:
        Evaluator instance
        
    Raises:
        ValueError: If evaluator not registered
    """
    # Lazy loading: ensure evaluator loaded
    _ensure_evaluator_loaded(name)
    
    return _EVALUATOR_REGISTRY[name](llm_provider)


def list_evaluators() -> List[str]:
    """List all available evaluators."""
    return list(_EVALUATOR_MODULES.keys())
