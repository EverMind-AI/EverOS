"""
Searchers package for different search implementations.
"""

from enum import Enum

from .base import BaseSearcher


class SearcherType:
    """Registry for available searcher types with lazy imports."""

    _REGISTRY = {
        "bm25": ".bm25_searcher.BM25Searcher",
        "faiss": ".faiss_searcher.FaissSearcher",
        "reasonir": ".faiss_searcher.ReasonIrSearcher",
        "custom": ".custom_searcher.CustomSearcher",
    }

    @classmethod
    def get_choices(cls):
        return list(cls._REGISTRY.keys())

    @classmethod
    def get_searcher_class(cls, cli_name):
        if cli_name not in cls._REGISTRY:
            raise ValueError(f"Unknown searcher type: {cli_name}")
        module_path, class_name = cls._REGISTRY[cli_name].rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path, package=__package__)
        return getattr(mod, class_name)


__all__ = ["BaseSearcher", "SearcherType"]
