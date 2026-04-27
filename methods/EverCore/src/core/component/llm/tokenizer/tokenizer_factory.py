"""
Tokenizer Factory

Provides tokenizer caching and management functionality.
Supports tiktoken and potentially other tokenizer providers in the future.
"""

from typing import Dict, Any
import tiktoken

from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


# Default tiktoken encodings to preload during application startup
DEFAULT_TIKTOKEN_ENCODINGS = [
    "o200k_base",   # GPT-4o, GPT-4o-mini
    "cl100k_base",  # GPT-4, GPT-3.5-turbo, text-embedding-ada-002
]


@component(name="tokenizer_factory", primary=True)
class TokenizerFactory:
    """
    Tokenizer Factory
    
    Provides tokenizer caching and management functionality.
    Cache key format: "{provider}:{encoding_name}" (e.g., "tiktoken:o200k_base")
    """

    def __init__(self):
        """Initialize tokenizer factory"""
        self._tokenizers: Dict[str, Any] = {}
        logger.info("TokenizerFactory initialized")

    def get_tokenizer_from_tiktoken(self, encoding_name: str) -> tiktoken.Encoding:
        """
        Get a tiktoken tokenizer by encoding name, with caching.
        
        Args:
            encoding_name: The name of the tiktoken encoding (e.g., "o200k_base", "cl100k_base")
            
        Returns:
            tiktoken.Encoding: The tokenizer instance
            
        Example:
            >>> tokenizer = factory.get_tokenizer_from_tiktoken("o200k_base")
            >>> tokens = tokenizer.encode("Hello, world!")
        """
        cache_key = f"tiktoken:{encoding_name}"
        
        if cache_key not in self._tokenizers:
            logger.debug("Loading tiktoken encoding: %s", encoding_name)
            self._tokenizers[cache_key] = tiktoken.get_encoding(encoding_name)
            logger.debug("Tiktoken encoding '%s' loaded and cached", encoding_name)
        
        return self._tokenizers[cache_key]

    def load_default_encodings(self) -> None:
        """
        Preload default tiktoken encodings during application startup.
        
        This method should be called during application lifespan startup
        to ensure tokenizers are ready before handling requests.
        
        The encodings loaded are defined in DEFAULT_TIKTOKEN_ENCODINGS.
        """
        logger.info("Preloading %d tiktoken encodings...", len(DEFAULT_TIKTOKEN_ENCODINGS))
        
        for encoding_name in DEFAULT_TIKTOKEN_ENCODINGS:
            try:
                self.get_tokenizer_from_tiktoken(encoding_name)
                logger.info("Successfully preloaded tiktoken encoding: %s", encoding_name)
            except (ValueError, KeyError, RuntimeError) as e:
                logger.error("Failed to preload tiktoken encoding '%s': %s", encoding_name, e)
        
        logger.info("Tiktoken encodings preload completed")

    def get_cached_tokenizer_count(self) -> int:
        """
        Get the number of cached tokenizers.
        
        Returns:
            int: Number of tokenizers currently in cache
        """
        return len(self._tokenizers)

    def clear_cache(self) -> None:
        """
        Clear the tokenizer cache.
        
        This is mainly useful for testing purposes.
        """
        self._tokenizers.clear()
        logger.debug("Tokenizer cache cleared")
