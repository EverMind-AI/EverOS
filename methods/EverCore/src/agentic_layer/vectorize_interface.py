"""
Vectorize Service Interface
向量化服务接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class UsageInfo:
    """Token usage information"""

    prompt_tokens: int
    total_tokens: int

    @classmethod
    def from_openai_usage(cls, usage) -> "UsageInfo":
        """Create UsageInfo object from OpenAI usage object"""
        return cls(prompt_tokens=usage.prompt_tokens, total_tokens=usage.total_tokens)


class VectorizeServiceInterface(ABC):
    """Vectorization service interface"""

    @abstractmethod
    async def get_embedding(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> np.ndarray:
        """Get embedding for a single text"""
        pass

    @abstractmethod
    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        """Get embedding with usage information"""
        pass

    @abstractmethod
    async def get_embeddings(
        self, texts: List[str], instruction: Optional[str] = None, is_query: bool = False
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts"""
        pass

    @abstractmethod
    async def get_embeddings_batch(
        self, text_batches: List[List[str]], instruction: Optional[str] = None, is_query: bool = False
    ) -> List[List[np.ndarray]]:
        """Get embeddings for multiple batches"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the current model name"""
        pass

    @abstractmethod
    async def close(self):
        """Close and cleanup resources"""
        pass


class VectorizeError(Exception):
    """Vectorize API error exception class"""
    pass

