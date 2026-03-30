"""
Simple Memory Extraction Base Class for EverMemOS

This module provides a simple base class for extracting memories
from boundary detection results (BoundaryResult).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from api_specs.memory_types import MemoryType, BaseMemory, MemCell


@dataclass
class MemoryExtractRequest:
    """
    Base class for memory extraction requests
    """

    memcell: MemCell
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    participants: Optional[List[str]] = None

    old_memory_list: Optional[List[BaseMemory]] = None

    user_organization: Optional[List] = None


class MemoryExtractor(ABC):
    """
    Simple abstract base class for memory extraction.

    This class provides a minimal interface for extracting memories
    from boundary detection results.
    """

    def __init__(self, memory_type: MemoryType):
        """
        Initialize the memory extractor.

        Args:
            memory_type: The type of memory this extractor generates
        """
        self.memory_type = memory_type

    @abstractmethod
    async def extract_memory(self, request: MemoryExtractRequest) -> Optional[BaseMemory]:
        """
        Extract memory from a boundary detection result.

        Args:
            boundary_result: The boundary detection result to extract from
            user_id: User ID for the memory

        Returns:
            MemoryExtractionResult if extraction is successful, None otherwise
        """
        pass

    def __str__(self) -> str:
        """String representation of the extractor."""
        return f"{self.__class__.__name__}(type={self.memory_type.value})"
