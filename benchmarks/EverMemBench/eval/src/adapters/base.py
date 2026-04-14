"""
Base adapter for memory systems.

Defines abstract interface that all memory system adapters must implement.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.src.core.data_models import Dataset, AddResult, SearchResult


class BaseAdapter(ABC):
    """
    Abstract base class for memory system adapters.
    
    All adapters must implement:
    - add(): Ingest conversation data into the memory system
    - search(): Retrieve relevant memories for QA questions
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        """
        Initialize adapter with configuration.
        
        Args:
            config: Configuration dictionary (from YAML)
            output_dir: Optional output directory for results
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.name = config.get("name", self.__class__.__name__)
    
    @abstractmethod
    async def add(
        self,
        dataset: Dataset,
        user_id: str,
        days_to_process: Optional[List[str]] = None,
        **kwargs
    ) -> AddResult:
        """
        Add dataset to memory system.
        
        Args:
            dataset: Dataset object containing conversations
            user_id: User ID for the memory system
            days_to_process: Optional list of date strings to process (None = all)
            **kwargs: Additional adapter-specific parameters
            
        Returns:
            AddResult with success status and statistics
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search memories for a query.
        
        Args:
            query: Search query (usually the question text)
            user_id: User ID for the memory system
            top_k: Number of memories to retrieve
            **kwargs: Additional adapter-specific parameters (e.g., filters)
            
        Returns:
            SearchResult with retrieved memories and formatted context
        """
        pass
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Return system information.
        
        Returns:
            Dict with system name and metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
        }
    
    async def close(self):
        """
        Clean up resources.
        
        Override in subclasses that need cleanup (e.g., close HTTP sessions).
        """
        pass

