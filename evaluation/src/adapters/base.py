"""
Adapter base class - define unified memory system adapter interface.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from evaluation.src.core.data_models import Conversation, SearchResult


class BaseAdapter(ABC):
    """Memory system adapter base class."""
    
    def __init__(self, config: dict):
        """
        Initialize adapter.
        
        Args:
            config: System config dict
        """
        self.config = config
    
    @abstractmethod
    async def add(
        self, 
        conversations: List[Conversation],
        **kwargs
    ) -> Any:
        """
        Ingest conversation data and build index (Add stage).
        
        This method encapsulates system-specific data ingestion and index building:
        - For EverMemOS: MemCell extraction + BM25/Embedding index building
        - For Mem0: Direct storage to vector database
        - For other systems: Their respective implementations
        
        Args:
            conversations: Standard format conversation list
            **kwargs: Extra parameters
            
        Returns:
            Index object (system internal format, different systems return different types)
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs
    ) -> SearchResult:
        """
        Retrieve relevant memories (Search stage).
        
        Args:
            query: Query text
            conversation_id: Conversation ID
            index: Index object (returned by add())
            **kwargs: Extra parameters (e.g., top_k)
            
        Returns:
            Standard format search result
        """
        pass
    
    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        """
        Preparation stage: operations executed before add.
        
        Optional preparation operations, e.g.:
        - Update project config (e.g., Mem0's custom_instructions)
        - Clean existing data (if clean_before_add configured)
        - Other system-specific initialization
        
        Args:
            conversations: Standard format conversation list (for extracting user_id etc.)
            **kwargs: Extra parameters
        
        Returns:
            None
        """
        pass  # Default: no operation
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Return system info (for result recording).
        
        Returns:
            System info dict
        """
        return {
            "name": self.__class__.__name__,
            "config": self.config
        }
    
    def build_lazy_index(self, conversations: List[Conversation], output_dir: Any) -> Any:
        """
        Build lazy-loaded index metadata.
        
        Default: return None (online API systems don't need index)
        Local systems (e.g., EverMemOS) should override this method
        
        Args:
            conversations: Conversation list
            output_dir: Output directory
            
        Returns:
            Index object or metadata (local systems return index metadata, online systems return None)
        """
        return None

