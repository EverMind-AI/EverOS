"""
Elasticsearch document converter base class

Provides basic functionality for converting data from arbitrary sources to Elasticsearch documents.
All ES document converters should inherit from this base class to obtain a unified conversion interface.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Any, get_args, get_origin
from core.oxm.es.doc_base import DocBase
from core.observation.logger import get_logger

logger = get_logger(__name__)

# Generic type variable - only constrains ES document type
EsDocType = TypeVar('EsDocType', bound=DocBase)


class BaseEsConverter(ABC, Generic[EsDocType]):
    """
    Elasticsearch document converter base class

    Provides basic functionality for converting data from arbitrary sources to Elasticsearch documents.
    All ES document converters should inherit from this class.

    Features:
    - Unified conversion interface (class methods)
    - Type-safe generic support for ES documents
    - Automatically retrieves ES document type from generics
    - Flexible data source support
    """

    @classmethod
    def get_es_model(cls) -> Type[EsDocType]:
        """
        Get the ES document model type from generic information

        Returns:
            Type[EsDocType]: ES document model class
        """
        # Get the generic base class of the current class
        if hasattr(cls, '__orig_bases__'):
            for base in cls.__orig_bases__:
                if get_origin(base) is BaseEsConverter:
                    args = get_args(base)
                    if args:
                        return args[0]

        raise ValueError(
            f"Cannot obtain ES document type from generic information of {cls.__name__}"
        )

    @classmethod
    @abstractmethod
    def from_mongo(cls, source_doc: Any) -> EsDocType:
        """
        Convert from source data to Elasticsearch document

        This is the core conversion method; subclasses must implement specific conversion logic.

        Args:
            source_doc: Source data (can be of any type)

        Returns:
            EsDocType: Elasticsearch document instance

        Raises:
            Exception: When an error occurs during conversion
        """
        raise NotImplementedError("Subclasses must implement the from_mongo method")
