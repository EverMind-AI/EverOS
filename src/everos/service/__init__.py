"""Application layer.

Orchestrates memory-layer capabilities into complete use cases. One CLI
command or API endpoint maps to one service method.

External usage:
    from everos.service import MemorizeResult, get, memorize, search
    from everos.service import (
        CategoryOverview, CreateDocumentResult, DuplicateDocumentError,
        ExtractionEmptyError,
        create_document, DocumentContext, DocumentDetail, DocumentListResult,
        DocumentNotFoundError, DocumentOverviewItem, DeleteResult,
        PatchResult, SearchHit, SearchKnowledgeResult,
        TopicDetail, TopicOverview, TopicNotFoundError,
        delete_document, get_document, get_topic, list_categories,
        list_documents, patch_document, replace_document, search_knowledge,
    )
"""

from everos.core.errors import DocumentNotFoundError as DocumentNotFoundError
from everos.core.errors import DuplicateDocumentError as DuplicateDocumentError
from everos.core.errors import ExtractionEmptyError as ExtractionEmptyError
from everos.core.errors import TopicNotFoundError as TopicNotFoundError

from .get import get as get
from .knowledge import CategoryOverview as CategoryOverview
from .knowledge import CreateDocumentResult as CreateDocumentResult
from .knowledge import DeleteResult as DeleteResult
from .knowledge import DocumentContext as DocumentContext
from .knowledge import DocumentDetail as DocumentDetail
from .knowledge import DocumentListResult as DocumentListResult
from .knowledge import DocumentOverviewItem as DocumentOverviewItem
from .knowledge import PatchResult as PatchResult
from .knowledge import SearchHit as SearchHit
from .knowledge import SearchKnowledgeResult as SearchKnowledgeResult
from .knowledge import TopicDetail as TopicDetail
from .knowledge import TopicOverview as TopicOverview
from .knowledge import compile_knowledge_where as compile_knowledge_where
from .knowledge import create_document as create_document
from .knowledge import delete_document as delete_document
from .knowledge import get_document as get_document
from .knowledge import get_topic as get_topic
from .knowledge import list_categories as list_categories
from .knowledge import list_documents as list_documents
from .knowledge import patch_document as patch_document
from .knowledge import replace_document as replace_document
from .knowledge import search_knowledge as search_knowledge
from .memorize import MemorizeResult as MemorizeResult
from .memorize import memorize as memorize
from .search import search as search

__all__ = [
    "CategoryOverview",
    "CreateDocumentResult",
    "DeleteResult",
    "DocumentContext",
    "DocumentDetail",
    "DocumentListResult",
    "DocumentNotFoundError",
    "DocumentOverviewItem",
    "DuplicateDocumentError",
    "ExtractionEmptyError",
    "MemorizeResult",
    "PatchResult",
    "SearchHit",
    "SearchKnowledgeResult",
    "TopicDetail",
    "TopicNotFoundError",
    "TopicOverview",
    "compile_knowledge_where",
    "create_document",
    "delete_document",
    "get",
    "get_document",
    "get_topic",
    "list_categories",
    "list_documents",
    "memorize",
    "patch_document",
    "replace_document",
    "search",
    "search_knowledge",
]
