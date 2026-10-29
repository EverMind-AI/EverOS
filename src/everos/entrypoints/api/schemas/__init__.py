"""Request and response models of the HTTP API, one module per resource.

Routes import from this facade; the class names are the OpenAPI schema names,
so renaming one is a breaking change to ``docs/openapi.json``.
"""

from __future__ import annotations

from .cascade import QuiesceResponse as QuiesceResponse
from .envelope import ErrorDetail as ErrorDetail
from .envelope import ErrorResponse as ErrorResponse
from .envelope import SuccessEnvelope as SuccessEnvelope
from .health import CascadeHealthBlock as CascadeHealthBlock
from .health import HealthCapabilities as HealthCapabilities
from .health import HealthResponse as HealthResponse
from .health import MemorizeQueueHealthBlock as MemorizeQueueHealthBlock
from .ids import AppScopeId as AppScopeId
from .ids import PathSafeId as PathSafeId
from .ids import ProjectScopeId as ProjectScopeId
from .knowledge import CategoryDTO as CategoryDTO
from .knowledge import CategoryListResponse as CategoryListResponse
from .knowledge import DocumentContextDTO as DocumentContextDTO
from .knowledge import DocumentCreateResponse as DocumentCreateResponse
from .knowledge import DocumentDeleteResponse as DocumentDeleteResponse
from .knowledge import DocumentDetailResponse as DocumentDetailResponse
from .knowledge import DocumentListResponse as DocumentListResponse
from .knowledge import DocumentOverviewItemDTO as DocumentOverviewItemDTO
from .knowledge import DocumentPatchRequest as DocumentPatchRequest
from .knowledge import DocumentPatchResponse as DocumentPatchResponse
from .knowledge import KnowledgeSearchRequest as KnowledgeSearchRequest
from .knowledge import KnowledgeSearchResponse as KnowledgeSearchResponse
from .knowledge import SearchHitDTO as SearchHitDTO
from .knowledge import TopicDetailResponse as TopicDetailResponse
from .knowledge import TopicOverviewDTO as TopicOverviewDTO
from .memorize import AcceptedResponseData as AcceptedResponseData
from .memorize import AddResponseData as AddResponseData
from .memorize import ContentItemDTO as ContentItemDTO
from .memorize import FlushResponseData as FlushResponseData
from .memorize import MemorizeAddRequest as MemorizeAddRequest
from .memorize import MemorizeFlushRequest as MemorizeFlushRequest
from .memorize import MessageItemDTO as MessageItemDTO
from .memorize import ToolCallDTO as ToolCallDTO
from .memorize import ToolFunctionDTO as ToolFunctionDTO
from .ome import RunSummary as RunSummary
from .ome import TriggerRequest as TriggerRequest
from .ome import TriggerResponse as TriggerResponse

__all__ = [
    "AcceptedResponseData",
    "AddResponseData",
    "AppScopeId",
    "CascadeHealthBlock",
    "CategoryDTO",
    "CategoryListResponse",
    "ContentItemDTO",
    "DocumentContextDTO",
    "DocumentCreateResponse",
    "DocumentDeleteResponse",
    "DocumentDetailResponse",
    "DocumentListResponse",
    "DocumentOverviewItemDTO",
    "DocumentPatchRequest",
    "DocumentPatchResponse",
    "ErrorDetail",
    "ErrorResponse",
    "FlushResponseData",
    "HealthCapabilities",
    "HealthResponse",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResponse",
    "MemorizeAddRequest",
    "MemorizeFlushRequest",
    "MemorizeQueueHealthBlock",
    "MessageItemDTO",
    "PathSafeId",
    "ProjectScopeId",
    "QuiesceResponse",
    "RunSummary",
    "SearchHitDTO",
    "SuccessEnvelope",
    "ToolCallDTO",
    "ToolFunctionDTO",
    "TopicDetailResponse",
    "TopicOverviewDTO",
    "TriggerRequest",
    "TriggerResponse",
]
