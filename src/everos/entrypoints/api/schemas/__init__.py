"""Request and response models of the HTTP API, one module per resource.

Routes import from this facade; the class names are the OpenAPI schema names,
so renaming one is a breaking change to ``docs/openapi.json``.
"""

from __future__ import annotations

from .cascade import QuiesceResponse as QuiesceResponse
from .delete import DeleteByIdsRequest as DeleteByIdsRequest
from .delete import DeleteByIdsResponseData as DeleteByIdsResponseData
from .delete import DeleteRequest as DeleteRequest
from .delete import DeleteResponseData as DeleteResponseData
from .delete import GroupDeleteResponseData as GroupDeleteResponseData
from .envelope import ErrorDetail as ErrorDetail
from .envelope import ErrorResponse as ErrorResponse
from .envelope import SuccessEnvelope as SuccessEnvelope
from .forget import ForgetImageRequest as ForgetImageRequest
from .forget import ForgetImageResponseData as ForgetImageResponseData
from .forget import ForgetSessionRequest as ForgetSessionRequest
from .forget import ForgetSessionResponseData as ForgetSessionResponseData
from .forget import ForgottenPictureData as ForgottenPictureData
from .forget import ReviseImageMemoryRequest as ReviseImageMemoryRequest
from .forget import ReviseImageMemoryResponseData as ReviseImageMemoryResponseData
from .forget import UndoForgetRequest as UndoForgetRequest
from .forget import UndoForgetResponseData as UndoForgetResponseData
from .forget import UndoForgetSessionRequest as UndoForgetSessionRequest
from .forget import UndoForgetSessionResponseData as UndoForgetSessionResponseData
from .group_forget import ForgetRequest as ForgetRequest
from .group_forget import ForgottenMessageData as ForgottenMessageData
from .group_forget import GroupForgetRequest as GroupForgetRequest
from .group_forget import GroupForgetResponseData as GroupForgetResponseData
from .group_forget import GroupUndoForgetRequest as GroupUndoForgetRequest
from .group_forget import GroupUndoForgetResponseData as GroupUndoForgetResponseData
from .group_forget import UndoRequest as UndoRequest
from .groups import GroupDTO as GroupDTO
from .groups import GroupUpsertRequest as GroupUpsertRequest
from .groups import GroupUpsertResponseData as GroupUpsertResponseData
from .groups import MemberIntervalDTO as MemberIntervalDTO
from .groups import RosterChangeRequest as RosterChangeRequest
from .groups import RosterChangeResponseData as RosterChangeResponseData
from .groups import RosterEventDTO as RosterEventDTO
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
from .projects import ProjectDTO as ProjectDTO
from .projects import ProjectGroupDTO as ProjectGroupDTO
from .projects import ProjectMemberDTO as ProjectMemberDTO
from .projects import ProjectProfileDTO as ProjectProfileDTO

__all__ = [
    "AcceptedResponseData",
    "AddResponseData",
    "AppScopeId",
    "CascadeHealthBlock",
    "CategoryDTO",
    "CategoryListResponse",
    "ContentItemDTO",
    "DeleteByIdsRequest",
    "DeleteByIdsResponseData",
    "DeleteRequest",
    "DeleteResponseData",
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
    "ForgetImageRequest",
    "ForgetImageResponseData",
    "ForgetRequest",
    "ForgetSessionRequest",
    "ForgetSessionResponseData",
    "ForgottenMessageData",
    "ForgottenPictureData",
    "GroupDTO",
    "GroupDeleteResponseData",
    "GroupForgetRequest",
    "GroupForgetResponseData",
    "GroupUndoForgetRequest",
    "GroupUndoForgetResponseData",
    "GroupUpsertRequest",
    "GroupUpsertResponseData",
    "HealthCapabilities",
    "HealthResponse",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResponse",
    "MemberIntervalDTO",
    "MemorizeAddRequest",
    "MemorizeFlushRequest",
    "MemorizeQueueHealthBlock",
    "MessageItemDTO",
    "PathSafeId",
    "ProjectDTO",
    "ProjectGroupDTO",
    "ProjectMemberDTO",
    "ProjectProfileDTO",
    "ProjectScopeId",
    "QuiesceResponse",
    "ReviseImageMemoryRequest",
    "ReviseImageMemoryResponseData",
    "RosterChangeRequest",
    "RosterChangeResponseData",
    "RosterEventDTO",
    "RunSummary",
    "SearchHitDTO",
    "SuccessEnvelope",
    "ToolCallDTO",
    "ToolFunctionDTO",
    "TopicDetailResponse",
    "TopicOverviewDTO",
    "TriggerRequest",
    "TriggerResponse",
    "UndoForgetRequest",
    "UndoForgetResponseData",
    "UndoForgetSessionRequest",
    "UndoForgetSessionResponseData",
    "UndoRequest",
]
