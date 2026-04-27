"""DTO (Data Transfer Object) types for API specifications.

This package organizes DTOs by resource type:
- base: Common base types (BaseApiResponse)
- memory: Memory resource DTOs (memorize, search, get, delete)
- settings: Global settings resource DTOs
- group: Group resource DTOs
- sender: Sender resource DTOs
"""

# Base types
from api_specs.dtos.base import BaseApiResponse, T

# Memory resource DTOs
from api_specs.dtos.memory import (
    # Raw data
    RawData,
    # Memorize (internal)
    MemorizeRequest,
    # Add / Flush
    PersonalAddRequest,
    GroupAddRequest,
    PersonalFlushRequest,
    GroupFlushRequest,
    AddResult,
    AddResponse,
    FlushResult,
    FlushResponse,
    # Search/Retrieve
    RetrieveMemRequest,
    RawMessageDTO,
    ProfileSearchItem,
    RetrieveMemResponse,
    SearchMemoriesResponse,
    # Get
    GetMemRequest,
    EpisodeItem,
    ProfileItem,
    GetMemResponse,
    GetMemoriesResponse,
    # Delete
    DeleteMemoriesResult,
    DeleteMemoriesResponse,
)

# Memory delete DTOs (v1)
from api_specs.dtos.memory_delete import DeleteMemoriesRequest

# Group resource DTOs
from api_specs.dtos.group import (
    CreateGroupRequest,
    PatchGroupRequest,
    GroupResponse,
    CreateGroupApiResponse,
    GetGroupApiResponse,
    PatchGroupApiResponse,
)

# Sender resource DTOs
from api_specs.dtos.sender import (
    CreateSenderRequest,
    PatchSenderRequest,
    SenderResponse,
    CreateSenderApiResponse,
    GetSenderApiResponse,
    PatchSenderApiResponse,
)

# Settings resource DTOs
from api_specs.dtos.settings import (
    LlmProviderConfig,
    LlmCustomSetting,
    UpdateSettingsRequest,
    SettingsResponse,
    GetSettingsApiResponse,
    UpdateSettingsApiResponse,
)

__all__ = [
    # Base
    "BaseApiResponse",
    "T",
    # Memory - Raw data
    "RawData",
    # Memory - Memorize (internal)
    "MemorizeRequest",
    # Memory - Add / Flush
    "PersonalAddRequest",
    "GroupAddRequest",
    "PersonalFlushRequest",
    "GroupFlushRequest",
    "AddResult",
    "AddResponse",
    "FlushResult",
    "FlushResponse",
    # Memory - Search/Retrieve
    "RetrieveMemRequest",
    "RawMessageDTO",
    "ProfileSearchItem",
    "RetrieveMemResponse",
    "SearchMemoriesResponse",
    # Memory - Delete
    "DeleteMemoriesRequest",
    # Memory - Get
    "GetMemRequest",
    "EpisodeItem",
    "ProfileItem",
    "GetMemResponse",
    "GetMemoriesResponse",
    # Settings
    "LlmProviderConfig",
    "LlmCustomSetting",
    "UpdateSettingsRequest",
    "SettingsResponse",
    "GetSettingsApiResponse",
    "UpdateSettingsApiResponse",
    # Group
    "CreateGroupRequest",
    "PatchGroupRequest",
    "GroupResponse",
    "CreateGroupApiResponse",
    "GetGroupApiResponse",
    "PatchGroupApiResponse",
    # Sender
    "CreateSenderRequest",
    "PatchSenderRequest",
    "SenderResponse",
    "CreateSenderApiResponse",
    "GetSenderApiResponse",
    "PatchSenderApiResponse",
]
