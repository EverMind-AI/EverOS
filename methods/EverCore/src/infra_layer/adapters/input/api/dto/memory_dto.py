# -*- coding: utf-8 -*-
"""
Memory API DTO

Request and response data transfer objects for Memory API.
These models are re-exported from api_specs.dtos for backward compatibility.
"""

# Re-export from api_specs.dtos
from api_specs.dtos import (
    # Base API Response
    BaseApiResponse,
    # Add / Flush DTOs
    PersonalAddRequest,
    GroupAddRequest,
    PersonalFlushRequest,
    GroupFlushRequest,
    AddResult,
    AddResponse,
    FlushResult,
    FlushResponse,
    # Command DTOs
    DeleteMemoriesRequest as DeleteMemoriesRequestDTO,
    # Request DTOs
    RetrieveMemRequest,
    # Response DTOs (result data)
    RetrieveMemResponse,
    DeleteMemoriesResult,
    # API Response wrappers
    SearchMemoriesResponse,
)

# Backward compatibility aliases
SearchMemoriesRequest = RetrieveMemRequest

__all__ = [
    # Base Response
    "BaseApiResponse",
    # Add / Flush DTOs
    "PersonalAddRequest",
    "GroupAddRequest",
    "PersonalFlushRequest",
    "GroupFlushRequest",
    "AddResult",
    "AddResponse",
    "FlushResult",
    "FlushResponse",
    # Command DTOs
    "DeleteMemoriesRequest",
    "DeleteMemoriesRequestDTO",
    # Query DTOs (Requests)
    "RetrieveMemRequest",
    # Response DTOs (result data)
    "RetrieveMemResponse",
    "DeleteMemoriesResult",
    # API Response wrappers
    "SearchMemoriesResponse",
    # Backward compatibility aliases
    "SearchMemoriesRequest",
]
