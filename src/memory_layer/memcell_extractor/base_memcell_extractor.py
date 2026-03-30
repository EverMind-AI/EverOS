"""
Simple Boundary Detection Base Class for EverMemOS

This module provides a simple and extensible base class for detecting
boundaries in various types of content (conversations, emails, notes, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from memory_layer.llm.llm_provider import LLMProvider
from api_specs.memory_types import RawDataType, BaseMemory, MemCell
from api_specs.dtos import RawData


@dataclass
class MemCellExtractRequest:
    history_raw_data_list: List[RawData]
    new_raw_data_list: List[RawData]
    # user id list of the entire group
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None

    old_memory_list: Optional[List[BaseMemory]] = None
    smart_mask_flag: Optional[bool] = False


@dataclass
class StatusResult:
    """Status control result."""

    # Indicates that when triggered next time, this conversation will be accumulated and input as new message
    should_wait: bool


class MemCellExtractor(ABC):
    def __init__(self, raw_data_type: RawDataType, llm_provider=LLMProvider):
        self.raw_data_type = raw_data_type
        self._llm_provider = llm_provider

    @abstractmethod
    async def extract_memcell(
        self, request: MemCellExtractRequest
    ) -> tuple[Optional[MemCell], Optional[StatusResult]]:
        pass
