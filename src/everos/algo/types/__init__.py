"""Public data contracts shared by the algorithms and their hosts.

This module re-exports the stable cross-distribution contracts. Adding another public type is a SemVer minor bump for
users that import from this module.
"""

from __future__ import annotations

__all__ = [
    "EXTENSION_TO_MODALITY",
    "MIME_TO_EXTENSION",
    "MIME_TO_MODALITY",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MIMES",
    "AgentCase",
    "AgentProfilePatch",
    "AgentProfileSignal",
    "AgentProfileUpdate",
    "AgentSkill",
    "AtomicFact",
    "Candidate",
    "CategorySpec",
    "ChatMessage",
    "ConversationItem",
    "Episode",
    "ExplicitInfoItem",
    "FactCandidate",
    "Foresight",
    "ImplicitTraitItem",
    "KnowledgeMemory",
    "MemCell",
    "Modality",
    "ParsedContent",
    "Profile",
    "RankInput",
    "RankOutput",
    "RawData",
    "RawFile",
    "ScoredItem",
    "ToolCall",
    "ToolCallFunction",
    "ToolCallRequest",
    "ToolCallResult",
    "get_extension_from_mime",
    "get_modality",
    "get_modality_from_mime",
]

import logging

from everos.algo.types.agent import (
    AgentCase,
    AgentProfilePatch,
    AgentProfileSignal,
    AgentProfileUpdate,
    AgentSkill,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)
from everos.algo.types.chat import ChatMessage
from everos.algo.types.conversation import ConversationItem, MemCell
from everos.algo.types.knowledge import CategorySpec, KnowledgeMemory
from everos.algo.types.memories import (
    AtomicFact,
    Episode,
    ExplicitInfoItem,
    Foresight,
    ImplicitTraitItem,
    Profile,
)
from everos.algo.types.modality import (
    EXTENSION_TO_MODALITY,
    MIME_TO_EXTENSION,
    MIME_TO_MODALITY,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIMES,
    Modality,
    get_extension_from_mime,
    get_modality,
    get_modality_from_mime,
)
from everos.algo.types.parsed import ParsedContent
from everos.algo.types.rank import (
    Candidate,
    FactCandidate,
    RankInput,
    RankOutput,
    ScoredItem,
)
from everos.algo.types.raw import RawData, RawFile

# Library-style logging: silent until the application configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())
