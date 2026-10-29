"""In-memory test doubles for the memory domain's ports.

Named after the Haystack ``InMemoryDocumentStore`` convention: each
``InMemory*`` class implements one port with plain containers a test can
seed and inspect. :func:`make_deps` assembles them into a
:class:`~everos.memory.deps.MemoryDeps` for calling a strategy directly::

    deps = make_deps()
    events = await extract_atomic_facts(event, deps)
    assert deps.stores.atomic_fact.records

:func:`make_application` does the same one level up: an
:class:`~everos.service.Application` whose facades are recording stubs.
"""

from __future__ import annotations

from .application import StubMemoryService as StubMemoryService
from .application import StubOpsService as StubOpsService
from .application import make_application as make_application
from .application import make_backfill_context as make_backfill_context
from .application import make_knowledge_service as make_knowledge_service
from .application import make_memory_service as make_memory_service
from .application import make_ops as make_ops
from .deps import bundled_instructions as bundled_instructions
from .deps import make_deps as make_deps
from .deps import make_indexes as make_indexes
from .deps import make_stores as make_stores
from .deps import unscripted_llm as unscripted_llm
from .embedding import FakeEmbeddingProvider as FakeEmbeddingProvider
from .indexes import InMemoryAgentCaseIndex as InMemoryAgentCaseIndex
from .indexes import InMemoryAgentSkillIndex as InMemoryAgentSkillIndex
from .indexes import InMemoryAtomicFactIndex as InMemoryAtomicFactIndex
from .indexes import InMemoryEpisodeIndex as InMemoryEpisodeIndex
from .indexes import InMemoryProfileIndex as InMemoryProfileIndex
from .llm import AGENT_TRACK_OPERATORS as AGENT_TRACK_OPERATORS
from .llm import KNOWLEDGE_OPERATORS as KNOWLEDGE_OPERATORS
from .llm import UNUSABLE_REPLY as UNUSABLE_REPLY
from .llm import USER_TRACK_OPERATORS as USER_TRACK_OPERATORS
from .llm import agent_track_llm as agent_track_llm
from .llm import agent_track_script as agent_track_script
from .llm import boundary_reply as boundary_reply
from .llm import boundary_window as boundary_window
from .llm import knowledge_llm as knowledge_llm
from .llm import knowledge_script as knowledge_script
from .llm import memorize_llm as memorize_llm
from .llm import user_track_llm as user_track_llm
from .llm import user_track_script as user_track_script
from .storage import Storage as Storage
from .storage import StorageView as StorageView
from .storage import handler_deps as handler_deps
from .storage import index_repositories as index_repositories
from .storage import make_storage as make_storage
from .storage import open_storage_view as open_storage_view
from .stores import InMemoryAgentCaseStore as InMemoryAgentCaseStore
from .stores import InMemoryAgentSkillStore as InMemoryAgentSkillStore
from .stores import InMemoryAtomicFactStore as InMemoryAtomicFactStore
from .stores import InMemoryClusterStore as InMemoryClusterStore
from .stores import InMemoryEpisodeStore as InMemoryEpisodeStore
from .stores import InMemoryForesightStore as InMemoryForesightStore
from .stores import InMemoryMemcellStore as InMemoryMemcellStore
from .stores import InMemoryProfileStore as InMemoryProfileStore
from .stores import InMemoryReflectionReportStore as InMemoryReflectionReportStore
from .stores import UnusedPort as UnusedPort
from .trajectories import TRAJECTORY_START_MS as TRAJECTORY_START_MS
from .trajectories import agent_trajectory as agent_trajectory
from .trajectories import agent_trajectory_messages as agent_trajectory_messages

__all__ = [
    "AGENT_TRACK_OPERATORS",
    "KNOWLEDGE_OPERATORS",
    "TRAJECTORY_START_MS",
    "UNUSABLE_REPLY",
    "USER_TRACK_OPERATORS",
    "FakeEmbeddingProvider",
    "InMemoryAgentCaseIndex",
    "InMemoryAgentCaseStore",
    "InMemoryAgentSkillIndex",
    "InMemoryAgentSkillStore",
    "InMemoryAtomicFactIndex",
    "InMemoryAtomicFactStore",
    "InMemoryClusterStore",
    "InMemoryEpisodeIndex",
    "InMemoryEpisodeStore",
    "InMemoryForesightStore",
    "InMemoryMemcellStore",
    "InMemoryProfileIndex",
    "InMemoryProfileStore",
    "InMemoryReflectionReportStore",
    "Storage",
    "StorageView",
    "StubMemoryService",
    "StubOpsService",
    "UnusedPort",
    "agent_track_llm",
    "agent_track_script",
    "agent_trajectory",
    "agent_trajectory_messages",
    "boundary_reply",
    "boundary_window",
    "bundled_instructions",
    "handler_deps",
    "index_repositories",
    "knowledge_llm",
    "knowledge_script",
    "make_application",
    "make_backfill_context",
    "make_deps",
    "make_indexes",
    "make_knowledge_service",
    "make_memory_service",
    "make_ops",
    "make_storage",
    "make_stores",
    "memorize_llm",
    "open_storage_view",
    "unscripted_llm",
    "user_track_llm",
    "user_track_script",
]
