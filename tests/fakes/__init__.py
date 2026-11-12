"""In-memory test doubles for the memory domain's ports.

Named after the Haystack ``InMemoryDocumentStore`` convention: each
``InMemory*`` class implements one port with plain containers a test can
seed and inspect. :func:`make_deps` assembles them into a
:class:`~everos.memory.deps.MemoryDeps` for calling a strategy directly::

    deps = make_deps()
    events = await extract_user_profile(event, deps)
    assert await deps.stores.profile.read(scope) is not None

:func:`make_application` does the same one level up: an
:class:`~everos.service.Application` whose facades are recording stubs.
"""

from __future__ import annotations

from .application import FakeOmeLock as FakeOmeLock
from .application import StubMemoryService as StubMemoryService
from .application import StubOpsService as StubOpsService
from .application import make_application as make_application
from .application import make_backfill_deps as make_backfill_deps
from .application import make_memory_service as make_memory_service
from .application import make_ops as make_ops
from .application import make_providers as make_providers
from .application import unused_backfill_deps as unused_backfill_deps
from .backfill import InMemoryVectorTable as InMemoryVectorTable
from .backfill import backfill_deps as backfill_deps
from .backfill import episode_row as episode_row
from .deps import bundled_instructions as bundled_instructions
from .deps import make_deps as make_deps
from .deps import make_indexes as make_indexes
from .deps import make_stores as make_stores
from .deps import seed_sources as seed_sources
from .deps import unscripted_llm as unscripted_llm
from .embedding import FakeEmbeddingProvider as FakeEmbeddingProvider
from .indexes import InMemoryEpisodeIndex as InMemoryEpisodeIndex
from .indexes import InMemoryGroupProfileIndex as InMemoryGroupProfileIndex
from .indexes import InMemoryImageMemoryIndex as InMemoryImageMemoryIndex
from .indexes import InMemoryProfileIndex as InMemoryProfileIndex
from .llm import UNUSABLE_REPLY as UNUSABLE_REPLY
from .llm import USER_TRACK_OPERATORS as USER_TRACK_OPERATORS
from .llm import boundary_reply as boundary_reply
from .llm import boundary_window as boundary_window
from .llm import memorize_llm as memorize_llm
from .llm import user_track_llm as user_track_llm
from .llm import user_track_script as user_track_script
from .sources import static_trace_path as static_trace_path
from .sources import static_tuning as static_tuning
from .sources import unused_raw_file_builder as unused_raw_file_builder
from .storage import TEST_DIM as TEST_DIM
from .storage import Storage as Storage
from .storage import StorageView as StorageView
from .storage import arrow_schema as arrow_schema
from .storage import handler_deps as handler_deps
from .storage import index_repositories as index_repositories
from .storage import make_storage as make_storage
from .storage import open_storage_view as open_storage_view
from .stores import InMemoryClusterStore as InMemoryClusterStore
from .stores import InMemoryEpisodeStore as InMemoryEpisodeStore
from .stores import InMemoryForgetLogStore as InMemoryForgetLogStore
from .stores import InMemoryGroupProfileStore as InMemoryGroupProfileStore
from .stores import InMemoryGroupStore as InMemoryGroupStore
from .stores import InMemoryImageMemoryStore as InMemoryImageMemoryStore
from .stores import InMemoryImageStore as InMemoryImageStore
from .stores import InMemoryMemcellStore as InMemoryMemcellStore
from .stores import InMemoryProfileStore as InMemoryProfileStore
from .stores import InMemorySceneStore as InMemorySceneStore
from .stores import UnusedPort as UnusedPort

__all__ = [
    "TEST_DIM",
    "UNUSABLE_REPLY",
    "USER_TRACK_OPERATORS",
    "FakeEmbeddingProvider",
    "FakeOmeLock",
    "InMemoryClusterStore",
    "InMemoryEpisodeIndex",
    "InMemoryEpisodeStore",
    "InMemoryForgetLogStore",
    "InMemoryGroupProfileIndex",
    "InMemoryGroupProfileStore",
    "InMemoryGroupStore",
    "InMemoryImageMemoryIndex",
    "InMemoryImageMemoryStore",
    "InMemoryImageStore",
    "InMemoryMemcellStore",
    "InMemoryProfileIndex",
    "InMemoryProfileStore",
    "InMemorySceneStore",
    "InMemoryVectorTable",
    "Storage",
    "StorageView",
    "StubMemoryService",
    "StubOpsService",
    "UnusedPort",
    "arrow_schema",
    "backfill_deps",
    "boundary_reply",
    "boundary_window",
    "bundled_instructions",
    "episode_row",
    "handler_deps",
    "index_repositories",
    "make_application",
    "make_backfill_deps",
    "make_deps",
    "make_indexes",
    "make_memory_service",
    "make_ops",
    "make_providers",
    "make_storage",
    "make_stores",
    "memorize_llm",
    "open_storage_view",
    "seed_sources",
    "static_trace_path",
    "static_tuning",
    "unscripted_llm",
    "unused_backfill_deps",
    "unused_raw_file_builder",
    "user_track_llm",
    "user_track_script",
]
