"""Construction of the storage adapters behind the memory ports.

The application bootstrap builds its connections, stores and indexes here
and injects them into domain objects; nothing in the domain constructs an
adapter.
"""

from __future__ import annotations

__all__ = [
    "build_derived_index",
    "build_indexes",
    "build_stores",
]

from everos.config import Settings
from everos.memory.deps import Indexes, Stores

from .groups import GroupRosterStore
from .images import FileImageStore
from .index import DerivedIndex
from .index.adapters import (
    EpisodeIndexAdapter,
    GroupProfileIndexAdapter,
    ImageMemoryIndexAdapter,
    ProfileIndexAdapter,
)
from .lancedb import LanceDbConnection, LanceIndexBackend
from .markdown import (
    MarkdownEpisodeStore,
    MarkdownGroupProfileStore,
    MarkdownImageMemoryStore,
    MarkdownOwnerDirStore,
    MarkdownProfileStore,
    MarkdownSceneStore,
)
from .memory_root import MemoryRoot
from .sqlite import (
    SqliteClusterStore,
    SqliteForgetLogStore,
    SqliteMemcellStore,
    SqliteMemorizeInboxStore,
    SqliteMessageForgetStore,
    SqliteRepos,
    SqliteSessionBufferStore,
    SqliteSessionPurgeStore,
)


def build_stores(root: MemoryRoot, sqlite: SqliteRepos) -> Stores:
    """Markdown truth under ``root`` plus the SQLite state behind ``sqlite``."""
    return Stores(
        episode=MarkdownEpisodeStore(root),
        image_memory=MarkdownImageMemoryStore(root),
        images=FileImageStore(root, sqlite.image_original),
        forget_log=SqliteForgetLogStore(sqlite.forget_log),
        profile=MarkdownProfileStore(root),
        session_buffer=SqliteSessionBufferStore(
            sqlite.unprocessed_buffer, sqlite.session_cut
        ),
        memcell=SqliteMemcellStore(sqlite.memcell),
        cluster=SqliteClusterStore(sqlite.cluster),
        memorize_inbox=SqliteMemorizeInboxStore(sqlite.memorize_inbox),
        owner_dir=MarkdownOwnerDirStore(root),
        session_purge=SqliteSessionPurgeStore(sqlite.session_purge),
        groups=GroupRosterStore(root, sqlite.group_roster),
        scene=MarkdownSceneStore(root),
        group_profile=MarkdownGroupProfileStore(root),
        message_forget=SqliteMessageForgetStore(sqlite.message_forget),
    )


def build_derived_index(settings: Settings, root: MemoryRoot) -> DerivedIndex:
    """The derived index of the engine ``settings.index.backend`` selects.

    Builds the engine's connection and backend; nothing connects until the
    index is first used. This is the one place the vector widths reach the
    derived index: ``[embedding] dim`` builds every text vector column, and
    ``[vision_embedding] dim`` builds ``image_memory.visual_vector`` -- the
    picture's own embedding, at the vision model's native width.
    """
    column_dims = (("visual_vector", settings.vision_embedding.dim),)
    if settings.index.backend == "milvus":
        # Deferred: pymilvus is an optional extra, needed only when selected.
        from .milvus import MilvusConnection, MilvusIndexBackend

        backend = MilvusIndexBackend(
            MilvusConnection(settings.milvus),
            dim=settings.embedding.dim,
            column_dims=column_dims,
        )
    else:
        backend = LanceIndexBackend(
            LanceDbConnection(root, settings.lancedb),
            dim=settings.embedding.dim,
            column_dims=column_dims,
        )
    return DerivedIndex(
        backend, embedding_dim=settings.embedding.dim, column_dims=column_dims
    )


def build_indexes(index: DerivedIndex) -> Indexes:
    """The domain's index readers over ``index``'s repositories."""
    repos = index.repositories
    return Indexes(
        episode=EpisodeIndexAdapter(repos.episode),
        image_memory=ImageMemoryIndexAdapter(repos.image_memory),
        profile=ProfileIndexAdapter(repos.user_profile),
        group_profile=GroupProfileIndexAdapter(repos.group_profile_item),
    )
