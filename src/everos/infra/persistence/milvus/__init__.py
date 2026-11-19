"""Milvus derived index backend.

This package mirrors the LanceDB business index surface but stores rows in
Milvus collections. It is selected through ``Settings.index.backend``; the
application builds :class:`MilvusIndexBackend` over a :class:`MilvusConnection`
only on that branch, so ``pymilvus`` stays an optional extra.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, cast

from everos.core.observability.logging import get_logger
from everos.infra.persistence.index.repositories import IndexRepositories
from everos.infra.persistence.records import (
    EMBEDDING_MODEL_KEY as EMBEDDING_MODEL_KEY,
)

from .milvus_manager import MilvusConfigurationError as MilvusConfigurationError
from .milvus_manager import MilvusConnection as MilvusConnection
from .milvus_manager import MilvusSchemaMismatchError as MilvusSchemaMismatchError
from .repos import (
    _AgentCaseRepo,
    _AgentSkillRepo,
    _AtomicFactRepo,
    _EpisodeRepo,
    _ForesightRepo,
    _KnowledgeTopicRepo,
    _UserProfileRepo,
)
from .repository import MilvusRepoBase as MilvusRepoBase
from .repository import MilvusValueLimitError as MilvusValueLimitError

logger = get_logger(__name__)


class MilvusIndexBackend:
    """Own remote Milvus collection and repository lifecycle over one connection.

    The ``IndexBackend`` port the application selects when ``index.backend``
    is ``"milvus"``.

    Args:
        connection: The application's Milvus connection.
        dim: ``[embedding] dim`` -- the width of every dense vector field.
    """

    name: ClassVar[str] = "milvus"

    def __init__(self, connection: MilvusConnection, *, dim: int) -> None:
        self._connection = connection
        self._episode = _EpisodeRepo(connection, dim=dim)
        self.repositories = IndexRepositories(
            episode=self._episode,
            atomic_fact=_AtomicFactRepo(connection, dim=dim),
            foresight=_ForesightRepo(connection, dim=dim),
            agent_case=_AgentCaseRepo(connection, dim=dim),
            agent_skill=_AgentSkillRepo(connection, dim=dim),
            user_profile=_UserProfileRepo(connection, dim=dim),
            knowledge_topic=_KnowledgeTopicRepo(connection, dim=dim),
        )

    @property
    def _collections(self) -> tuple[MilvusRepoBase[Any], ...]:
        return cast(tuple[MilvusRepoBase[Any], ...], self.repositories.all())

    async def connect(self) -> object:
        return await self._connection.client()

    async def shutdown(self) -> None:
        await self._connection.close()
        MilvusRepoBase._reset_collection_cache()

    async def ensure_business_indexes(self) -> None:
        """Create or verify every EverOS Milvus collection."""
        for repo in self._collections:
            await repo.ensure_collection()

    async def verify_business_schemas(self) -> None:
        # Same call as ensure_business_indexes: ensure_collection creates a
        # missing collection and drift-checks an existing one. Kept distinct
        # because the port and the CLI both address verification by name.
        await self.ensure_business_indexes()

    async def drop_business_tables(self) -> list[str]:
        """Drop every configured Milvus collection and return their names."""
        client = await self._connection.client()
        dropped: list[str] = []
        for repo in self._collections:
            name = repo.collection_name
            if await asyncio.to_thread(client.has_collection, name):
                try:
                    await asyncio.to_thread(client.drop_collection, name)
                except Exception:
                    # Zilliz Serverless can complete the drop server-side while
                    # its gateway returns DEADLINE_EXCEEDED. Confirm state before
                    # turning an already-successful rebuild/cleanup into failure.
                    if await asyncio.to_thread(client.has_collection, name):
                        raise
                    logger.warning(
                        "milvus_collection_drop_confirmed_after_client_error",
                        collection=name,
                    )
                dropped.append(name)
        MilvusRepoBase._reset_collection_cache()
        return dropped

    async def read_embedding_model(self) -> str | None:
        """Model this index's vectors were built with, or ``None`` if unattested.

        Lives on the episode collection as a property, mirroring what the
        LanceDB backend hangs on its vector column's Arrow metadata: the
        attestation belongs with the thing it describes, so a remote cluster
        shared by several machines cannot disagree with itself about which
        model built it.
        """
        client = await self._connection.client()
        described = await asyncio.to_thread(
            client.describe_collection, self._episode.collection_name
        )
        return (described.get("properties") or {}).get(EMBEDDING_MODEL_KEY)

    async def write_embedding_model(self, model: str) -> None:
        """Record ``model`` as a property of the episode collection."""
        client = await self._connection.client()
        await asyncio.to_thread(
            client.alter_collection_properties,
            self._episode.collection_name,
            {EMBEDDING_MODEL_KEY: model},
        )
        logger.info("milvus_embedding_model_recorded", model=model)


__all__ = [
    "MilvusConfigurationError",
    "MilvusConnection",
    "MilvusIndexBackend",
    "MilvusRepoBase",
    "MilvusSchemaMismatchError",
    "MilvusValueLimitError",
]
