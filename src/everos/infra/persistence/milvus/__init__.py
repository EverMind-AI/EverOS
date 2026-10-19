"""Milvus derived index backend.

This package mirrors the LanceDB business index surface but stores rows in
Milvus collections. It is selected through ``Settings.index.backend`` and is
normally reached through :mod:`everos.infra.persistence.index`.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from everos.core.observability.logging import get_logger
from everos.infra.persistence.records import (
    EMBEDDING_MODEL_KEY as EMBEDDING_MODEL_KEY,
)

from .milvus_manager import MilvusConfigurationError as MilvusConfigurationError
from .milvus_manager import MilvusSchemaMismatchError as MilvusSchemaMismatchError
from .milvus_manager import dispose_connection as dispose_connection
from .milvus_manager import get_client as get_client
from .repos import ALL_REPOS as ALL_REPOS
from .repos import agent_case_repo as agent_case_repo
from .repos import agent_skill_repo as agent_skill_repo
from .repos import atomic_fact_repo as atomic_fact_repo
from .repos import episode_repo as episode_repo
from .repos import foresight_repo as foresight_repo
from .repos import knowledge_topic_repo as knowledge_topic_repo
from .repos import user_profile_repo as user_profile_repo
from .repository import MilvusRepoBase as MilvusRepoBase
from .repository import MilvusValueLimitError as MilvusValueLimitError

logger = get_logger(__name__)


async def ensure_business_indexes() -> None:
    """Create or verify every EverOS Milvus collection."""
    for repo in ALL_REPOS:
        await repo.ensure_collection()


async def drop_business_tables() -> list[str]:
    """Drop every configured Milvus collection and return their names."""
    client = await get_client()
    dropped: list[str] = []
    for repo in ALL_REPOS:
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


class MilvusIndexBackend:
    """Own remote Milvus connection, collection, and repository lifecycle.

    The ``IndexBackend`` port the ``everos.infra.persistence.index`` facade
    selects when ``index.backend`` is ``"milvus"``. The facade imports this
    package only on that branch, so ``pymilvus`` stays an optional extra.
    """

    name: ClassVar[str] = "milvus"

    @property
    def repositories(self) -> tuple[Any, ...]:
        return tuple(ALL_REPOS)

    async def connect(self) -> Any:
        return await get_client()

    async def shutdown(self) -> None:
        await dispose_connection()

    async def ensure_business_indexes(self) -> None:
        await ensure_business_indexes()

    async def verify_business_schemas(self) -> None:
        # Same call as ensure_business_indexes: ensure_collection creates a
        # missing collection and drift-checks an existing one. Kept distinct
        # because the port and the CLI both address verification by name.
        await self.ensure_business_indexes()

    async def drop_business_tables(self) -> list[str]:
        return await drop_business_tables()

    async def read_embedding_model(self) -> str | None:
        """Model this index's vectors were built with, or ``None`` if unattested.

        Lives on the episode collection as a property, mirroring what the
        LanceDB backend hangs on its vector column's Arrow metadata: the
        attestation belongs with the thing it describes, so a remote cluster
        shared by several machines cannot disagree with itself about which
        model built it.
        """
        client = await get_client()
        described = await asyncio.to_thread(
            client.describe_collection, episode_repo.collection_name
        )
        return (described.get("properties") or {}).get(EMBEDDING_MODEL_KEY)

    async def write_embedding_model(self, model: str) -> None:
        """Record ``model`` as a property of the episode collection."""
        client = await get_client()
        await asyncio.to_thread(
            client.alter_collection_properties,
            episode_repo.collection_name,
            {EMBEDDING_MODEL_KEY: model},
        )
        logger.info("milvus_embedding_model_recorded", model=model)


index_backend = MilvusIndexBackend()

__all__ = [
    "ALL_REPOS",
    "MilvusConfigurationError",
    "MilvusIndexBackend",
    "MilvusSchemaMismatchError",
    "MilvusValueLimitError",
    "agent_case_repo",
    "agent_skill_repo",
    "atomic_fact_repo",
    "dispose_connection",
    "drop_business_tables",
    "ensure_business_indexes",
    "episode_repo",
    "foresight_repo",
    "get_client",
    "index_backend",
    "knowledge_topic_repo",
    "user_profile_repo",
]
