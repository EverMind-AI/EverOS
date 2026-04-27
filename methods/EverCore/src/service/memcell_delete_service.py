"""
MemCell Delete Service - Handle soft delete logic for MemCell

Provides two independent delete methods:
- delete_by_id: single MemCell + MongoDB cascade by parent_id
- delete_by_filters: filter across MongoDB, Milvus, and Elasticsearch
"""

import asyncio
from typing import Any, Optional, TypedDict

from core.di.decorators import component
from core.observation.logger import get_logger
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.atomic_fact_record_raw_repository import (
    AtomicFactRecordRawRepository,
)
from infra_layer.adapters.out.persistence.repository.foresight_record_raw_repository import (
    ForesightRecordRawRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.atomic_fact_milvus_repository import (
    AtomicFactMilvusRepository,
)
from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
    ForesightMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from infra_layer.adapters.out.search.repository.atomic_fact_es_repository import (
    AtomicFactEsRepository,
)
from infra_layer.adapters.out.search.repository.foresight_es_repository import (
    ForesightEsRepository,
)
from infra_layer.adapters.out.persistence.repository.raw_message_repository import (
    RawMessageRepository,
)

logger = get_logger(__name__)


class DeleteResult(TypedDict):
    """Internal delete result for logging/monitoring. Not exposed via API."""

    deleted_memcell_count: int
    deleted_episodes: int
    deleted_atomic_facts: int
    deleted_foresights: int


@component("memcell_delete_service")
class MemCellDeleteService:
    """MemCell soft delete service"""

    def __init__(
        self,
        memcell_repository: MemCellRawRepository,
        episodic_memory_repository: EpisodicMemoryRawRepository,
        atomic_fact_repository: AtomicFactRecordRawRepository,
        foresight_repository: ForesightRecordRawRepository,
        episodic_memory_milvus_repository: EpisodicMemoryMilvusRepository,
        atomic_fact_milvus_repository: AtomicFactMilvusRepository,
        foresight_milvus_repository: ForesightMilvusRepository,
        episodic_memory_es_repository: EpisodicMemoryEsRepository,
        atomic_fact_es_repository: AtomicFactEsRepository,
        foresight_es_repository: ForesightEsRepository,
        raw_message_repository: RawMessageRepository,
    ):
        """
        Initialize deletion service

        Args:
            memcell_repository: MemCell data repository
            episodic_memory_repository: EpisodicMemory data repository
            atomic_fact_repository: AtomicFactRecord data repository
            foresight_repository: ForesightRecord data repository
            episodic_memory_milvus_repository: EpisodicMemory Milvus repository
            atomic_fact_milvus_repository: AtomicFact Milvus repository
            foresight_milvus_repository: Foresight Milvus repository
            episodic_memory_es_repository: EpisodicMemory ES repository
            atomic_fact_es_repository: AtomicFact ES repository
            foresight_es_repository: Foresight ES repository
            raw_message_repository: RawMessage repository
        """
        self.memcell_repository = memcell_repository
        self.episodic_memory_repository = episodic_memory_repository
        self.atomic_fact_repository = atomic_fact_repository
        self.foresight_repository = foresight_repository
        self.episodic_memory_milvus_repository = episodic_memory_milvus_repository
        self.atomic_fact_milvus_repository = atomic_fact_milvus_repository
        self.foresight_milvus_repository = foresight_milvus_repository
        self.episodic_memory_es_repository = episodic_memory_es_repository
        self.atomic_fact_es_repository = atomic_fact_es_repository
        self.foresight_es_repository = foresight_es_repository
        self.raw_message_repository = raw_message_repository
        logger.info("MemCellDeleteService initialized")

    # ------------------------------------------------------------------
    # Public API — two mutually exclusive delete modes
    # ------------------------------------------------------------------

    async def delete_by_id(self, memory_id: str) -> DeleteResult:
        """
        Delete a single MemCell by ID and cascade related records.

        Cascade uses parent_id only (MongoDB).
        Milvus/ES are skipped because they cannot filter by parent_id.
        """
        logger.info("Deleting by memory_id=%s", memory_id)

        try:
            deleted_count = await self.memcell_repository.delete_by_filters(
                memcell_id=memory_id
            )
            counts = await self._cascade_delete_by_parent_id(memory_id)

            result = DeleteResult(
                deleted_memcell_count=deleted_count,
                deleted_episodes=counts.get("episodes", 0),
                deleted_atomic_facts=counts.get("atomic_facts", 0),
                deleted_foresights=counts.get("foresights", 0),
            )
            logger.info("Delete by ID completed: %s", result)
            return result

        except Exception as e:
            logger.error(
                "Failed to delete by memory_id=%s: error=%s",
                memory_id,
                e,
                exc_info=True,
            )
            raise

    async def delete_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        session_id: Optional[str] = MAGIC_ALL,
        sender_id: Optional[str] = MAGIC_ALL,
    ) -> DeleteResult:
        """
        Batch delete memory records by filter conditions.

        Deletes across MongoDB, Milvus, and Elasticsearch.
        MemCell itself is not deleted (only child records).
        """
        logger.info(
            "Deleting by filters: user_id=%s, group_id=%s, "
            "session_id=%s, sender_id=%s",
            user_id,
            group_id,
            session_id,
            sender_id,
        )

        try:
            deleted = await self._batch_delete_records(
                user_id=user_id,
                group_id=group_id,
                session_id=session_id,
                sender_id=sender_id,
            )

            result = DeleteResult(
                deleted_memcell_count=0,
                deleted_episodes=deleted.get("episodes", 0),
                deleted_atomic_facts=deleted.get("atomic_facts", 0),
                deleted_foresights=deleted.get("foresights", 0),
            )

            logger.info("Delete by filters completed: %s", result)
            return result

        except Exception as e:
            logger.error("Failed to delete by filters: error=%s", e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _gather_deletes(self, *tasks: tuple[str, Any, dict]) -> dict[str, int]:
        """Run delete tasks in parallel, aggregate counts by category name.
        Args:
            tasks: (category_name, repository, kwargs) tuples.
                   Duplicate category names are summed automatically.
        """
        names = [t[0] for t in tasks]
        coros = [t[1].delete_by_filters(**t[2]) for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)
        from common_utils.async_utils import reraise_critical_errors

        reraise_critical_errors(results)
        counts: dict[str, int] = {}
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.error("Failed to cascade delete %s: %s", name, result)
            else:
                counts[name] = counts.get(name, 0) + result
                logger.debug("Cascade deleted %s: count=%d", name, result)
        return counts

    async def _cascade_delete_by_parent_id(self, parent_id: str) -> dict[str, int]:
        """Cascade by parent_id (MongoDB only, Milvus/ES skipped)."""
        kwargs = {"parent_id": parent_id}
        counts = await self._gather_deletes(
            ("episodes", self.episodic_memory_repository, kwargs),
            ("atomic_facts", self.atomic_fact_repository, kwargs),
            ("foresights", self.foresight_repository, kwargs),
        )
        return {"episodes": 0, "atomic_facts": 0, "foresights": 0, **counts}

    async def _batch_delete_records(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        session_id: Optional[str] = MAGIC_ALL,
        sender_id: Optional[str] = MAGIC_ALL,
    ) -> dict[str, int]:
        """Batch delete memory records across MongoDB, Milvus, and Elasticsearch."""
        if user_id == MAGIC_ALL and group_id == MAGIC_ALL:
            return {"episodes": 0, "atomic_facts": 0, "foresights": 0}

        mongo_kwargs = dict(
            user_id=user_id,
            group_id=group_id,
            session_id=session_id,
            sender_id=sender_id,
        )
        scope_kwargs = dict(user_id=user_id, group_id=group_id)

        # RawMessage and MemCell are source data — not deleted by filters.
        # Milvus/ES only support user_id/group_id (no session_id/sender_id).
        counts = await self._gather_deletes(
            # MongoDB (session_id/sender_id narrow scoping)
            ("episodes", self.episodic_memory_repository, mongo_kwargs),
            ("atomic_facts", self.atomic_fact_repository, mongo_kwargs),
            ("foresights", self.foresight_repository, mongo_kwargs),
            # Milvus + ES (user_id/group_id scope only)
            ("episodes", self.episodic_memory_milvus_repository, scope_kwargs),
            ("atomic_facts", self.atomic_fact_milvus_repository, scope_kwargs),
            ("foresights", self.foresight_milvus_repository, scope_kwargs),
            ("episodes", self.episodic_memory_es_repository, scope_kwargs),
            ("atomic_facts", self.atomic_fact_es_repository, scope_kwargs),
            ("foresights", self.foresight_es_repository, scope_kwargs),
        )
        return {"episodes": 0, "atomic_facts": 0, "foresights": 0, **counts}
