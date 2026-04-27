"""Foresight and atomic fact synchronization service

Responsible for writing unified foresight and atomic facts into Milvus / Elasticsearch.
"""

from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
)
from infra_layer.adapters.out.search.elasticsearch.converter.foresight_converter import (
    ForesightConverter,
)
from infra_layer.adapters.out.search.milvus.converter.foresight_milvus_converter import (
    ForesightMilvusConverter,
)
from infra_layer.adapters.out.persistence.document.memory.atomic_fact_record import (
    AtomicFactRecord,
)
from infra_layer.adapters.out.search.elasticsearch.converter.atomic_fact_converter import (
    AtomicFactConverter,
)
from infra_layer.adapters.out.search.milvus.converter.atomic_fact_milvus_converter import (
    AtomicFactMilvusConverter,
)
from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
    ForesightMilvusRepository,
)
from infra_layer.adapters.out.search.repository.atomic_fact_milvus_repository import (
    AtomicFactMilvusRepository,
)
from infra_layer.adapters.out.search.repository.foresight_es_repository import (
    ForesightEsRepository,
)
from infra_layer.adapters.out.search.repository.atomic_fact_es_repository import (
    AtomicFactEsRepository,
)
from core.di import get_bean_by_type, service
from common_utils.datetime_utils import get_now_with_timezone

logger = logging.getLogger(__name__)


@service(name="memory_sync_service", primary=True)
class MemorySyncService:
    """Foresight and atomic fact synchronization service"""

    def __init__(
        self,
        foresight_milvus_repo: Optional[ForesightMilvusRepository] = None,
        atomic_fact_milvus_repo: Optional[AtomicFactMilvusRepository] = None,
        foresight_es_repo: Optional[ForesightEsRepository] = None,
        atomic_fact_es_repo: Optional[AtomicFactEsRepository] = None,
    ):
        """Initialize synchronization service

        Args:
            foresight_milvus_repo: Foresight Milvus repository instance (optional, obtained from DI if not provided)
            atomic_fact_milvus_repo: Atomic fact Milvus repository instance (optional, obtained from DI if not provided)
            foresight_es_repo: Foresight ES repository instance (optional, obtained from DI if not provided)
            atomic_fact_es_repo: Atomic fact ES repository instance (optional, obtained from DI if not provided)
        """
        self.foresight_milvus_repo = foresight_milvus_repo or get_bean_by_type(
            ForesightMilvusRepository
        )
        self.atomic_fact_milvus_repo = atomic_fact_milvus_repo or get_bean_by_type(
            AtomicFactMilvusRepository
        )
        self.foresight_es_repo = foresight_es_repo or get_bean_by_type(
            ForesightEsRepository
        )
        self.atomic_fact_es_repo = atomic_fact_es_repo or get_bean_by_type(
            AtomicFactEsRepository
        )

        logger.info("MemorySyncService initialization completed")

    @staticmethod
    def _normalize_datetime(value: Optional[datetime | str]) -> Optional[datetime]:
        """Convert str/None to datetime (supports date-only strings)"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    logger.warning("Unable to parse date string: %s", value)
                    return None
        return None

    async def sync_foresight(
        self,
        foresight: ForesightRecord,
        sync_to_es: bool = True,
        sync_to_milvus: bool = True,
    ) -> Dict[str, int]:
        """Synchronize a single foresight to Milvus/ES

        Args:
            foresight: ForesightRecord document object
            sync_to_es: Whether to sync to ES (default True)
            sync_to_milvus: Whether to sync to Milvus (default True)

        Returns:
            Synchronization statistics {"foresight": 1}
        """
        stats = {"foresight": 0, "es_records": 0}

        try:
            # Read embedding from MongoDB, skip if not exists
            if not foresight.vector:
                logger.warning(
                    f"Foresight {foresight.id} has no embedding, skipping sync"
                )
                return stats

            # Sync to Milvus
            if sync_to_milvus:
                # Use converter to generate Milvus entity
                milvus_entity = ForesightMilvusConverter.from_mongo(foresight)
                await self.foresight_milvus_repo.insert(milvus_entity, flush=False)
                stats["foresight"] += 1
                logger.debug(f"Foresight synced to Milvus: {foresight.id}")

            # Sync to ES
            if sync_to_es:
                # Use converter to generate correct ES document (including jieba tokenized search_content)
                es_doc = ForesightConverter.from_mongo(foresight)
                await self.foresight_es_repo.create(es_doc)
                stats["es_records"] += 1
                logger.debug(f"Foresight synced to ES: {foresight.id}")

        except Exception as e:
            logger.error(f"Failed to sync foresight: {e}", exc_info=True)
            raise

        return stats

    async def sync_atomic_fact(
        self,
        atomic_fact_record: AtomicFactRecord,
        sync_to_es: bool = True,
        sync_to_milvus: bool = True,
    ) -> Dict[str, int]:
        """Synchronize a single atomic fact to Milvus/ES

        Args:
            atomic_fact_record: AtomicFactRecord document object
            sync_to_es: Whether to sync to ES (default True)
            sync_to_milvus: Whether to sync to Milvus (default True)

        Returns:
            Synchronization statistics {"atomic_fact": 1}
        """
        stats = {"atomic_fact": 0, "es_records": 0}

        try:
            # Read existing vector from MongoDB
            if not atomic_fact_record.vector:
                logger.warning(
                    f"Atomic fact {atomic_fact_record.id} has no embedding, skipping sync"
                )
                return stats

            # Sync to Milvus
            if sync_to_milvus:
                # Use converter to generate Milvus entity
                milvus_entity = AtomicFactMilvusConverter.from_mongo(atomic_fact_record)
                await self.atomic_fact_milvus_repo.insert(milvus_entity, flush=False)
                stats["atomic_fact"] += 1
                logger.debug(f"Atomic fact synced to Milvus: {atomic_fact_record.id}")

            # Sync to ES
            if sync_to_es:
                # Use converter to generate correct ES document (including jieba tokenized search_content)
                es_doc = AtomicFactConverter.from_mongo(atomic_fact_record)
                await self.atomic_fact_es_repo.create(es_doc)
                stats["es_records"] += 1
                logger.debug(f"Atomic fact synced to ES: {atomic_fact_record.id}")

        except Exception as e:
            logger.error(f"Failed to sync atomic fact: {e}", exc_info=True)
            raise

        return stats

    async def sync_batch_foresights(
        self,
        foresights: List[ForesightRecord],
        sync_to_es: bool = True,
        sync_to_milvus: bool = True,
    ) -> Dict[str, int]:
        """Batch synchronize foresights

        Args:
            foresights: List of ForesightRecord
            sync_to_es: Whether to sync to ES (default True)
            sync_to_milvus: Whether to sync to Milvus (default True)

        Returns:
            Synchronization statistics
        """
        total_stats = {"foresight": 0, "es_records": 0}

        for foresight_mem in foresights:
            try:
                stats = await self.sync_foresight(
                    foresight_mem, sync_to_es=sync_to_es, sync_to_milvus=sync_to_milvus
                )
                total_stats["foresight"] += stats.get("foresight", 0)
                total_stats["es_records"] += stats.get("es_records", 0)
            except Exception as e:
                logger.error(
                    f"Failed to batch sync foresight: {foresight_mem.id}, error: {e}",
                    exc_info=True,
                )
                # Do not silently swallow exceptions

        logger.info(
            f"✅ Foresight Milvus flush completed: {total_stats['foresight']} records"
        )

        return total_stats

    async def sync_batch_atomic_facts(
        self,
        atomic_facts: List[AtomicFactRecord],
        sync_to_es: bool = True,
        sync_to_milvus: bool = True,
    ) -> Dict[str, int]:
        """Batch synchronize atomic facts

        Args:
            atomic_facts: List of AtomicFactRecord
            sync_to_es: Whether to sync to ES (default True)
            sync_to_milvus: Whether to sync to Milvus (default True)

        Returns:
            Synchronization statistics
        """
        total_stats = {"atomic_fact": 0, "es_records": 0}

        for fact_record in atomic_facts:
            try:
                stats = await self.sync_atomic_fact(
                    fact_record, sync_to_es=sync_to_es, sync_to_milvus=sync_to_milvus
                )
                total_stats["atomic_fact"] += stats.get("atomic_fact", 0)
                total_stats["es_records"] += stats.get("es_records", 0)
            except Exception as e:
                logger.error(
                    f"Failed to batch sync atomic fact: {fact_record.id}, error: {e}",
                    exc_info=True,
                )
                # Do not silently swallow exceptions, let it surface
                raise

        logger.info(
            f"Atomic fact Milvus flush completed: {total_stats['atomic_fact']} records"
        )

        return total_stats
