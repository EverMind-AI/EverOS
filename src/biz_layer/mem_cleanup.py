"""
Memory cleanup utilities.

Provides scheduled cleanup tasks for expired memory records.
Currently handles foresight expiry: records whose validity window has passed
are removed from all three stores (Milvus → Elasticsearch → MongoDB) in that
order to minimise the window where a record is searchable but absent from the
primary store.
"""

from datetime import datetime
from typing import Dict

from common_utils.datetime_utils import get_now_with_timezone
from core.di import get_bean_by_type
from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.repository.foresight_record_repository import (
    ForesightRecordRawRepository,
)
from infra_layer.adapters.out.search.repository.foresight_es_repository import (
    ForesightEsRepository,
)
from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
    ForesightMilvusRepository,
)

logger = get_logger(__name__)


async def cleanup_expired_foresights(
    before: datetime | None = None,
) -> Dict[str, int]:
    """
    Delete foresight records that have passed their validity end time.

    Deletion order: Milvus → Elasticsearch → MongoDB.
    This ensures that even if a later step fails, the record is no longer
    returned by vector or keyword search.

    Args:
        before: Treat records with end_time < before as expired.
                Defaults to the current time when not provided.

    Returns:
        Dict with keys ``milvus``, ``es``, ``mongo`` and the number of
        records deleted from each store.
    """
    if before is None:
        before = get_now_with_timezone()

    stats: Dict[str, int] = {"milvus": 0, "es": 0, "mongo": 0}

    foresight_milvus_repo = get_bean_by_type(ForesightMilvusRepository)
    foresight_es_repo = get_bean_by_type(ForesightEsRepository)
    foresight_mongo_repo = get_bean_by_type(ForesightRecordRawRepository)

    # Step 1: remove from Milvus (vector search)
    try:
        stats["milvus"] = await foresight_milvus_repo.delete_by_filters(end_time=before)
    except Exception as exc:
        logger.error("Failed to delete expired foresights from Milvus: %s", exc)

    # Step 2: remove from Elasticsearch (keyword search)
    try:
        stats["es"] = await foresight_es_repo.delete_expired(before=before)
    except Exception as exc:
        logger.error("Failed to delete expired foresights from ES: %s", exc)

    # Step 3: remove from MongoDB (primary store)
    try:
        stats["mongo"] = await foresight_mongo_repo.delete_expired(before=before)
    except Exception as exc:
        logger.error("Failed to delete expired foresights from MongoDB: %s", exc)

    logger.info(
        "✅ Expired foresight cleanup complete (before=%s): milvus=%d es=%d mongo=%d",
        before.isoformat(),
        stats["milvus"],
        stats["es"],
        stats["mongo"],
    )
    return stats
