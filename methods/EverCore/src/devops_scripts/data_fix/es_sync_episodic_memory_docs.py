import traceback
from datetime import timedelta
from typing import Optional, AsyncIterator, Dict, Any

from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type
from elasticsearch.helpers import async_streaming_bulk


logger = get_logger(__name__)


async def sync_episodic_memory_docs(
    batch_size: int, limit: Optional[int], days: Optional[int]
) -> None:
    """
    Sync episodic memory documents to Elasticsearch.

    Args:
        batch_size: Batch size
        limit: Maximum number of documents to process
        days: Only process documents created in the last N days; None means process all
    """
    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
        EpisodicMemoryRawRepository,
    )
    from infra_layer.adapters.out.search.elasticsearch.converter.episodic_memory_converter import (
        EpisodicMemoryConverter,
    )
    from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
        EpisodicMemoryDoc,
    )

    from common_utils.datetime_utils import get_now_with_timezone

    mongo_repo = get_bean_by_type(EpisodicMemoryRawRepository)
    index_name = EpisodicMemoryDoc.get_index_name()

    query_filter = {}
    if days is not None:
        now = get_now_with_timezone()
        start_time = now - timedelta(days=days)
        query_filter["created_at"] = {"$gte": start_time}
        logger.info(
            "Only processing documents created in the past %s days (starting from %s)",
            days,
            start_time,
        )

    logger.info("Starting to sync episodic memory documents to ES...")

    total_processed = 0
    success_count = 0
    error_count = 0

    # Get ES async client and index name
    try:
        async_client = EpisodicMemoryDoc.get_connection()
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to get Elasticsearch client: %s", e)
        raise

    async def generate_actions() -> AsyncIterator[Dict[str, Any]]:
        nonlocal total_processed
        skip = 0
        while True:
            # Use repository method to query with pagination
            mongo_docs = await mongo_repo.find_by_filter_paginated(
                query_filter=query_filter,
                skip=skip,
                limit=batch_size,
                sort_field="created_at",
                sort_desc=False,
            )

            if not mongo_docs:
                logger.info("No more documents to process")
                break

            first_doc_time = (
                mongo_docs[0].created_at
                if hasattr(mongo_docs[0], "created_at")
                else "unknown"
            )
            last_doc_time = (
                mongo_docs[-1].created_at
                if hasattr(mongo_docs[-1], "created_at")
                else "unknown"
            )
            logger.info(
                "Preparing to bulk write documents %s - %s, time range: %s ~ %s",
                skip + 1,
                skip + len(mongo_docs),
                first_doc_time,
                last_doc_time,
            )

            for mongo_doc in mongo_docs:
                es_doc = EpisodicMemoryConverter.from_mongo(mongo_doc)
                src = es_doc.to_dict()
                doc_id = es_doc.meta.id

                yield {
                    "retry_on_conflict": 3,
                    "_op_type": "update",
                    "_index": index_name,
                    "doc_as_upsert": True,
                    "_id": doc_id,
                    "doc": src,
                }

                total_processed += 1
                if limit and total_processed >= limit:
                    logger.info(
                        "Reached processing limit %s, stop generating actions", limit
                    )
                    return

            skip += batch_size
            if len(mongo_docs) < batch_size:
                logger.info("All documents have been processed")
                break

    try:
        # Use streaming bulk to perform bulk upsert
        async for ok, info in async_streaming_bulk(
            async_client, generate_actions(), chunk_size=batch_size
        ):
            if ok:
                success_count += 1
            else:
                error_count += 1
                logger.error("Bulk write failed: %s", info)

        # Refresh index
        await async_client.indices.refresh(index=index_name)

        logger.info(
            "Sync completed! Total processed: %s, Success: %s, Failed: %s",
            total_processed,
            success_count,
            error_count,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("An error occurred during sync: %s", exc)
        traceback.print_exc()
        raise
