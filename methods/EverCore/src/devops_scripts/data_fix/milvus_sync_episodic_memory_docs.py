"""
Sync episodic memory documents to Milvus

Bulk retrieve episodic memory documents from MongoDB, convert them, and insert into Milvus.
Focuses on efficiency using a strategy of bulk retrieval, bulk conversion, and bulk insertion.

Technical implementation:
- Bulk read documents from MongoDB (controlled by batch_size)
- Use EpisodicMemoryMilvusConverter for format conversion
- Bulk insert into Milvus Collection
- Supports incremental sync (based on days parameter)
- Supports idempotent operations (using upsert semantics)
"""

import traceback
from datetime import timedelta
from typing import Optional, List, Dict, Any

from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type


logger = get_logger(__name__)


async def sync_episodic_memory_docs(
    batch_size: int, limit: Optional[int], days: Optional[int]
) -> None:
    """
    Sync episodic memory documents to Milvus.

    Implementation strategy:
    1. Bulk retrieve documents from MongoDB (batch_size per batch)
    2. Bulk convert to Milvus entity format
    3. Bulk insert into Milvus (using upsert semantics, supports idempotency)
    4. Loop until all documents are processed

    Args:
        batch_size: Batch size, recommended 500-1000
        limit: Maximum number of documents to process, None means process all
        days: Only process documents created in the last N days, None means process all
    """
    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
        EpisodicMemoryRawRepository,
    )
    from infra_layer.adapters.out.search.milvus.converter.episodic_memory_milvus_converter import (
        EpisodicMemoryMilvusConverter,
    )
    from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
        EpisodicMemoryCollection,
    )
    from common_utils.datetime_utils import get_now_with_timezone

    # Get MongoDB Repository
    mongo_repo = get_bean_by_type(EpisodicMemoryRawRepository)

    # Build query filter
    query_filter: Dict[str, Any] = {}
    if days is not None:
        now = get_now_with_timezone()
        start_time = now - timedelta(days=days)
        query_filter["created_at"] = {"$gte": start_time}
        logger.info(
            "Only processing documents created in the past %s days (starting from %s)",
            days,
            start_time,
        )

    logger.info("Starting to sync episodic memory documents to Milvus...")

    # Statistics counters
    total_processed = 0
    success_count = 0
    error_count = 0

    # Get Milvus Collection
    try:
        # Directly use the async_collection() method of EpisodicMemoryCollection
        collection = EpisodicMemoryCollection.async_collection()
        collection_name = collection.collection.name
        logger.info("Using Milvus Collection: %s", collection_name)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to get Milvus Collection: %s", e)
        raise

    # Main loop for batch processing
    try:
        skip = 0
        while True:
            # Bulk retrieve documents from MongoDB
            query = mongo_repo.model.find(query_filter).sort("created_at")
            mongo_docs = await query.skip(skip).limit(batch_size).to_list()

            if not mongo_docs:
                logger.info("No more documents to process")
                break

            # Record time range of current batch
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
                "Preparing to write batch %s - %s, time range: %s ~ %s",
                skip + 1,
                skip + len(mongo_docs),
                first_doc_time,
                last_doc_time,
            )

            # Bulk convert to Milvus entities
            milvus_entities: List[Dict[str, Any]] = []
            batch_errors = 0

            for mongo_doc in mongo_docs:
                try:
                    # Convert individual document
                    milvus_entity = EpisodicMemoryMilvusConverter.from_mongo(mongo_doc)

                    # Validate required fields
                    if not milvus_entity.get("id"):
                        logger.warning(
                            "Document missing id field, skipping: %s", mongo_doc.id
                        )
                        batch_errors += 1
                        continue

                    if not milvus_entity.get("vector"):
                        logger.warning(
                            "Document missing vector field, skipping: id=%s",
                            milvus_entity.get("id"),
                        )
                        batch_errors += 1
                        continue

                    milvus_entities.append(milvus_entity)

                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "Failed to convert document: id=%s, error=%s",
                        getattr(mongo_doc, 'id', 'unknown'),
                        e,
                    )
                    batch_errors += 1
                    continue

            # Bulk insert into Milvus
            if milvus_entities:
                try:
                    # Milvus insert method accepts list format data
                    # Need to convert list of entity dictionaries to list of lists format
                    insert_data = milvus_entities

                    _ = await collection.insert(insert_data)

                    # Count successful insertions
                    inserted_count = len(milvus_entities)
                    success_count += inserted_count
                    logger.info("Bulk insert successful: %d records", inserted_count)

                except Exception as e:  # noqa: BLE001
                    logger.error("Bulk insert to Milvus failed: %s", e)
                    traceback.print_exc()
                    error_count += len(milvus_entities)

            # Update statistics
            total_processed += len(mongo_docs)
            error_count += batch_errors

            # Check if limit reached
            if limit and total_processed >= limit:
                logger.info("Processing limit %s reached, stopping", limit)
                break

            # Move to next batch
            skip += batch_size
            if len(mongo_docs) < batch_size:
                logger.info("All documents have been processed")
                break

        # Flush Collection to ensure data persistence
        try:
            await collection.flush()
            logger.info("Milvus Collection flush completed")
        except Exception as e:  # noqa: BLE001
            logger.warning("Milvus Collection flush failed: %s", e)

        # Output statistics
        logger.info(
            "Sync completed! Total processed: %s, Success: %s, Failed: %s",
            total_processed,
            success_count,
            error_count,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Error occurred during sync: %s", exc)
        traceback.print_exc()
        raise
