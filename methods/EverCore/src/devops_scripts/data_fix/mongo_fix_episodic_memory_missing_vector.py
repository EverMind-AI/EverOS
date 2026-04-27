#!/usr/bin/env python3
"""
Fix missing vector fields in historical EpisodicMemory documents.

How to run (recommended to run via bootstrap, which automatically loads application context and dependency injection):
  python src/bootstrap.py src/scripts/data_fix/fix_episodic_memory_missing_vector.py --limit 1000 --batch 200 --concurrency 8

Arguments:
  --limit         Maximum number of documents to process (default 1000)
  --batch         Number of documents to fetch from database each time (default 200, larger is faster but uses more memory)
  --concurrency   Concurrency level (default 8)
"""

import argparse
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from agentic_layer.vectorize_service import get_vectorize_service
from common_utils.datetime_utils import from_iso_format, to_iso_format


logger = get_logger(__name__)

# Target vector model: records not using this model also need to be re-processed
TARGET_VECTOR_MODEL = "Qwen/Qwen3-Embedding-4B"


async def _fetch_candidates(
    size: int,
    created_before: Optional[Any],
    created_gte: Optional[Any],
    created_lte: Optional[Any],
) -> List[EpisodicMemory]:
    """
    Query candidate episodic memory documents missing vectors.

    Returns two types of documents:
    1) Documents where episode is not empty but vector is missing/None/empty array
    2) Documents where vector_model is not equal to the target model (TARGET_VECTOR_MODEL) (i.e., need re-processing)
    """
    and_filters: List[Dict[str, Any]] = [
        {"episode": {"$exists": True, "$ne": ""}},
        {
            "$or": [
                {"vector": {"$exists": False}},
                {"vector": None},
                {"vector": []},
                {"vector_model": {"$ne": TARGET_VECTOR_MODEL}},
                {"vector_model": {"$exists": False}},
                {"vector_model": None},
                {"vector_model": ""},
            ]
        },
    ]

    # created_at filter conditions (range + pagination anchor)
    created_at_filter: Dict[str, Any] = {}
    if created_gte is not None:
        created_at_filter["$gte"] = created_gte
    if created_lte is not None:
        created_at_filter["$lte"] = created_lte
    # Pagination anchor: prioritize recently created data, then continue with earlier data
    if created_before is not None:
        created_at_filter["$lt"] = created_before
    if created_at_filter:
        and_filters.append({"created_at": created_at_filter})

    query: Dict[str, Any] = {"$and": and_filters}

    cursor = EpisodicMemory.find(query).sort("-created_at").limit(size)  # Recent first

    results = await cursor.to_list()
    return results


async def _process_one(
    document: EpisodicMemory, semaphore: asyncio.Semaphore
) -> Tuple[Optional[str], Optional[str]]:
    """
    Process a single document: vectorize episode and write back vector and vector_model.

    Returns (doc_id, error); error is None on success.
    """
    async with semaphore:
        try:
            if not document.episode:
                return str(document.id), "episode is empty, skipping"

            vectorize_service = get_vectorize_service()
            embedding = await vectorize_service.get_embedding(document.episode)
            vector_list = embedding.tolist()  # Consistent with repository logic
            model_name = vectorize_service.get_model_name()

            # Update precisely by _id to avoid overwriting other fields
            await EpisodicMemory.find({"_id": document.id}).update(
                {"$set": {"vector": vector_list, "vector_model": model_name}}
            )

            return str(document.id), None
        except Exception as exc:  # noqa: BLE001 Non-critical error, log and continue
            return str(document.id), str(exc)


async def run_fix(
    limit: int = 1000,
    batch: int = 200,
    concurrency: int = 10,
    start_created_at: Optional[Any] = None,
    end_created_at: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute the fix task.

    Args:
        limit:    Maximum number of documents to process
        batch:    Number of documents to fetch from database in each batch
        concurrency: Concurrency level (coroutine concurrency)

    Returns:
        Statistics dictionary
    """
    if limit <= 0:
        limit = 1
    if batch <= 0:
        batch = 1
    if concurrency <= 0:
        concurrency = 1

    semaphore = asyncio.Semaphore(concurrency)

    processed_total = 0
    succeeded = 0
    errors: List[Tuple[str, str]] = []
    created_before: Optional[Any] = None
    # Range filtering passed via function parameters
    created_gte: Optional[Any] = start_created_at
    created_lte: Optional[Any] = end_created_at

    logger.info(
        "🔍 Starting scan for documents to fix (limit=%d, batch=%d, concurrency=%d)",
        limit,
        batch,
        concurrency,
    )

    while processed_total < limit:
        fetch_size = min(batch, limit - processed_total)
        candidates = await _fetch_candidates(
            size=fetch_size,
            created_before=created_before,
            created_gte=created_gte,
            created_lte=created_lte,
        )

        if not candidates:
            break

        # Next page anchor: earliest created_at in this batch
        try:
            created_before = candidates[-1].created_at
            try:
                logger.info(
                    "⏱️ Currently processing created_at=%s",
                    to_iso_format(created_before),
                )
            except Exception:  # noqa: BLE001
                logger.info("⏱️ Currently processing created_at=%s", str(created_before))
        except AttributeError:
            # If model lacks this field or exception occurs, fall back to skip logic (do not update anchor)
            pass

        logger.info(
            "📦 Fetched %d candidates (cumulative processed=%d/%d)",
            len(candidates),
            processed_total,
            limit,
        )

        tasks: List[asyncio.Task] = []
        for doc in candidates:
            task = asyncio.create_task(_process_one(doc, semaphore))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=False)

        for doc_id, err in results:
            if err is None:
                succeeded += 1
            else:
                errors.append((doc_id or "unknown", err))

        processed_total += len(candidates)

    failed = len(errors)
    if failed:
        for doc_id, err_msg in errors[:20]:  # Avoid excessive logging
            logger.error("❌ Fix failed doc=%s, error=%s", doc_id, err_msg)
        if failed > 20:
            logger.error("… %d more errors not printed individually", failed - 20)

    logger.info(
        "✅ Fix completed | total=%d, succeeded=%d, failed=%d",
        processed_total,
        succeeded,
        failed,
    )
    return {
        "total": processed_total,
        "succeeded": succeeded,
        "failed": failed,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix missing vector data in historical EpisodicMemory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of documents to process (default 1000)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=200,
        help="Number of documents to fetch from database each time (default 200)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Concurrency level (default 8)"
    )
    parser.add_argument(
        "--start-created-at",
        dest="start_created_at",
        type=str,
        default=None,
        help="Only process documents with created_at greater than or equal to this time (ISO format, e.g., 2025-09-16T20:20:06+00:00)",
    )
    parser.add_argument(
        "--end-created-at",
        dest="end_created_at",
        type=str,
        default=None,
        help="Only process documents with created_at less than or equal to this time (ISO format, e.g., 2025-09-30T23:59:59+00:00)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    # When running via bootstrap, application context is already loaded; execute async task directly here
    # Parse time range arguments (ISO -> timezone-aware datetime)
    start_dt = from_iso_format(args.start_created_at) if args.start_created_at else None
    end_dt = from_iso_format(args.end_created_at) if args.end_created_at else None

    if start_dt or end_dt:
        try:
            start_str = to_iso_format(start_dt) if start_dt else "(not specified)"
            end_str = to_iso_format(end_dt) if end_dt else "(not specified)"
        except Exception:  # noqa: BLE001
            start_str = str(start_dt) if start_dt else "(not specified)"
            end_str = str(end_dt) if end_dt else "(not specified)"
        logger.info("⛳ Using created_at filter range: [%s, %s]", start_str, end_str)

    asyncio.run(
        run_fix(
            limit=args.limit,
            batch=args.batch,
            concurrency=args.concurrency,
            start_created_at=start_dt,
            end_created_at=end_dt,
        )
    )


if __name__ == "__main__":
    main()
