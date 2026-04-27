"""
Batch resync foresight records to Milvus/ES.

Usage:
    uv run python src/bootstrap.py demo/tools/resync_personal_memories.py
"""

import asyncio
from typing import List

from core.di import get_bean_by_type
from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
)
from biz_layer.personal_memory_sync import MemorySyncService

logger = get_logger(__name__)


async def main():
    service = get_bean_by_type(MemorySyncService)

    docs: List[ForesightRecord] = await ForesightRecord.find_all().to_list()
    if not docs:
        logger.info("No foresight_records found in MongoDB, skipping")
        return

    logger.info("Starting resync of %s foresight records", len(docs))
    stats = await service.sync_batch_foresights(
        docs,
        sync_to_es=True,
        sync_to_milvus=True,
    )
    logger.info("Resync completed: %s", stats)


if __name__ == "__main__":
    asyncio.run(main())
