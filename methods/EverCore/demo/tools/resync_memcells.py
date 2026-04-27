"""
Batch resync existing MemCell.episode from MongoDB to Milvus / ES.

Usage:
    uv run python src/bootstrap.py demo/tools/resync_memcells.py
"""

import asyncio
from typing import List

from core.di import get_bean_by_type
from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
from biz_layer.memcell_sync import MemCellSyncService

logger = get_logger(__name__)


async def main() -> None:
    service = get_bean_by_type(MemCellSyncService)

    memcells: List[MemCell] = await MemCell.find_all().to_list()
    if not memcells:
        logger.info("No MemCell records in MongoDB, skipping")
        return

    logger.info("Starting resync of %s MemCell records", len(memcells))
    success = 0
    for memcell in memcells:
        await service.sync_memcell(memcell, sync_to_es=True, sync_to_milvus=True)
        success += 1

    logger.info("Resync completed, success: %s", success)


if __name__ == "__main__":
    asyncio.run(main())
