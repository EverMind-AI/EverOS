"""
MemCell Restore Service - Handle restore logic for soft-deleted MemCells

Provides multiple restoration methods:
- Restore by single event_id
- Batch restore by user_id
- Batch restore by combined criteria
"""

from typing import Optional
from core.di.decorators import component
from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)

logger = get_logger(__name__)


@component("memcell_restore_service")
class MemCellRestoreService:
    """MemCell restore service for soft-deleted records"""

    def __init__(self, memcell_repository: MemCellRawRepository):
        self.memcell_repository = memcell_repository
        logger.info("MemCellRestoreService initialized")

    async def restore_by_combined_criteria(
        self,
        event_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        """
        Restore soft-deleted MemCells based on combined criteria

        Args:
            event_id: The event_id of MemCell
            user_id: User ID (batch restore all deleted memories of a user)

        Returns:
            dict: Dictionary containing restoration results
                - filters: List of filter conditions used
                - count: Number of restored records
                - success: Whether the operation succeeded
        """
        from core.oxm.constants import MAGIC_ALL

        filters_used = []

        # Restore by event_id
        if event_id and event_id != MAGIC_ALL:
            filters_used.append("event_id")
            try:
                success = await self.memcell_repository.restore_by_event_id(event_id)
                return {
                    "filters": filters_used,
                    "count": 1 if success else 0,
                    "success": success,
                }
            except Exception as e:
                logger.error("Failed to restore by event_id: %s", e)
                return {
                    "filters": filters_used,
                    "count": 0,
                    "success": False,
                    "error": str(e),
                }

        # Restore by user_id
        if user_id and user_id != MAGIC_ALL:
            filters_used.append("user_id")
            try:
                count = await self.memcell_repository.restore_by_user_id(user_id)
                return {
                    "filters": filters_used,
                    "count": count,
                    "success": count > 0,
                }
            except Exception as e:
                logger.error("Failed to restore by user_id: %s", e)
                return {
                    "filters": filters_used,
                    "count": 0,
                    "success": False,
                    "error": str(e),
                }

        # No filter conditions provided
        logger.warning("No restore criteria provided")
        return {
            "filters": [],
            "count": 0,
            "success": False,
            "error": "At least one of event_id or user_id must be provided",
        }
