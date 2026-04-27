"""
Group raw data repository

Provides CRUD operations for Group documents (v1_groups collection).
"""

from typing import Optional, Dict, Any

from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.group import Group
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("group_raw_repository", primary=True)
class GroupRawRepository(BaseRepository[Group]):
    """
    Group raw data repository

    Provides CRUD operations and query capabilities for Group data.
    """

    def __init__(self):
        super().__init__(Group)

    async def get_by_group_id(self, group_id: str) -> Optional[Group]:
        """Get group by group_id"""
        try:
            result = await self.model.find_one({"group_id": group_id})
            if result:
                logger.debug("Retrieved group: group_id=%s", group_id)
            else:
                logger.debug("Group not found: group_id=%s", group_id)
            return result
        except Exception as e:
            logger.error("Failed to get group by group_id: %s", e)
            return None

    async def upsert_by_group_id(
        self, group_id: str, update_data: Dict[str, Any]
    ) -> Optional[Group]:
        """Update or insert group by group_id

        Uses find-then-save pattern with duplicate key retry
        to handle concurrency safely.

        Args:
            group_id: Group identifier
            update_data: Fields to update (only non-None values)

        Returns:
            The updated or created Group document
        """
        try:
            existing_doc = await self.model.find_one({"group_id": group_id})

            if existing_doc:
                for key, value in update_data.items():
                    setattr(existing_doc, key, value)
                await existing_doc.save()
                logger.debug("Updated existing group: group_id=%s", group_id)
                return existing_doc

            # Not found, create new
            try:
                new_doc = Group(group_id=group_id, **update_data)
                await new_doc.create()
                logger.info("Created new group: group_id=%s", group_id)
                return new_doc

            except Exception as create_error:
                # Handle concurrent duplicate key
                error_str = str(create_error)
                if "E11000" in error_str and "duplicate key" in error_str:
                    logger.warning(
                        "Concurrent creation conflict, retrying: group_id=%s", group_id
                    )
                    retry_doc = await self.model.find_one({"group_id": group_id})
                    if retry_doc:
                        for key, value in update_data.items():
                            setattr(retry_doc, key, value)
                        await retry_doc.save()
                        logger.debug(
                            "Updated after concurrency conflict: group_id=%s", group_id
                        )
                        return retry_doc
                    else:
                        logger.error(
                            "Record not found after concurrency conflict: group_id=%s",
                            group_id,
                        )
                        return None
                else:
                    raise create_error

        except Exception as e:
            logger.error("Failed to upsert group: %s", e)
            return None
