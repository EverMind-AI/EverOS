"""
Sender raw data repository

Provides CRUD operations for Sender documents (v1_senders collection).
"""

from typing import Optional, Dict, Any, List

from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.sender import Sender
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("sender_raw_repository", primary=True)
class SenderRawRepository(BaseRepository[Sender]):
    """
    Sender raw data repository

    Provides CRUD operations and query capabilities for Sender data.
    """

    def __init__(self):
        super().__init__(Sender)

    async def get_by_sender_id(self, sender_id: str) -> Optional[Sender]:
        """Get sender by sender_id"""
        try:
            result = await self.model.find_one({"sender_id": sender_id})
            if result:
                logger.debug("Retrieved sender: sender_id=%s", sender_id)
            else:
                logger.debug("Sender not found: sender_id=%s", sender_id)
            return result
        except Exception as e:
            logger.error("Failed to get sender by sender_id: %s", e)
            return None

    async def get_by_sender_ids(self, sender_ids: List[str]) -> List[Sender]:
        """Batch get senders by sender_ids.

        Args:
            sender_ids: List of sender identifiers

        Returns:
            List of matching Sender documents
        """
        if not sender_ids:
            return []
        try:
            results = await self.model.find(
                {"sender_id": {"$in": sender_ids}}
            ).to_list()
            logger.debug(
                "Batch retrieved %d senders for %d ids", len(results), len(sender_ids)
            )
            return results
        except Exception as e:
            logger.error("Failed to batch get senders: %s", e)
            return []

    async def upsert_by_sender_id(
        self, sender_id: str, update_data: Dict[str, Any]
    ) -> Optional[Sender]:
        """Update or insert sender by sender_id

        Uses find-then-save pattern with duplicate key retry
        to handle concurrency safely.

        Args:
            sender_id: Sender identifier
            update_data: Fields to update (only non-None values)

        Returns:
            The updated or created Sender document
        """
        try:
            existing_doc = await self.model.find_one({"sender_id": sender_id})

            if existing_doc:
                for key, value in update_data.items():
                    setattr(existing_doc, key, value)
                await existing_doc.save()
                logger.debug("Updated existing sender: sender_id=%s", sender_id)
                return existing_doc

            # Not found, create new
            try:
                new_doc = Sender(sender_id=sender_id, **update_data)
                await new_doc.create()
                logger.info("Created new sender: sender_id=%s", sender_id)
                return new_doc

            except Exception as create_error:
                # Handle concurrent duplicate key
                error_str = str(create_error)
                if "E11000" in error_str and "duplicate key" in error_str:
                    logger.warning(
                        "Concurrent creation conflict, retrying: sender_id=%s",
                        sender_id,
                    )
                    retry_doc = await self.model.find_one({"sender_id": sender_id})
                    if retry_doc:
                        for key, value in update_data.items():
                            setattr(retry_doc, key, value)
                        await retry_doc.save()
                        logger.debug(
                            "Updated after concurrency conflict: sender_id=%s",
                            sender_id,
                        )
                        return retry_doc
                    else:
                        logger.error(
                            "Record not found after concurrency conflict: sender_id=%s",
                            sender_id,
                        )
                        return None
                else:
                    raise create_error

        except Exception as e:
            logger.error("Failed to upsert sender: %s", e)
            return None
