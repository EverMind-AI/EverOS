"""
Session raw data repository

Provides CRUD operations for Session documents (v1_sessions collection).
"""

from typing import Optional, Dict, Any

from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.session import Session
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("session_raw_repository", primary=True)
class SessionRawRepository(BaseRepository[Session]):
    """
    Session raw data repository

    Provides CRUD operations and query capabilities for Session data.
    """

    def __init__(self):
        super().__init__(Session)

    async def get_by_session_id(self, session_id: str) -> Optional[Session]:
        """Get session by session_id"""
        try:
            result = await self.model.find_one({"session_id": session_id})
            if result:
                logger.debug("Retrieved session: session_id=%s", session_id)
            else:
                logger.debug("Session not found: session_id=%s", session_id)
            return result
        except Exception as e:
            logger.error("Failed to get session by session_id: %s", e)
            return None

    async def upsert_by_session_id(
        self, session_id: str, update_data: Dict[str, Any]
    ) -> Optional[Session]:
        """Update or insert session by session_id

        Uses find-then-save pattern with duplicate key retry
        to handle concurrency safely.

        Args:
            session_id: Session identifier
            update_data: Fields to update (only non-None values)

        Returns:
            The updated or created Session document
        """
        try:
            existing_doc = await self.model.find_one({"session_id": session_id})

            if existing_doc:
                for key, value in update_data.items():
                    setattr(existing_doc, key, value)
                await existing_doc.save()
                logger.debug("Updated existing session: session_id=%s", session_id)
                return existing_doc

            # Not found, create new
            try:
                new_doc = Session(session_id=session_id, **update_data)
                await new_doc.create()
                logger.info("Created new session: session_id=%s", session_id)
                return new_doc

            except Exception as create_error:
                # Handle concurrent duplicate key
                error_str = str(create_error)
                if "E11000" in error_str and "duplicate key" in error_str:
                    logger.warning(
                        "Concurrent creation conflict, retrying: session_id=%s",
                        session_id,
                    )
                    retry_doc = await self.model.find_one({"session_id": session_id})
                    if retry_doc:
                        for key, value in update_data.items():
                            setattr(retry_doc, key, value)
                        await retry_doc.save()
                        logger.debug(
                            "Updated after concurrency conflict: session_id=%s",
                            session_id,
                        )
                        return retry_doc
                    else:
                        logger.error(
                            "Record not found after concurrency conflict: "
                            "session_id=%s",
                            session_id,
                        )
                        return None
                else:
                    raise create_error

        except Exception as e:
            logger.error("Failed to upsert session: %s", e)
            return None
