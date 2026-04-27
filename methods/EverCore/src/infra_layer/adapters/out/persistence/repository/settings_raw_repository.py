"""
GlobalSettings Raw Repository

Provides database operations for the singleton GlobalSettings document.
"""

import logging
from typing import Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession

from core.oxm.mongo.base_repository import BaseRepository
from core.di.decorators import repository
from core.constants.exceptions import ValidationException
from infra_layer.adapters.out.persistence.document.memory.global_settings import (
    GlobalSettings,
)

logger = logging.getLogger(__name__)


@repository("settings_raw_repository", primary=True)
class GlobalSettingsRawRepository(BaseRepository[GlobalSettings]):
    """
    Repository for the singleton GlobalSettings document (v1_global_settings collection).

    GlobalSettings is a singleton per space. Group-level metadata has moved
    to the Session model (v1_sessions).
    """

    def __init__(self):
        """Initialize repository"""
        super().__init__(GlobalSettings)

    # =========================================================================
    # Singleton methods (primary API)
    # =========================================================================

    async def get_global_settings(
        self, session: Optional[AsyncClientSession] = None
    ) -> Optional[GlobalSettings]:
        """
        Get the singleton GlobalSettings document.

        Args:
            session: Optional MongoDB session for transaction support

        Returns:
            GlobalSettings document or None if not found
        """
        try:
            doc = await self.model.find_one({}, session=session)
            if doc:
                logger.debug("Retrieved global settings")
            return doc
        except Exception as e:
            logger.error("Failed to retrieve global settings: %s", e)
            return None

    async def upsert_global_settings(
        self,
        settings_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[GlobalSettings]:
        """
        Create or update the singleton GlobalSettings document.

        Args:
            settings_data: Dictionary of settings fields to set
            session: Optional MongoDB session

        Returns:
            Updated or created GlobalSettings document

        """
        try:
            existing_doc = await self.model.find_one({}, session=session)

            if existing_doc:
                for key, value in settings_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug("Updated existing global settings")
                return existing_doc

            # No record found, create new singleton
            try:
                new_doc = GlobalSettings(**settings_data)
                await new_doc.insert(session=session)
                logger.info("Created new global settings")
                return new_doc
            except Exception as create_error:
                logger.error(
                    "Failed to create global settings: %s", create_error, exc_info=True
                )
                return None

        except ValidationException:
            raise
        except Exception as e:
            logger.error("Failed to upsert global settings: %s", e, exc_info=True)
            return None

    async def update_global_settings(
        self, update_data: Dict[str, Any], session: Optional[AsyncClientSession] = None
    ) -> Optional[GlobalSettings]:
        """
        Update the singleton GlobalSettings document (must already exist).

        Args:
            update_data: Dictionary of fields to update
            session: Optional MongoDB session

        Returns:
            Updated GlobalSettings document or None if not found

        """
        try:
            doc = await self.model.find_one({}, session=session)
            if not doc:
                return None

            for key, value in update_data.items():
                if hasattr(doc, key):
                    setattr(doc, key, value)
            await doc.save(session=session)
            logger.debug("Updated global settings")
            return doc
        except ValidationException:
            raise
        except Exception as e:
            logger.error("Failed to update global settings: %s", e, exc_info=True)
            return None
