"""
Sender service

Provides business logic for sender CRUD operations.
"""

import logging
from typing import Dict, List, Optional

from core.di import service
from core.di.utils import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.sender_raw_repository import (
    SenderRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.sender import Sender
from api_specs.dtos.sender import SenderResponse

logger = logging.getLogger(__name__)


@service("sender_service")
class SenderService:
    """
    Sender service

    Provides:
    - Create or update a sender (upsert)
    - Get sender by sender_id
    - Partial update sender fields
    - Auto-registration during memorize (fire-and-forget)
    """

    def __init__(self):
        self._repository: Optional[SenderRawRepository] = None

    def _get_repository(self) -> SenderRawRepository:
        """Get repository (lazy loading)"""
        if self._repository is None:
            self._repository = get_bean_by_type(SenderRawRepository)
        return self._repository

    def _to_response(self, doc: Sender) -> SenderResponse:
        """Convert Sender document to response DTO"""
        return SenderResponse(
            sender_id=doc.sender_id,
            name=doc.name,
            created_at=doc.created_at.isoformat() if doc.created_at else "",
            updated_at=doc.updated_at.isoformat() if doc.updated_at else "",
        )

    async def create_or_update(
        self, sender_id: str, name: Optional[str] = None
    ) -> Optional[SenderResponse]:
        """Create or update a sender (upsert by sender_id)

        Args:
            sender_id: Sender identifier
            name: Sender display name

        Returns:
            SenderResponse or None if failed
        """
        repo = self._get_repository()

        update_data = {}
        if name is not None:
            update_data["name"] = name

        doc = await repo.upsert_by_sender_id(sender_id, update_data)
        if not doc:
            logger.error("Failed to create/update sender: sender_id=%s", sender_id)
            return None

        logger.info("Sender created/updated: sender_id=%s", sender_id)
        return self._to_response(doc)

    async def get_by_sender_id(self, sender_id: str) -> Optional[SenderResponse]:
        """Get sender by sender_id

        Args:
            sender_id: Sender identifier

        Returns:
            SenderResponse or None if not found
        """
        repo = self._get_repository()
        doc = await repo.get_by_sender_id(sender_id)
        if not doc:
            return None
        return self._to_response(doc)

    async def patch(
        self, sender_id: str, name: Optional[str] = None
    ) -> Optional[SenderResponse]:
        """Partial update sender fields

        Args:
            sender_id: Sender identifier
            name: New display name (if provided)

        Returns:
            SenderResponse or None if not found
        """
        repo = self._get_repository()

        doc = await repo.get_by_sender_id(sender_id)
        if not doc:
            return None

        if name is not None:
            doc.name = name
            await doc.save()
            logger.info("Sender patched: sender_id=%s", sender_id)

        return self._to_response(doc)

    async def batch_get_sender_names(self, sender_ids: List[str]) -> Dict[str, str]:
        """Batch get sender display names by sender_ids.

        Only returns entries where the sender has a non-empty name stored.

        Args:
            sender_ids: List of sender identifiers

        Returns:
            Dict mapping sender_id to display name
        """
        if not sender_ids:
            return {}
        repo = self._get_repository()
        docs = await repo.get_by_sender_ids(sender_ids)
        return {doc.sender_id: doc.name for doc in docs if doc.name}

    async def ensure_sender_exists(
        self, sender_id: str, name: Optional[str] = None
    ) -> None:
        """Ensure a sender exists (auto-registration during memorize)

        Creates the sender if it doesn't exist. If it exists and name is
        provided, updates it.
        Designed to be called as fire-and-forget via asyncio.create_task().

        Args:
            sender_id: Sender identifier
            name: Sender display name (optional)
        """
        try:
            repo = self._get_repository()

            update_data = {}
            if name is not None:
                update_data["name"] = name

            await repo.upsert_by_sender_id(sender_id, update_data)
            logger.debug("Sender auto-registered: sender_id=%s", sender_id)
        except Exception as e:
            # Fire-and-forget: log error but don't raise
            logger.warning(
                "Failed to auto-register sender: sender_id=%s, error=%s", sender_id, e
            )
