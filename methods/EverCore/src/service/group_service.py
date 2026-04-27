"""
Group service

Provides business logic for group CRUD operations.
"""

import logging
from typing import Optional

from core.di import service
from core.di.utils import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.group_raw_repository import (
    GroupRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.group import Group
from api_specs.dtos.group import GroupResponse

logger = logging.getLogger(__name__)


@service("group_service")
class GroupService:
    """
    Group service

    Provides:
    - Create or update a group (upsert)
    - Get group by group_id
    - Partial update group fields
    - Auto-registration during memorize (fire-and-forget)
    """

    def __init__(self):
        self._repository: Optional[GroupRawRepository] = None

    def _get_repository(self) -> GroupRawRepository:
        """Get repository (lazy loading)"""
        if self._repository is None:
            self._repository = get_bean_by_type(GroupRawRepository)
        return self._repository

    def _to_response(self, doc: Group) -> GroupResponse:
        """Convert Group document to response DTO"""
        return GroupResponse(
            group_id=doc.group_id,
            name=doc.name,
            description=doc.description,
            created_at=doc.created_at.isoformat() if doc.created_at else "",
            updated_at=doc.updated_at.isoformat() if doc.updated_at else "",
        )

    async def create_or_update(
        self,
        group_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[GroupResponse]:
        """Create or update a group (upsert by group_id)

        Args:
            group_id: Group identifier
            name: Group display name
            description: Group description

        Returns:
            GroupResponse or None if failed
        """
        repo = self._get_repository()

        update_data = {}
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description

        doc = await repo.upsert_by_group_id(group_id, update_data)
        if not doc:
            logger.error("Failed to create/update group: group_id=%s", group_id)
            return None

        logger.info("Group created/updated: group_id=%s", group_id)
        return self._to_response(doc)

    async def get_by_group_id(self, group_id: str) -> Optional[GroupResponse]:
        """Get group by group_id

        Args:
            group_id: Group identifier

        Returns:
            GroupResponse or None if not found
        """
        repo = self._get_repository()
        doc = await repo.get_by_group_id(group_id)
        if not doc:
            return None
        return self._to_response(doc)

    async def patch(
        self,
        group_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[GroupResponse]:
        """Partial update group fields

        Args:
            group_id: Group identifier
            name: New display name (if provided)
            description: New description (if provided)

        Returns:
            GroupResponse or None if not found
        """
        repo = self._get_repository()

        doc = await repo.get_by_group_id(group_id)
        if not doc:
            return None

        updated = False
        if name is not None:
            doc.name = name
            updated = True
        if description is not None:
            doc.description = description
            updated = True

        if updated:
            await doc.save()
            logger.info("Group patched: group_id=%s", group_id)

        return self._to_response(doc)

    async def ensure_group_exists(
        self,
        group_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Ensure a group exists (auto-registration during memorize)

        Creates the group if it doesn't exist. If it exists and name/description
        are provided, updates them (PATCH-merge semantics).
        Designed to be called as fire-and-forget via asyncio.create_task().

        Args:
            group_id: Group identifier
            name: Group display name (optional)
            description: Group description (optional)
        """
        try:
            repo = self._get_repository()

            update_data = {}
            if name is not None:
                update_data["name"] = name
            if description is not None:
                update_data["description"] = description

            await repo.upsert_by_group_id(group_id, update_data)
            logger.debug("Group auto-registered: group_id=%s", group_id)
        except Exception as e:
            # Fire-and-forget: log error but don't raise
            logger.warning(
                "Failed to auto-register group: group_id=%s, error=%s", group_id, e
            )
