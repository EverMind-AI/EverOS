"""
Session service

Provides auto-registration logic for sessions during memorize.
"""

import logging
from typing import Optional

from core.di import service
from core.di.utils import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.session_raw_repository import (
    SessionRawRepository,
)

logger = logging.getLogger(__name__)


@service("session_service")
class SessionService:
    """
    Session service

    Provides:
    - Auto-registration during memorize (fire-and-forget)
    """

    def __init__(self):
        self._repository: Optional[SessionRawRepository] = None

    def _get_repository(self) -> SessionRawRepository:
        """Get repository (lazy loading)"""
        if self._repository is None:
            self._repository = get_bean_by_type(SessionRawRepository)
        return self._repository

    async def ensure_session_exists(
        self,
        session_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Ensure a session exists (auto-registration during memorize)

        Creates the session if it doesn't exist. If it exists and name/description
        are provided, updates them (PATCH-merge semantics).
        Designed to be called as fire-and-forget via asyncio.create_task().

        Args:
            session_id: Session identifier
            name: Session display name (optional)
            description: Session description (optional)
        """
        try:
            repo = self._get_repository()

            update_data = {}
            if name is not None:
                update_data["name"] = name
            if description is not None:
                update_data["description"] = description

            await repo.upsert_by_session_id(session_id, update_data)
            logger.debug("Session auto-registered: session_id=%s", session_id)
        except Exception as e:
            # Fire-and-forget: log error but don't raise
            logger.warning(
                "Failed to auto-register session: session_id=%s, error=%s",
                session_id,
                e,
            )
