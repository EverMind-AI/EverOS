"""Get use case — lazy singleton wiring for ``POST /api/v1/memory/get``.

Mirrors :mod:`everos.service.search`: the :class:`GetManager` and its
LanceDB repo singletons are built on first call so the FastAPI module
import order stays decoupled from the lifespan that brings up LanceDB.

``/get`` is read-only and uses no embedding / LLM / rerank clients —
it never blocks on optional components the way ``/search`` does.
"""

from __future__ import annotations

from everos.core.observability.logging import get_logger
from everos.infra.persistence.lancedb import (
    agent_case_repo,
    agent_skill_repo,
    episode_repo,
    user_profile_repo,
)
from everos.memory.get import GetManager, GetRequest, GetResponse

logger = get_logger(__name__)

_manager: GetManager | None = None


def _get_manager() -> GetManager:
    global _manager
    if _manager is None:
        _manager = GetManager(
            episode_repo=episode_repo,
            agent_case_repo=agent_case_repo,
            agent_skill_repo=agent_skill_repo,
            user_profile_repo=user_profile_repo,
        )
        logger.info("get_manager_built")
    return _manager


async def get(req: GetRequest) -> GetResponse:
    """Dispatch one /get request through the lazily-built manager."""
    return await _get_manager().get(req)
