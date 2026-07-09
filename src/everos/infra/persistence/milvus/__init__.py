"""Milvus derived index backend.

This package mirrors the LanceDB business index surface but stores rows in
Milvus collections. It is selected through ``Settings.index.backend`` and is
normally reached through :mod:`everos.infra.persistence.index`.
"""

from __future__ import annotations

from .milvus_manager import MilvusSchemaMismatchError as MilvusSchemaMismatchError
from .milvus_manager import dispose_connection as dispose_connection
from .milvus_manager import get_client as get_client
from .repos import ALL_REPOS as ALL_REPOS
from .repos import agent_case_repo as agent_case_repo
from .repos import agent_skill_repo as agent_skill_repo
from .repos import atomic_fact_repo as atomic_fact_repo
from .repos import episode_repo as episode_repo
from .repos import foresight_repo as foresight_repo
from .repos import knowledge_topic_repo as knowledge_topic_repo
from .repos import user_profile_repo as user_profile_repo


async def ensure_business_indexes() -> None:
    """Create or verify every EverOS Milvus collection."""
    for repo in ALL_REPOS:
        await repo.ensure_collection()


async def verify_business_schemas() -> None:
    """Fail loud if existing Milvus collections drift from EverOS schemas."""
    for repo in ALL_REPOS:
        await repo.verify_collection()


__all__ = [
    "ALL_REPOS",
    "MilvusSchemaMismatchError",
    "agent_case_repo",
    "agent_skill_repo",
    "atomic_fact_repo",
    "dispose_connection",
    "ensure_business_indexes",
    "episode_repo",
    "foresight_repo",
    "get_client",
    "knowledge_topic_repo",
    "user_profile_repo",
    "verify_business_schemas",
]
