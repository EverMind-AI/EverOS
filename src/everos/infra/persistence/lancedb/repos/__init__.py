"""LanceDB repo singletons (one per business table).

Each repo is a module-level singleton — the table connection is
resolved lazily on first call via :func:`..lancedb_manager.get_table`.
Subclassing :class:`LanceRepoBase` lets each repo carry table-specific
helpers later (e.g. ``find_by_owner``, ``search_for_owner``) without
needing a separate factory.

External usage::

    from everos.infra.persistence.lancedb.repos import (
        episode_repo,
        atomic_fact_repo,
        foresight_repo,
        agent_case_repo,
        agent_skill_repo,
        user_profile_repo,
        knowledge_topic_repo,
    )

    await episode_repo.add([Episode(...)])
"""

from .agent_case import agent_case_repo as agent_case_repo
from .agent_skill import agent_skill_repo as agent_skill_repo
from .atomic_fact import atomic_fact_repo as atomic_fact_repo
from .episode import episode_repo as episode_repo
from .foresight import foresight_repo as foresight_repo
from .knowledge_topic import knowledge_topic_repo as knowledge_topic_repo
from .user_profile import user_profile_repo as user_profile_repo

__all__ = [
    "agent_case_repo",
    "agent_skill_repo",
    "atomic_fact_repo",
    "episode_repo",
    "foresight_repo",
    "knowledge_topic_repo",
    "user_profile_repo",
]
