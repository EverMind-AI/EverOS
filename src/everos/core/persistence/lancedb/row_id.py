"""LanceDB storage identities and stable public wire identities.

LanceDB upserts need an identifier that is unique across the complete
``(app, project, owner, logical id)`` partition.  Public API identifiers keep
their historical shape for compatibility and are deliberately constructed
separately at the response boundary.
"""

from __future__ import annotations

STORAGE_ID_GENERATION = 2


def make_storage_id(*parts: str) -> str:
    """Return an injective length-prefixed encoding of UTF-8 string parts."""
    return "".join(f"{len(part.encode('utf-8'))}:{part}" for part in parts)


def daily_log_storage_id(
    *, app_id: str, project_id: str, owner_id: str, entry_id: str
) -> str:
    """Storage primary key for episode/fact/foresight/case rows."""
    return make_storage_id(app_id, project_id, owner_id, entry_id)


def agent_skill_storage_id(
    *, app_id: str, project_id: str, owner_id: str, name: str
) -> str:
    """Storage primary key for one named agent skill."""
    return make_storage_id(app_id, project_id, owner_id, name)


def user_profile_storage_id(*, app_id: str, project_id: str, owner_id: str) -> str:
    """Storage primary key for one scoped user profile."""
    return make_storage_id(app_id, project_id, owner_id)


def daily_log_wire_id(*, owner_id: str, entry_id: str) -> str:
    """Historical HTTP identifier for a daily-log row."""
    return f"{owner_id}_{entry_id}"


def agent_skill_wire_id(*, owner_id: str, name: str) -> str:
    """Historical HTTP identifier for an agent skill."""
    return f"{owner_id}_{name}"


def user_profile_wire_id(*, owner_id: str) -> str:
    """Historical HTTP identifier for a user profile."""
    return owner_id


__all__ = [
    "STORAGE_ID_GENERATION",
    "agent_skill_storage_id",
    "agent_skill_wire_id",
    "daily_log_storage_id",
    "daily_log_wire_id",
    "make_storage_id",
    "user_profile_storage_id",
    "user_profile_wire_id",
]
