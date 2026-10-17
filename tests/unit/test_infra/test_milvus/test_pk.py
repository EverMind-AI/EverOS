"""The Milvus primary key is the declared PRIMARY_KEY joined into ``__pk``."""

from __future__ import annotations

import pytest

from everos.infra.persistence.index import Episode, eq
from everos.infra.persistence.index.schema import schema_for
from everos.infra.persistence.milvus.repository import (
    MilvusValueLimitError,
    _compose_pk,
)


def test_compose_pk_joins_primary_key_in_declared_order() -> None:
    row = {
        "app_id": "claude-code",
        "project_id": "alpha",
        "id": "u1_ep_20260914_00000001",
    }
    assert (
        _compose_pk(schema_for(Episode), row)
        == "claude-code/alpha/u1_ep_20260914_00000001"
    )


def test_compose_pk_rejects_separator_in_any_component() -> None:
    row = {"app_id": "claude-code", "project_id": "a/b", "id": "x"}
    with pytest.raises(MilvusValueLimitError, match="project_id contains '/'"):
        _compose_pk(schema_for(Episode), row)


def test_compose_pk_rejects_overlong_key() -> None:
    row = {"app_id": "a" * 40_000, "project_id": "b" * 30_000, "id": "x"}
    with pytest.raises(MilvusValueLimitError, match="UTF-8 bytes"):
        _compose_pk(schema_for(Episode), row)


async def test_update_refuses_to_change_primary_key_columns() -> None:
    """``update()`` re-upserts the whole row under a recomputed ``__pk``; a
    primary-key column in ``updates`` would leave the old row behind."""
    from everos.infra.persistence.milvus.repos import episode_repo

    repo = episode_repo._repo() if hasattr(episode_repo, "_repo") else episode_repo
    with pytest.raises(ValueError, match="primary-key column"):
        await repo.update({"project_id": "beta"}, where=eq("id", "u1_ep_1"))
