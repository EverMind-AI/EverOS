"""The Milvus primary key is the declared PRIMARY_KEY joined into ``__pk``."""

from __future__ import annotations

import pytest

from everos.config.settings import MilvusSettings
from everos.infra.persistence.index import Episode, eq
from everos.infra.persistence.index.schema import schema_for
from everos.infra.persistence.milvus import MilvusConnection, MilvusIndexBackend
from everos.infra.persistence.milvus._codec import compose_pk
from everos.infra.persistence.milvus.repository import MilvusValueLimitError


def test_compose_pk_joins_primary_key_in_declared_order() -> None:
    row = {
        "app_id": "claude-code",
        "project_id": "alpha",
        "id": "u1_ep_20260914_00000001",
    }
    assert (
        compose_pk(schema_for(Episode), row)
        == "claude-code/alpha/u1_ep_20260914_00000001"
    )


def test_compose_pk_rejects_separator_in_any_component() -> None:
    row = {"app_id": "claude-code", "project_id": "a/b", "id": "x"}
    with pytest.raises(MilvusValueLimitError, match="project_id contains '/'"):
        compose_pk(schema_for(Episode), row)


def test_compose_pk_rejects_overlong_key() -> None:
    row = {"app_id": "a" * 40_000, "project_id": "b" * 30_000, "id": "x"}
    with pytest.raises(MilvusValueLimitError, match="UTF-8 bytes"):
        compose_pk(schema_for(Episode), row)


async def test_update_refuses_to_change_primary_key_columns() -> None:
    """``update()`` re-upserts the whole row under a recomputed ``__pk``; a
    primary-key column in ``updates`` would leave the old row behind."""
    repo = MilvusIndexBackend(MilvusConnection(MilvusSettings())).repositories.episode
    with pytest.raises(ValueError, match="primary-key column"):
        await repo.update({"project_id": "beta"}, where=eq("id", "u1_ep_1"))
