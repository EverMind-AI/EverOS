"""Every business table declares its primary key; schema_for enforces it."""

from __future__ import annotations

from typing import ClassVar

import pytest

from everos.core.persistence import BaseLanceTable
from everos.infra.persistence.index import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    UserProfile,
    check_primary_keys,
)
from everos.infra.persistence.index.schema import schema_for

BUSINESS_TABLES = (
    Episode,
    AtomicFact,
    AgentCase,
    Foresight,
    AgentSkill,
    UserProfile,
    KnowledgeTopic,
)


@pytest.mark.parametrize("model", BUSINESS_TABLES, ids=lambda m: m.__name__)
def test_business_table_declares_scoped_primary_key(
    model: type[BaseLanceTable],
) -> None:
    assert "PRIMARY_KEY" in vars(model), "declared on the class itself, not inherited"
    assert model.PRIMARY_KEY[:2] == ("app_id", "project_id")
    assert schema_for(model).primary_key == ("app_id", "project_id", "id")


def test_undeclared_table_is_rejected_by_schema_for() -> None:
    class _Undeclared(BaseLanceTable):
        TABLE_NAME: ClassVar[str] = "_undeclared"
        id: str

    with pytest.raises(ValueError, match="has no PRIMARY_KEY"):
        schema_for(_Undeclared)


def test_unknown_primary_key_column_is_rejected() -> None:
    class _Bad(BaseLanceTable):
        TABLE_NAME: ClassVar[str] = "_bad"
        PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "nope", "id")
        id: str
        app_id: str = "default"

    with pytest.raises(ValueError, match="nope"):
        schema_for(_Bad)


def test_empty_primary_key_is_rejected() -> None:
    class _Empty(BaseLanceTable):
        TABLE_NAME: ClassVar[str] = "_empty"
        PRIMARY_KEY: ClassVar[tuple[str, ...]] = ()
        id: str

    with pytest.raises(ValueError, match="empty PRIMARY_KEY"):
        schema_for(_Empty)


def test_check_primary_keys_rejects_undeclared_schema() -> None:
    class _Undeclared(BaseLanceTable):
        TABLE_NAME: ClassVar[str] = "_undeclared"
        id: str

    with pytest.raises(ValueError, match="_Undeclared has no PRIMARY_KEY"):
        check_primary_keys((_Undeclared,))


def test_check_primary_keys_accepts_every_business_table() -> None:
    check_primary_keys(BUSINESS_TABLES)
