"""Every business table declares its primary key; schema_for enforces it and
stamps the configured vector width."""

from __future__ import annotations

from typing import ClassVar

import pytest

from everos.infra.persistence.index import check_primary_keys
from everos.infra.persistence.index.schema import schema_for
from everos.infra.persistence.records import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    IndexRecord,
    KnowledgeTopic,
    UserProfile,
)
from tests.fakes import TEST_DIM

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
    model: type[IndexRecord],
) -> None:
    assert "PRIMARY_KEY" in vars(model), "declared on the class itself, not inherited"
    assert model.PRIMARY_KEY[:2] == ("app_id", "project_id")
    assert schema_for(model, dim=TEST_DIM).primary_key == ("app_id", "project_id", "id")


def test_undeclared_table_is_rejected_by_schema_for() -> None:
    class _Undeclared(IndexRecord):
        TABLE_NAME: ClassVar[str] = "_undeclared"
        id: str

    with pytest.raises(ValueError, match="has no PRIMARY_KEY"):
        schema_for(_Undeclared, dim=TEST_DIM)


def test_unknown_primary_key_column_is_rejected() -> None:
    class _Bad(IndexRecord):
        TABLE_NAME: ClassVar[str] = "_bad"
        PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "nope", "id")
        id: str
        app_id: str = "default"

    with pytest.raises(ValueError, match="nope"):
        schema_for(_Bad, dim=TEST_DIM)


def test_empty_primary_key_is_rejected() -> None:
    class _Empty(IndexRecord):
        TABLE_NAME: ClassVar[str] = "_empty"
        PRIMARY_KEY: ClassVar[tuple[str, ...]] = ()
        id: str

    with pytest.raises(ValueError, match="empty PRIMARY_KEY"):
        schema_for(_Empty, dim=TEST_DIM)


def test_check_primary_keys_rejects_undeclared_schema() -> None:
    class _Undeclared(IndexRecord):
        TABLE_NAME: ClassVar[str] = "_undeclared"
        id: str

    with pytest.raises(ValueError, match="_Undeclared has no PRIMARY_KEY"):
        check_primary_keys((_Undeclared,), dim=TEST_DIM)


def test_check_primary_keys_accepts_every_business_table() -> None:
    check_primary_keys(BUSINESS_TABLES, dim=TEST_DIM)


def test_vector_width_comes_from_the_argument() -> None:
    """The record names no width; ``schema_for`` stamps the one it is given."""
    for dim in (3, 768, 4096):
        fields = schema_for(Episode, dim=dim).vector_fields
        assert {field.name for field in fields} == {"vector", "subject_vector"}
        assert {field.dimension for field in fields} == {dim}


@pytest.mark.parametrize("dim", [0, -1])
def test_a_non_positive_width_is_refused(dim: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        schema_for(Episode, dim=dim)
