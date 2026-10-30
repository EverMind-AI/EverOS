"""Verify vector columns on business tables are nullable.

Soft-dependency embedding mode lets cascade handlers write rows with
``vector=None`` when no embedding provider is configured. Each business
table's ``vector`` field must therefore be declared ``DenseVector | None``.

The neutral schema check alone is not sufficient: LanceDB's
``alter_columns`` (see ``migrate_table_schemas`` in
``everos.infra.persistence.lancedb``) operates on the **Arrow** schema. This
module also asserts the derived Arrow field is nullable, so a regression in
the neutral -> Arrow mapping is caught here rather than only surfacing as a
migration failure at startup.
"""

from __future__ import annotations

import pytest

from everos.infra.persistence.index import schema_for
from everos.infra.persistence.records import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
)
from tests.fakes import TEST_DIM, arrow_schema

_VECTOR_COLUMNS = [
    (Episode, "vector"),
    (Episode, "subject_vector"),
    (AtomicFact, "vector"),
    (Foresight, "vector"),
    (AgentCase, "vector"),
    (AgentSkill, "vector"),
    (KnowledgeTopic, "vector"),
]


@pytest.mark.parametrize(
    ("cls", "column"), _VECTOR_COLUMNS, ids=lambda v: getattr(v, "__name__", v)
)
def test_vector_is_declared_nullable(cls: type, column: str) -> None:
    field = schema_for(cls, dim=TEST_DIM).field(column)
    assert field.nullable is True
    assert field.dimension == TEST_DIM


@pytest.mark.parametrize(
    ("cls", "column"), _VECTOR_COLUMNS, ids=lambda v: getattr(v, "__name__", v)
)
def test_vector_nullable_at_pyarrow_layer(cls: type, column: str) -> None:
    """``alter_columns`` (the migration primitive) operates on the Arrow
    schema, not the pydantic annotation — assert nullability there too."""
    assert arrow_schema(cls).field(column).nullable is True


def test_width_is_whatever_the_caller_configures() -> None:
    """The records carry no width: the same class yields any ``dim``."""
    assert arrow_schema(Episode, dim=768).field("vector").type.list_size == 768
    assert arrow_schema(Episode, dim=3072).field("vector").type.list_size == 3072
