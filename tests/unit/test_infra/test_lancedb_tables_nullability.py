"""Verify vector columns on business tables are nullable.

Soft-dependency embedding mode lets cascade handlers write rows with
``vector=None`` when no embedding provider is configured. Each business
table's ``vector`` field annotation must therefore include ``None`` in its
type union (``Vector(_DIM) | None``), not just the bare vector type.

The pydantic-level annotation check alone is not sufficient: LanceDB's
``alter_columns`` (see ``migrate_table_schemas`` in
``everos.infra.persistence.lancedb``) operates on the **pyarrow** schema,
not the pydantic one. This module also asserts
``to_arrow_schema().field("vector").nullable`` directly, so a regression
in the Pydantic->PyArrow conversion (e.g. LanceDB's converter failing to
carry ``| None`` through) is caught here rather than only surfacing as a
migration failure at startup.
"""

from __future__ import annotations

import pytest

from everos.infra.persistence.lancedb.tables import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
)


def _vector_field_type(cls, field_name: str = "vector"):
    """Return the resolved annotation of a Vector field."""
    return cls.model_fields[field_name].annotation


def test_episode_vector_nullable():
    """Verify: vector: Vector(_DIM) | None"""
    ann = str(_vector_field_type(Episode))
    assert "None" in ann


def test_episode_subject_vector_nullable():
    """Verify: subject_vector: Vector(_DIM) | None"""
    ann = str(_vector_field_type(Episode, "subject_vector"))
    assert "None" in ann


def test_atomic_fact_vector_nullable():
    ann = str(_vector_field_type(AtomicFact))
    assert "None" in ann


def test_foresight_vector_nullable():
    ann = str(_vector_field_type(Foresight))
    assert "None" in ann


def test_agent_case_vector_nullable():
    ann = str(_vector_field_type(AgentCase))
    assert "None" in ann


def test_agent_skill_vector_nullable():
    ann = str(_vector_field_type(AgentSkill))
    assert "None" in ann


def test_knowledge_topic_vector_nullable():
    ann = str(_vector_field_type(KnowledgeTopic))
    assert "None" in ann


def test_dim_unchanged():
    """_DIM stays at 1024 across all table modules — only nullability moves."""
    from everos.infra.persistence.lancedb.tables import agent_case as m1
    from everos.infra.persistence.lancedb.tables import agent_skill as m2
    from everos.infra.persistence.lancedb.tables import atomic_fact as m3
    from everos.infra.persistence.lancedb.tables import episode as m4
    from everos.infra.persistence.lancedb.tables import foresight as m5
    from everos.infra.persistence.lancedb.tables import knowledge_topic as m6

    for module in (m1, m2, m3, m4, m5, m6):
        assert module._DIM == 1024


@pytest.mark.parametrize(
    "cls",
    [Episode, AtomicFact, Foresight, AgentCase, AgentSkill, KnowledgeTopic],
)
def test_vector_nullable_at_pyarrow_layer(cls):
    """``alter_columns`` (the migration primitive) operates on the pyarrow
    schema, not the pydantic annotation — assert nullability there too."""
    arrow_schema = cls.to_arrow_schema()
    assert arrow_schema.field("vector").nullable is True


def test_episode_subject_vector_nullable_at_pyarrow_layer():
    """``subject_vector`` was already nullable pre-refactor and is not part
    of the v2 migration; confirm the pyarrow schema still agrees."""
    arrow_schema = Episode.to_arrow_schema()
    assert arrow_schema.field("subject_vector").nullable is True
