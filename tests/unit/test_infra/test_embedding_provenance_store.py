"""Backends store the embedding model alongside the data it describes.

Vectors from two different models are not comparable, so an index has to
remember which model produced it. Each backend decides *where*: LanceDB hangs
it on the episode table's ``vector`` column as Arrow field metadata, Milvus on
the episode collection as a property. This module pins the LanceDB half
against a real on-disk table — the Milvus half needs a live cluster and lives
in the acceptance run.

The load-bearing property is that attesting does **not** rewrite data: an
existing deployment adopting this field keeps its rows and column types.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from everos.component.utils.datetime import get_utc_now
from everos.infra.persistence import lancedb as lance
from everos.infra.persistence.lancedb import lancedb_manager
from everos.infra.persistence.records.episode import Episode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture(autouse=True)
async def _isolated_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


def _episode_row(entry_id: str) -> dict[str, object]:
    """One valid Episode, built through the model rather than hand-rolled.

    ``model_validate`` is the point: a dict assembled by hand drifts from what
    the product actually writes, and the first version of this helper was
    missing four required fields.
    """
    width = Episode.to_arrow_schema().field("vector").type.list_size
    episode = Episode.model_validate(
        {
            "id": f"u1_{entry_id}",
            "entry_id": entry_id,
            "owner_id": "u1",
            "owner_type": "user",
            "timestamp": get_utc_now(),
            "parent_id": "mc_1",
            "sender_ids": ["u1"],
            "episode": "the body",
            "episode_tokens": "the body",
            "md_path": f"users/u1/episodes/{entry_id}.md",
            "content_sha256": "0" * 64,
            "vector": [0.1] * width,
        }
    )
    return episode.model_dump()


async def test_unattested_index_reads_none() -> None:
    """A fresh index has no record, and that is not an error (C10/P5)."""
    assert await lance.index_backend.read_embedding_model() is None


async def test_roundtrip_survives_a_new_connection() -> None:
    await lance.index_backend.write_embedding_model("fake-768")
    await lancedb_manager.dispose_connection()
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    assert await lance.index_backend.read_embedding_model() == "fake-768"


async def test_attesting_does_not_rewrite_data() -> None:
    """Row count and column type must not move — this is the upgrade path."""
    table = await lancedb_manager.get_table(Episode.TABLE_NAME, Episode)
    await table.add([_episode_row("e0")])
    rows_before = await table.count_rows()
    type_before = (await table.schema()).field("vector").type

    await lance.index_backend.write_embedding_model("fake-768")

    assert await table.count_rows() == rows_before
    assert (await table.schema()).field("vector").type == type_before


async def test_record_survives_writes_and_optimize() -> None:
    """Routine maintenance must not drop the attestation (A13)."""
    await lance.index_backend.write_embedding_model("fake-768")
    table = await lancedb_manager.get_table(Episode.TABLE_NAME, Episode)
    for index in range(3):
        await table.add([_episode_row(f"e{index}")])
    await table.optimize()
    assert await lance.index_backend.read_embedding_model() == "fake-768"


async def test_record_hangs_on_the_episode_vector_column() -> None:
    """Pin *which* column carries it, by reading the table instead of the accessor.

    Going back through ``read_embedding_model`` cannot see this: reader and
    writer would agree on any column. Six tables carry a vector and ``Episode``
    carries two, and the choice is load-bearing — ``subject_vector`` exists
    only on episodes, so attaching it there would make the attestation
    invisible to a deployment that never wrote one.
    """
    await lance.index_backend.write_embedding_model("fake-768")
    table = await lancedb_manager.get_table(Episode.TABLE_NAME, Episode)
    schema = await table.schema()

    on_vector = schema.field("vector").metadata or {}
    assert on_vector.get(lance.EMBEDDING_MODEL_KEY.encode()) == b"fake-768"
    assert not (schema.field("subject_vector").metadata or {})


async def test_rewriting_replaces_the_previous_model() -> None:
    await lance.index_backend.write_embedding_model("fake-768")
    await lance.index_backend.write_embedding_model("fake-1536")
    assert await lance.index_backend.read_embedding_model() == "fake-1536"


def test_both_backends_use_the_one_key() -> None:
    """One definition, re-exported -- not two copies that can drift.

    An earlier version declared the string in each backend, with a comment
    saying a shared constant would be a cycle. It would not: both backends
    already import ``records``, which imports no sibling persistence package.
    """
    pytest.importorskip("pymilvus")
    from everos.infra.persistence import milvus
    from everos.infra.persistence.records import EMBEDDING_MODEL_KEY

    assert lance.EMBEDDING_MODEL_KEY is EMBEDDING_MODEL_KEY
    assert milvus.EMBEDDING_MODEL_KEY is EMBEDDING_MODEL_KEY
