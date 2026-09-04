"""Contract tests for the backend-neutral derived-index boundary."""

from __future__ import annotations

import datetime as dt
import os
import uuid
from pathlib import Path

import pytest

from everos.infra.persistence.index import (
    ALL_REPOS,
    Episode,
    IndexBackend,
    IndexRepository,
    all_of,
    episode_repo,
    eq,
)
from everos.infra.persistence.index.lancedb import lance_index_backend, render_predicate
from everos.infra.persistence.index.milvus import milvus_index_backend
from everos.infra.persistence.index.schema import IndexFieldKind, schema_for
from everos.infra.persistence.lancedb import lancedb_manager


def _behavioural_backends() -> list[str]:
    """Backends whose behaviour this module can actually exercise.

    The structural half of the contract needs no server and covers both
    adapters unconditionally. Driving records through one does: Milvus is
    remote, so it joins only when EVEROS_TEST_MILVUS_URI points at a server.
    """
    backends = ["lancedb"]
    if os.environ.get("EVEROS_TEST_MILVUS_URI"):
        backends.append("milvus")
    return backends


@pytest.fixture(params=_behavioural_backends(), ids=lambda b: f"index={b}")
async def index_backend(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Route the shared repositories at one backend and clean up after."""
    backend = request.param
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_INDEX__BACKEND", backend)
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()

    if backend == "milvus":
        monkeypatch.setenv("EVEROS_MILVUS__URI", os.environ["EVEROS_TEST_MILVUS_URI"])
        monkeypatch.setenv(
            "EVEROS_MILVUS__TOKEN", os.environ.get("EVEROS_TEST_MILVUS_TOKEN", "")
        )
        monkeypatch.setenv(
            "EVEROS_MILVUS__DB_NAME", os.environ.get("EVEROS_TEST_MILVUS_DB_NAME", "")
        )
        monkeypatch.setenv(
            "EVEROS_MILVUS__COLLECTION_PREFIX", f"everos_ct_{uuid.uuid4().hex}"
        )

    from everos.config import load_settings

    load_settings.cache_clear()
    yield backend

    if backend == "milvus":
        from everos.infra.persistence.index import drop_business_tables

        load_settings.cache_clear()
        try:
            await drop_business_tables()
        finally:
            from everos.infra.persistence.index import shutdown

            await shutdown()
    await lancedb_manager.dispose_connection()


def _episode(number: int, *, owner_id: str = "owner") -> Episode:
    return Episode(
        id=f"{owner_id}_ep_{number:04d}",
        entry_id=f"ep_{number:04d}",
        owner_id=owner_id,
        owner_type="user",
        session_id="session",
        timestamp=dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=number),
        parent_id="memcell",
        sender_ids=[owner_id],
        subject="subject",
        summary="summary",
        episode=f"portable record {number}",
        episode_tokens=f"portable record {number}",
        md_path=f"users/{owner_id}/episodes/episode.md",
        content_sha256=f"{number:064x}",
    )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(lance_index_backend, id="lancedb"),
        pytest.param(milvus_index_backend, id="milvus"),
    ],
)
def test_every_backend_satisfies_the_ports(backend: IndexBackend) -> None:
    """Both adapters, not just the default one.

    This is the only assertion that says the two implementations answer to the
    same contract, so checking one of them was checking nothing -- a second
    adapter could omit a method or register a different table set and no test
    would notice. Needs no server: the ports and the registered repository set
    are structural.
    """
    assert isinstance(backend, IndexBackend)
    assert {repo.table_name for repo in backend.repositories} == {
        repo.table_name for repo in ALL_REPOS
    }
    assert all(isinstance(repo, IndexRepository) for repo in backend.repositories)


def test_the_registered_repository_set_is_the_seven_business_tables() -> None:
    assert len({repo.table_name for repo in ALL_REPOS}) == len(ALL_REPOS) == 7
    assert all(isinstance(repo, IndexRepository) for repo in ALL_REPOS)


def test_all_logical_schemas_are_portable_and_validate_bm25_fields() -> None:
    for repo in ALL_REPOS:
        logical = schema_for(repo.schema)
        assert logical.table_name == repo.table_name
        declared = {field.name for field in logical.fields}
        assert set(logical.bm25_fields) <= declared
        for field in logical.vector_fields:
            assert field.kind is IndexFieldKind.DENSE_VECTOR
            assert field.dimension == 1024


def test_lance_predicate_renderer_owns_escaping() -> None:
    rendered = render_predicate(
        all_of(eq("owner_id", "o'reilly"), eq("session_id", "session"))
    )
    assert "owner_id = 'o''reilly'" in rendered
    assert "session_id = 'session'" in rendered


async def test_repository_port_crud_count_and_uncapped_scan(
    index_backend: str,
) -> None:
    rows = [_episode(number) for number in range(101)]
    await episode_repo.upsert(rows)

    owner = eq("owner_id", "owner")
    assert await episode_repo.count() == 101
    assert await episode_repo.count_where(owner) == 101
    assert len(await episode_repo.scan(owner)) == 101

    found = await episode_repo.find_one_where(eq("id", rows[0].id))
    assert found is not None and found.entry_id == rows[0].entry_id

    await episode_repo.update(
        {"subject": "updated"},
        where=eq("id", rows[0].id),
    )
    updated = await episode_repo.get_by_id(rows[0].id)
    assert updated is not None and updated.subject == "updated"

    await episode_repo.delete(eq("id", rows[0].id))
    assert await episode_repo.count_where(owner) == 100


async def test_repository_maintenance_contract_is_executable(
    index_backend: str,
) -> None:
    await episode_repo.upsert([_episode(1)])
    await episode_repo.optimize()
    await episode_repo.prune(dt.timedelta(days=7))
    await episode_repo.rebuild_indexes()
