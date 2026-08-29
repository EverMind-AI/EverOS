"""Contract tests for the backend-neutral derived-index boundary."""

from __future__ import annotations

import datetime as dt
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
from everos.infra.persistence.index.schema import IndexFieldKind, schema_for
from everos.infra.persistence.lancedb import lancedb_manager


@pytest.fixture(autouse=True)
async def _isolated_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
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


def test_all_registered_repositories_and_backend_satisfy_ports() -> None:
    assert isinstance(lance_index_backend, IndexBackend)
    assert {repo.table_name for repo in lance_index_backend.repositories} == {
        repo.table_name for repo in ALL_REPOS
    }
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


async def test_repository_port_crud_count_and_uncapped_scan() -> None:
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


async def test_repository_maintenance_contract_is_executable() -> None:
    await episode_repo.upsert([_episode(1)])
    await episode_repo.optimize()
    await episode_repo.prune(dt.timedelta(days=7))
    await episode_repo.rebuild_indexes()
