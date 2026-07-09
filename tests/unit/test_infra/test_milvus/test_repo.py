"""Real Milvus Lite coverage for the optional derived index backend."""

from __future__ import annotations

import datetime as dt
import importlib.util
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from everos.config import load_settings

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymilvus") is None,
    reason="pymilvus optional extra is not installed",
)


def test_milvus_score_normalization_handles_lite_and_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from everos.infra.persistence.milvus import repository

    monkeypatch.delenv("EVEROS_MILVUS__URI", raising=False)
    monkeypatch.setattr(repository, "_milvus_lite_version", lambda: "3.0")
    load_settings.cache_clear()
    assert repository._cosine_distance_from_milvus(0.25) == pytest.approx(0.25)
    assert repository._bm25_score_from_distance(-1.5) == pytest.approx(1.5)

    monkeypatch.setattr(repository, "_milvus_lite_version", lambda: "3.0.0")
    assert repository._cosine_distance_from_milvus(0.25) == pytest.approx(0.25)

    monkeypatch.setenv("EVEROS_MILVUS__URI", "https://example.zillizcloud.com")
    load_settings.cache_clear()
    assert repository._cosine_distance_from_milvus(0.75) == pytest.approx(0.25)
    assert repository._bm25_score_from_distance(1.5) == pytest.approx(1.5)

    monkeypatch.delenv("EVEROS_MILVUS__URI", raising=False)
    monkeypatch.setattr(repository, "_milvus_lite_version", lambda: "3.0.1")
    load_settings.cache_clear()
    assert repository._cosine_distance_from_milvus(0.75) == pytest.approx(0.25)


@pytest_asyncio.fixture(autouse=True)
async def _milvus_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_INDEX__BACKEND", "milvus")
    monkeypatch.setenv("EVEROS_MILVUS__COLLECTION_PREFIX", f"test_{uuid.uuid4().hex}")
    load_settings.cache_clear()
    yield
    from everos.infra.persistence.index import shutdown

    await shutdown()
    load_settings.cache_clear()


async def test_milvus_episode_repo_upsert_search_filter_and_delete() -> None:
    from everos.infra.persistence.index import Episode, episode_repo
    from everos.memory.search.filters import compile_filters_for_backends

    v_apple = [0.0] * 1024
    v_apple[0] = 1.0
    v_banana = [0.0] * 1024
    v_banana[1] = 1.0
    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

    await episode_repo.upsert(
        [
            Episode(
                id="u1_ep1",
                entry_id="ep1",
                owner_id="u1",
                owner_type="user",
                app_id="default",
                project_id="default",
                session_id="s1",
                timestamp=now,
                parent_id="mc1",
                sender_ids=["user"],
                episode="red apple memory",
                episode_tokens="red apple memory",
                md_path="default_app/default_project/users/u1/episodes/day.md",
                content_sha256="a",
                vector=v_apple,
            ),
            Episode(
                id="u1_ep2",
                entry_id="ep2",
                owner_id="u1",
                owner_type="user",
                app_id="default",
                project_id="default",
                session_id="s1",
                timestamp=now + dt.timedelta(seconds=1),
                parent_id="mc2",
                sender_ids=["assistant"],
                episode="blue banana memory",
                episode_tokens="blue banana memory",
                md_path="default_app/default_project/users/u1/episodes/day.md",
                content_sha256="b",
                vector=v_banana,
            ),
        ]
    )

    filters = compile_filters_for_backends(None, owner_id="u1", owner_type="user")
    rows = await episode_repo.find_where(filters, limit=10)
    assert {row.id for row in rows} == {"u1_ep1", "u1_ep2"}
    assert rows[0].timestamp.utcoffset() == dt.timedelta(0)
    assert await episode_repo.count() == 2

    sparse = await episode_repo.sparse_search(
        ["apple"], filters, columns=Episode.BM25_FIELDS, limit=5
    )
    assert [row["id"] for row in sparse] == ["u1_ep1"]
    assert sparse[0]["_score"] > 0

    dense = await episode_repo.dense_search(v_apple, filters, limit=5)
    assert dense[0]["id"] == "u1_ep1"
    assert dense[0]["_distance"] == pytest.approx(0.0)

    deleted = await episode_repo.delete_by_md_path(
        "default_app/default_project/users/u1/episodes/day.md"
    )
    assert deleted == 2
    assert await episode_repo.count() == 0
