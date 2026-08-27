"""Cross-backend behavior against Milvus Server or Zilliz Cloud.

Set ``EVEROS_TEST_MILVUS_URI`` and, when required,
``EVEROS_TEST_MILVUS_TOKEN``. The same test is used for self-hosted and cloud
endpoints and creates uniquely prefixed, disposable collections.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import uuid

import pytest
import pytest_asyncio

from everos.config import load_settings

_URI = os.environ.get("EVEROS_TEST_MILVUS_URI", "")

pytestmark = pytest.mark.skipif(
    not _URI,
    reason="EVEROS_TEST_MILVUS_URI is not configured",
)


@pytest_asyncio.fixture(autouse=True)
async def _remote_milvus(monkeypatch: pytest.MonkeyPatch):
    prefix = f"everos_e2e_{uuid.uuid4().hex}"
    monkeypatch.setenv("EVEROS_INDEX__BACKEND", "milvus")
    monkeypatch.setenv("EVEROS_MILVUS__URI", _URI)
    monkeypatch.setenv(
        "EVEROS_MILVUS__TOKEN",
        os.environ.get("EVEROS_TEST_MILVUS_TOKEN", ""),
    )
    monkeypatch.setenv(
        "EVEROS_MILVUS__DB_NAME",
        os.environ.get("EVEROS_TEST_MILVUS_DB_NAME", ""),
    )
    monkeypatch.setenv("EVEROS_MILVUS__COLLECTION_PREFIX", prefix)
    load_settings.cache_clear()

    from everos.infra.persistence.index import episode_repo, startup

    try:
        if os.environ.get("EVEROS_TEST_MILVUS_FULL_STARTUP") == "1":
            await startup()
        else:
            await episode_repo._repo().ensure_collection()  # type: ignore[attr-defined]
        yield
    finally:
        from everos.infra.persistence.index import drop_business_tables, shutdown

        await drop_business_tables()
        await shutdown()
        load_settings.cache_clear()


def _episode(
    *,
    row_id: str,
    entry_id: str,
    session_id: str,
    text: str,
    vector_axis: int,
    subject_axis: int,
    timestamp: dt.datetime,
):  # type: ignore[no-untyped-def]
    from everos.infra.persistence.index import Episode

    vector = [0.0] * 1024
    vector[vector_axis] = 1.0
    subject_vector = [0.0] * 1024
    subject_vector[subject_axis] = 1.0
    return Episode(
        id=row_id,
        entry_id=entry_id,
        owner_id="u1",
        owner_type="user",
        app_id="test_app",
        project_id="test_project",
        session_id=session_id,
        timestamp=timestamp,
        parent_id=f"mc_{entry_id}",
        sender_ids=["user"],
        subject=f"subject {text}",
        episode=text,
        episode_tokens=text,
        md_path="test_app/test_project/users/u1/episodes/day.md",
        content_sha256=entry_id,
        vector=vector,
        subject_vector=subject_vector,
    )


async def test_remote_milvus_matches_derived_index_contract() -> None:
    from everos.infra.persistence.index import (
        Episode,
        UserProfile,
        episode_repo,
        eq,
        is_null,
        user_profile_repo,
    )
    from everos.memory.search import FilterNode
    from everos.memory.search.filters import compile_filters

    first_vector = [1.0] + [0.0] * 1023
    first_subject = [0.0, 1.0] + [0.0] * 1022
    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    await episode_repo.upsert(
        [
            _episode(
                row_id="u1_ep1",
                entry_id="ep1",
                session_id="abc=",
                text="red apple memory",
                vector_axis=0,
                subject_axis=1,
                timestamp=now,
            ),
            _episode(
                row_id="u1_ep2",
                entry_id="ep2",
                session_id="other",
                text="blue banana memory",
                vector_axis=1,
                subject_axis=0,
                timestamp=now + dt.timedelta(seconds=1),
            ),
        ]
    )

    where = compile_filters(
        None,
        owner_id="u1",
        owner_type="user",
        app_id="test_app",
        project_id="test_project",
    )
    rows = await episode_repo.find_where(where, limit=10)
    assert {row.id for row in rows} == {"u1_ep1", "u1_ep2"}
    assert await episode_repo.count() == 2

    equals_filter = compile_filters(
        FilterNode.model_validate({"session_id": "abc="}),
        owner_id="u1",
        owner_type="user",
        app_id="test_app",
        project_id="test_project",
    )
    equals_rows = await episode_repo.find_where(equals_filter, limit=10)
    assert [row.id for row in equals_rows] == ["u1_ep1"]

    sparse = await episode_repo.sparse_search(
        ["apple"], where, columns=Episode.BM25_FIELDS, limit=5
    )
    assert sparse[0]["id"] == "u1_ep1"
    assert sparse[0]["_score"] > 0

    dense = await episode_repo.dense_search(first_vector, where, limit=5)
    assert dense[0]["id"] == "u1_ep1"
    assert dense[0]["_distance"] == pytest.approx(0.0, abs=1e-5)

    by_subject = await episode_repo.dense_search(
        first_subject,
        where,
        limit=5,
        vector_field="subject_vector",
    )
    assert by_subject[0]["id"] == "u1_ep1"
    assert by_subject[0]["_distance"] == pytest.approx(0.0, abs=1e-5)

    page, total = await episode_repo.find_where_paginated(
        where,
        sort_by="timestamp",
        page=1,
        page_size=1,
    )
    assert total == 2
    assert len(page) == 1

    concurrent = await asyncio.gather(
        *(episode_repo.find_where(where, limit=10) for _ in range(4))
    )
    assert all(len(result) == 2 for result in concurrent)

    if os.environ.get("EVEROS_TEST_MILVUS_FULL_STARTUP") == "1":
        profile = UserProfile(
            id="u1",
            owner_id="u1",
            owner_type="user",
            app_id="test_app",
            project_id="test_project",
            summary="initial profile",
            explicit_info_json="[]",
            implicit_traits_json="[]",
            profile_timestamp_ms=1,
            md_path="test_app/test_project/users/u1/user.md",
            content_sha256="profile-v1",
        )
        await user_profile_repo.upsert([profile])
        await user_profile_repo.update(
            {"summary": "updated profile"}, where=eq("id", "u1")
        )
        updated_profile = await user_profile_repo.get_by_id("u1")
        assert updated_profile is not None
        assert updated_profile.summary == "updated profile"

    # Exercise the logical-null mapping for physical Milvus vector fields and
    # the iterator-backed scan path. This also guards against reintroducing the
    # old implicit 100-row maintenance cap.
    null_vector_rows = []
    for index in range(101):
        row = _episode(
            row_id=f"u1_null_{index}",
            entry_id=f"null_{index}",
            session_id="null-vectors",
            text=f"unembedded memory {index}",
            vector_axis=0,
            subject_axis=0,
            timestamp=now + dt.timedelta(minutes=index + 1),
        )
        null_vector_rows.append(
            row.model_copy(update={"vector": None, "subject_vector": None})
        )
    await episode_repo.upsert(null_vector_rows)
    assert await episode_repo.count_where(is_null("vector")) == 101
    assert len(await episode_repo.scan()) == 103

    assert (
        await episode_repo.delete_by_md_path(
            "test_app/test_project/users/u1/episodes/day.md"
        )
        == 103
    )
    assert await episode_repo.count() == 0


async def test_update_preserves_vectors_on_a_row_that_has_them() -> None:
    """Milvus has no partial-column update, so update() re-upserts whole rows.

    The only prior coverage ran against ``user_profile``, which carries no
    vector column — so the read-back-and-re-upsert of a 1024-d vector (the
    path backfill and reflection both take) was never exercised remotely.
    """
    from everos.infra.persistence.index import episode_repo, eq

    now = dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.UTC)
    row = _episode(
        row_id="u1_vec_update",
        entry_id="vec_update",
        session_id="vector-update",
        text="vector bearing episode",
        vector_axis=7,
        subject_axis=11,
        timestamp=now,
    )
    await episode_repo.upsert([row])

    await episode_repo.update(
        {"deprecated_by": "u1_replacement"}, where=eq("id", "u1_vec_update")
    )

    stored = await episode_repo.get_by_id("u1_vec_update")
    assert stored is not None
    assert stored.deprecated_by == "u1_replacement"
    # The untouched columns must survive the round-trip intact.
    assert stored.vector is not None
    assert stored.subject_vector is not None
    assert stored.vector[7] == pytest.approx(1.0)
    assert stored.subject_vector[11] == pytest.approx(1.0)
    assert stored.timestamp == now
    assert stored.sender_ids == ["user"]
    assert stored.episode == "vector bearing episode"

    await episode_repo.delete_by_md_path(
        "test_app/test_project/users/u1/episodes/day.md"
    )


async def test_datetime_round_trip_survives_pre_2001_instants() -> None:
    """Epoch-ms values below 1e12 must not be re-read as epoch seconds.

    ``to_timestamp_ms`` always writes milliseconds, so the read side has to
    parse milliseconds unconditionally. A seconds-vs-ms heuristic sends any
    pre-2001-09-09 instant into the year 30000 and raises on the way back.
    """
    from everos.infra.persistence.index import episode_repo, eq

    old = dt.datetime(1999, 1, 1, tzinfo=dt.UTC)
    row = _episode(
        row_id="u1_pre2001",
        entry_id="pre2001",
        session_id="old-instants",
        text="an episode from before the ms/seconds threshold",
        vector_axis=3,
        subject_axis=5,
        timestamp=old,
    )
    await episode_repo.upsert([row])

    stored = await episode_repo.get_by_id("u1_pre2001")
    assert stored is not None
    assert stored.timestamp == old

    assert await episode_repo.count_where(eq("timestamp", old)) == 1

    await episode_repo.delete_by_md_path(
        "test_app/test_project/users/u1/episodes/day.md"
    )
