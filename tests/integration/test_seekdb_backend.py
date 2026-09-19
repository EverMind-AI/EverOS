"""Behavioral contract for embedded SeekDB and remote seekdb/OceanBase.

Set ``EVEROS_TEST_SEEKDB_PATH`` for the embedded engine or
``EVEROS_TEST_SEEKDB_HOST`` plus the optional connection variables for a
server. Each test owns a uniquely prefixed set of disposable tables.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from everos.config import load_settings

_PATH = os.environ.get("EVEROS_TEST_SEEKDB_PATH", "")
_HOST = os.environ.get("EVEROS_TEST_SEEKDB_HOST", "")

pytestmark = pytest.mark.skipif(
    not (_PATH or _HOST),
    reason="EVEROS_TEST_SEEKDB_PATH or EVEROS_TEST_SEEKDB_HOST is not configured",
)


@pytest_asyncio.fixture(autouse=True)
async def _seekdb_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[None]:
    prefix = os.environ.get(
        "EVEROS_TEST_SEEKDB_PREFIX", f"everos_e2e_{uuid.uuid4().hex}"
    )
    assert re.fullmatch(r"everos_e2e_[0-9a-f]{32}", prefix)
    monkeypatch.setenv("EVEROS_INDEX__BACKEND", "seekdb")
    monkeypatch.setenv("EVEROS_SEEKDB__TABLE_PREFIX", prefix)
    if _PATH:
        monkeypatch.setenv("EVEROS_SEEKDB__MODE", "embedded")
        monkeypatch.setenv("EVEROS_SEEKDB__PATH", _PATH)
    else:
        monkeypatch.setenv("EVEROS_SEEKDB__MODE", "remote")
        monkeypatch.setenv("EVEROS_SEEKDB__HOST", _HOST)
        for name, default in (
            ("PORT", "2881"),
            ("TENANT", ""),
            ("USER", "root"),
            ("PASSWORD", ""),
            ("DATABASE", "everos_test"),
        ):
            monkeypatch.setenv(
                f"EVEROS_SEEKDB__{name}",
                os.environ.get(f"EVEROS_TEST_SEEKDB_{name}", default),
            )
    load_settings.cache_clear()

    from everos.infra.persistence.index import startup

    await startup()
    try:
        yield
    finally:
        from everos.infra.persistence.index import drop_business_tables, shutdown

        try:
            await drop_business_tables()
        finally:
            await shutdown()
            load_settings.cache_clear()


def _episode(number: int, *, vector: bool = True, **overrides: object):  # type: ignore[no-untyped-def]
    from everos.infra.persistence.index import Episode

    dense = [0.0] * 1024
    dense[number % 2] = 1.0
    values: dict[str, object] = dict(
        id=f"u1_ep_{number:04d}",
        entry_id=f"ep_{number:04d}",
        owner_id="u1",
        owner_type="user",
        app_id="test_app",
        project_id="test_project",
        session_id="seekdb",
        timestamp=dt.datetime(1999, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=number),
        parent_id="mc1",
        sender_ids=["u1"],
        subject="red apple" if number % 2 == 0 else "blue banana",
        episode="red apple memory" if number % 2 == 0 else "blue banana memory",
        episode_tokens=(
            "red apple memory" if number % 2 == 0 else "blue banana memory"
        ),
        md_path="test_app/test_project/users/u1/episodes/day.md",
        content_sha256=f"{number:064x}",
        vector=dense if vector else None,
        subject_vector=dense if vector else None,
    )
    values.update(overrides)
    return Episode(**values)  # type: ignore[arg-type]


def _foresight(number: int, *, primary_text: str, evidence_text: str):  # type: ignore[no-untyped-def]
    from everos.infra.persistence.index import Foresight

    dense = [0.0] * 1024
    dense[number] = 1.0
    return Foresight(
        id=f"u1_fs_{number}",
        entry_id=f"fs_{number}",
        owner_id="u1",
        owner_type="user",
        app_id="test_app",
        project_id="test_project",
        session_id="seekdb",
        timestamp=dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=number),
        parent_id="mc1",
        sender_ids=["u1"],
        foresight=primary_text,
        foresight_tokens=primary_text,
        evidence=evidence_text,
        evidence_tokens=evidence_text,
        md_path="test_app/test_project/users/u1/.foresights/day.md",
        content_sha256=f"f{number}",
        vector=dense,
    )


def _agent_skill(number: int):  # type: ignore[no-untyped-def]
    from everos.infra.persistence.index import AgentSkill

    dense = [0.0] * 1024
    dense[number] = 1.0
    return AgentSkill(
        id=f"agent1_skill_{number}",
        owner_id="agent1",
        owner_type="agent",
        app_id="test_app",
        project_id="test_project",
        name=f"skill_{number}",
        description=f"SeekDB skill {number}",
        description_tokens=f"seekdb skill {number}",
        content=f"Reusable procedure {number}",
        content_tokens=f"reusable procedure {number}",
        confidence=0.9,
        maturity_score=0.8,
        source_case_ids=["case1"],
        cluster_id="cluster1",
        md_path=f"test_app/test_project/agents/agent1/skills/skill_{number}/SKILL.md",
        content_sha256=f"s{number}",
        vector=dense,
    )


async def test_seekdb_matches_the_derived_index_contract() -> None:
    from everos.infra.persistence.index import (
        Episode,
        episode_repo,
        eq,
        is_null,
    )

    records = [_episode(number) for number in range(2)]
    await episode_repo.upsert(records)
    assert await episode_repo.count() == 2
    assert (await episode_repo.get_by_id(records[0].id)).timestamp == records[
        0
    ].timestamp  # type: ignore[union-attr]

    await episode_repo.update({"subject": "updated"}, where=eq("id", records[0].id))
    updated = await episode_repo.get_by_id(records[0].id)
    assert updated is not None and updated.subject == "updated"

    sparse = await episode_repo.sparse_search(
        ["apple"], None, columns=Episode.BM25_FIELDS, limit=5
    )
    assert sparse[0]["id"] == records[0].id
    assert sparse[0]["_score"] > 0

    query = [1.0] + [0.0] * 1023
    dense = await episode_repo.dense_search(query, None, limit=5)
    assert dense[0]["id"] == records[0].id
    assert dense[0]["_distance"] == pytest.approx(0.0, abs=1e-5)

    by_subject = await episode_repo.dense_search(
        query, None, limit=5, vector_field="subject_vector"
    )
    assert by_subject[0]["id"] == records[0].id
    assert by_subject[0]["_distance"] == pytest.approx(0.0, abs=1e-5)

    page, total = await episode_repo.find_where_paginated(
        eq("owner_id", "u1"), sort_by="timestamp", page=2, page_size=1
    )
    assert total == 2
    assert len(page) == 1

    await episode_repo.upsert(
        [_episode(number, vector=False) for number in range(2, 103)]
    )
    assert await episode_repo.count_where(is_null("vector")) == 101
    assert len(await episode_repo.scan()) == 103

    assert (
        await episode_repo.delete_by_md_path(
            "test_app/test_project/users/u1/episodes/day.md"
        )
        == 103
    )
    assert await episode_repo.count() == 0


async def test_ids_differing_only_in_case_are_distinct() -> None:
    from everos.infra.persistence.index import episode_repo

    lower = _episode(200, id="case-sensitive-id", entry_id="case-lower")
    upper = _episode(201, id="CASE-SENSITIVE-ID", entry_id="case-upper")
    await episode_repo.add([lower, upper])
    assert await episode_repo.count() == 2
    assert (await episode_repo.get_by_id(lower.id)).id == lower.id  # type: ignore[union-attr]
    assert (await episode_repo.get_by_id(upper.id)).id == upper.id  # type: ignore[union-attr]


@pytest.mark.skipif(not _HOST or bool(_PATH), reason="requires remote SeekDB")
async def test_remote_replacement_restores_session_and_round_trips_text() -> None:
    from everos.infra.persistence.index import episode_repo
    from everos.infra.persistence.seekdb.seekdb_manager import get_session, run

    text = "quote ' backslash \\ newline\n中文"
    before = _episode(300, subject=text, vector=False)
    await episode_repo.upsert([before])
    assert (await episode_repo.get_by_id(before.id)).subject == text  # type: ignore[union-attr]
    session = await get_session()

    def replace_connection() -> None:
        # Reproduce pyseekdb's transparent replacement of a closed connection,
        # with a server sql_mode that would break our literal renderer.
        server = session._server
        server.get_raw_connection().close()
        raw = server.get_raw_connection()
        with raw.cursor() as cursor:
            cursor.execute("SET SESSION sql_mode = 'NO_BACKSLASH_ESCAPES'")

    await run(replace_connection)
    after = _episode(301, subject=text, vector=False)
    await episode_repo.upsert([after])
    assert (await episode_repo.get_by_id(after.id)).subject == text  # type: ignore[union-attr]
    mode = await run(session.fetch_scalar, "SELECT @@SESSION.sql_mode")
    assert "NO_BACKSLASH_ESCAPES" not in mode


async def test_seekdb_runs_specialized_repositories_and_real_filters() -> None:
    from everos.infra.persistence.index import (
        Episode,
        agent_skill_repo,
        episode_repo,
        foresight_repo,
    )
    from everos.memory.search import FilterNode
    from everos.memory.search.filters import compile_filters

    await episode_repo.upsert([_episode(0), _episode(1)])
    where = compile_filters(
        FilterNode.model_validate({"session_id": "seekdb"}),
        owner_id="u1",
        owner_type="user",
        app_id="test_app",
        project_id="test_project",
    )
    assert {row.id for row in await episode_repo.find_where(where, limit=10)} == {
        "u1_ep_0000",
        "u1_ep_0001",
    }

    projected = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=int(dt.datetime(1998, 1, 1, tzinfo=dt.UTC).timestamp()),
        parent_type="memcell",
        app_id="test_app",
        project_id="test_project",
        columns=["id", "subject"],
    )
    assert [row["id"] for row in projected] == ["u1_ep_0000", "u1_ep_0001"]
    assert all(set(row) == {"id", "subject", "timestamp"} for row in projected)

    await foresight_repo.upsert(
        [
            _foresight(0, primary_text="orchard plan", evidence_text="history"),
            _foresight(1, primary_text="travel plan", evidence_text="orchard note"),
        ]
    )
    sparse = await foresight_repo.sparse_search(
        ["orchard"], None, columns=["foresight_tokens", "evidence_tokens"], limit=5
    )
    assert {row["id"] for row in sparse} == {"u1_fs_0", "u1_fs_1"}
    assert all(row["_score"] > 0 for row in sparse)

    skills = [_agent_skill(0), _agent_skill(1)]
    await agent_skill_repo.upsert(skills)
    assert (
        await agent_skill_repo.count_in_cluster(
            owner_id="agent1", cluster_id="cluster1"
        )
        == 2
    )
    assert (
        len(
            await agent_skill_repo.find_in_cluster(
                owner_id="agent1", cluster_id="cluster1", limit=10
            )
        )
        == 2
    )
    query = [1.0] + [0.0] * 1023
    top = await agent_skill_repo.find_topk_relevant_in_cluster(
        owner_id="agent1",
        cluster_id="cluster1",
        query_vector=query,
        top_k=1,
    )
    assert [row.id for row in top] == [skills[0].id]

    # Keep the imported model live so the test also confirms its BM25 metadata.
    assert Episode.BM25_FIELDS == ["episode_tokens"]


async def test_seekdb_schema_drift_is_loud_and_rebuildable() -> None:
    from everos.infra.persistence.index import (
        drop_business_tables,
        ensure_business_indexes,
        verify_business_schemas,
    )
    from everos.infra.persistence.seekdb import episode_repo as concrete_episode_repo
    from everos.infra.persistence.seekdb.errors import SeekdbSchemaMismatchError
    from everos.infra.persistence.seekdb.repository import SeekdbRepoBase

    await concrete_episode_repo._execute(  # type: ignore[attr-defined]
        f"ALTER TABLE `{concrete_episode_repo.physical_table_name}` "
        "DROP COLUMN `subject`"
    )
    SeekdbRepoBase._reset_table_cache()
    with pytest.raises(SeekdbSchemaMismatchError, match="cascade rebuild"):
        await verify_business_schemas()

    await drop_business_tables()
    await ensure_business_indexes()
    SeekdbRepoBase._reset_table_cache()
    await verify_business_schemas()
