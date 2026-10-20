"""Cross-backend behavior against Milvus Server or Zilliz Cloud.

Set ``EVEROS_TEST_MILVUS_URI`` and, when required,
``EVEROS_TEST_MILVUS_TOKEN``. The same test is used for self-hosted and cloud
endpoints and creates uniquely prefixed, disposable collections.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
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
    prefix = os.environ.get(
        "EVEROS_TEST_MILVUS_PREFIX", f"everos_e2e_{uuid.uuid4().hex}"
    )
    assert re.fullmatch(r"everos_e2e_[0-9a-f]{32}", prefix)
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
    print(f"EVEROS_E2E_PREFIX={prefix}")

    from everos.core.observability.logging import configure_logging
    from everos.infra.persistence.index import episode_repo, startup

    configure_logging(level="WARNING")
    try:
        if os.environ.get("EVEROS_TEST_MILVUS_FULL_STARTUP") == "1":
            await startup()
        else:
            await episode_repo._repo().ensure_collection()  # type: ignore[attr-defined]
        yield
    finally:
        from everos.infra.persistence.index import drop_business_tables, shutdown

        try:
            await drop_business_tables()
        finally:
            try:
                await shutdown()
            finally:
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
        primary_key,
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
        updated_profile = await user_profile_repo.find_one_where(
            primary_key("test_app", "test_project", "u1")
        )
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


async def test_episode_only_probe_covers_metadata_search_and_http() -> None:
    """One collection covers metadata, dense/BM25 search, and HTTP ``/get``."""
    from importlib import import_module

    from httpx import ASGITransport, AsyncClient

    from everos.entrypoints.api.app import create_app
    from everos.infra.persistence.index import Episode, episode_repo
    from everos.infra.persistence.milvus.milvus_manager import get_client
    from everos.memory.search import FilterNode
    from everos.memory.search.filters import compile_filters

    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    await milvus_repo.verify_collection()

    client = await get_client()
    description = await asyncio.to_thread(
        client.describe_collection, milvus_repo.collection_name
    )
    analyzers = [
        {
            "field_name": field["name"],
            "enable_analyzer": str(
                (field.get("params") or {}).get("enable_analyzer")
            ).casefold()
            == "true",
        }
        for field in description.get("fields", [])
        if field.get("name") == "episode_tokens"
    ]
    functions = [
        {
            "type": getattr(function.get("type"), "name", str(function.get("type"))),
            "input_field_names": list(function.get("input_field_names") or []),
            "output_field_names": list(function.get("output_field_names") or []),
        }
        for function in description.get("functions", [])
    ]
    indexes = []
    for index_name in await asyncio.to_thread(
        client.list_indexes, milvus_repo.collection_name
    ):
        index = await asyncio.to_thread(
            client.describe_index, milvus_repo.collection_name, index_name
        )
        if not isinstance(index, dict):
            continue
        field_name = index.get("field_name")
        if field_name not in {"vector", "subject_vector", "episode_tokens__sparse"}:
            continue
        metric = index.get("metric_type")
        if metric is None and isinstance(index.get("params"), dict):
            metric = index["params"].get("metric_type")
        indexes.append({"field_name": field_name, "metric_type": str(metric).upper()})
    indexes.sort(key=lambda item: str(item["field_name"]))

    assert analyzers == [{"field_name": "episode_tokens", "enable_analyzer": True}]
    assert functions == [
        {
            "type": "BM25",
            "input_field_names": ["episode_tokens"],
            "output_field_names": ["episode_tokens__sparse"],
        }
    ]
    assert indexes == [
        {"field_name": "episode_tokens__sparse", "metric_type": "BM25"},
        {"field_name": "subject_vector", "metric_type": "COSINE"},
        {"field_name": "vector", "metric_type": "COSINE"},
    ]

    now = dt.datetime(2026, 6, 1, tzinfo=dt.UTC)
    await episode_repo.upsert(
        [
            _episode(
                row_id="probe_ep1",
                entry_id="probe_ep1",
                session_id="probe",
                text="red apple probe",
                vector_axis=0,
                subject_axis=1,
                timestamp=now,
            ),
            _episode(
                row_id="probe_ep2",
                entry_id="probe_ep2",
                session_id="probe",
                text="blue banana probe",
                vector_axis=1,
                subject_axis=0,
                timestamp=now + dt.timedelta(seconds=1),
            ),
        ]
    )
    where = compile_filters(
        FilterNode.model_validate({"session_id": "probe"}),
        owner_id="u1",
        owner_type="user",
        app_id="test_app",
        project_id="test_project",
    )
    dense_query = [1.0] + [0.0] * 1023
    dense = await episode_repo.dense_search(dense_query, where, limit=2)
    sparse = await episode_repo.sparse_search(
        ["apple"], where, columns=Episode.BM25_FIELDS, limit=2
    )
    assert dense[0]["id"] == "probe_ep1"
    assert dense[0]["_distance"] == pytest.approx(0.0, abs=1e-5)
    assert sparse[0]["id"] == "probe_ep1"
    assert sparse[0]["_score"] > 0

    get_service = import_module("everos.service.get")
    get_service._manager = None
    app = create_app(lifespan_providers=[])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post(
            "/api/v1/memory/get",
            json={
                "user_id": "u1",
                "memory_type": "episode",
                "app_id": "test_app",
                "project_id": "test_project",
                "filters": {"session_id": "probe"},
                "page": 1,
                "page_size": 2,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["count"] == 2
    assert body["data"]["total_count"] == 2

    evidence = {
        "analyzers": analyzers,
        "functions": functions,
        "indexes": indexes,
        "dense_cosine": "passed",
        "bm25": "passed",
        "http_get": {"status": response.status_code, "count": 2, "total": 2},
    }
    print(f"EVEROS_E2E_EVIDENCE={json.dumps(evidence, sort_keys=True)}")


async def test_update_preserves_vectors_on_a_row_that_has_them() -> None:
    """Milvus has no partial-column update, so update() re-upserts whole rows.

    The only prior coverage ran against ``user_profile``, which carries no
    vector column — so the read-back-and-re-upsert of a 1024-d vector (the
    path backfill and reflection both take) was never exercised remotely.
    """
    from everos.infra.persistence.index import episode_repo, eq, primary_key

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

    stored = await episode_repo.find_one_where(
        primary_key("test_app", "test_project", "u1_vec_update")
    )
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
    from everos.infra.persistence.index import episode_repo, eq, primary_key

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

    stored = await episode_repo.find_one_where(
        primary_key("test_app", "test_project", "u1_pre2001")
    )
    assert stored is not None
    assert stored.timestamp == old

    assert await episode_repo.count_where(eq("timestamp", old)) == 1

    await episode_repo.delete_by_md_path(
        "test_app/test_project/users/u1/episodes/day.md"
    )


async def test_verify_accepts_collection_schema_functions_and_indexes() -> None:
    """The decisive check for physical metadata verification.

    Every offline test builds its fake ``describe_collection`` reply from the
    same descriptor the verifier compares against, so they can only prove that
    creation and verification agree with each other. Whether that descriptor
    matches what Milvus actually stores and reports back — VARCHAR lengths it
    may normalize, flags it may omit, the analyzer it attaches for BM25 — can
    only be established against a real server.

    The fixture creates the collections with a fresh prefix, so verification
    never runs during setup; dropping the process-local readiness cache forces
    the existing-collection path.
    """
    from everos.infra.persistence.index import ALL_REPOS, episode_repo
    from everos.infra.persistence.milvus.repository import MilvusRepoBase

    await episode_repo.upsert(
        [
            _episode(
                row_id="u1_verify",
                entry_id="verify",
                session_id="verify",
                text="a row so the collection is not empty",
                vector_axis=1,
                subject_axis=2,
                timestamp=dt.datetime(2026, 5, 1, tzinfo=dt.UTC),
            )
        ]
    )

    for repo in ALL_REPOS:
        milvus_repo = repo._repo()  # type: ignore[attr-defined]
        await milvus_repo.ensure_collection()

    MilvusRepoBase._reset_collection_cache()

    for repo in ALL_REPOS:
        # Raises MilvusSchemaMismatchError if fields, BM25 functions, or
        # vector index metrics disagree with our declaration.
        await repo._repo().verify_collection()  # type: ignore[attr-defined]

    await episode_repo.delete_by_md_path(
        "test_app/test_project/users/u1/episodes/day.md"
    )


async def test_cross_project_rows_share_an_id() -> None:
    """A12 / A1 on Milvus: the same ``id`` in two projects is two rows."""
    from everos.infra.persistence.index import episode_repo, eq, primary_key

    now = dt.datetime(2026, 9, 14, 10, 0, tzinfo=dt.UTC)
    alpha = _episode(
        row_id="u1_ep_1",
        entry_id="ep_1",
        session_id="s",
        text="rust",
        vector_axis=1,
        subject_axis=2,
        timestamp=now,
    )
    beta = alpha.model_copy(
        update={
            "project_id": "beta",
            "episode": "go",
            "md_path": "test_app/beta/users/u1/episodes/day.md",
        }
    )
    await episode_repo.upsert([alpha])
    await episode_repo.upsert([beta])
    assert await episode_repo.count_where(eq("owner_id", "u1")) == 2
    got_alpha = await episode_repo.find_one_where(
        primary_key("test_app", "test_project", "u1_ep_1")
    )
    got_beta = await episode_repo.find_one_where(
        primary_key("test_app", "beta", "u1_ep_1")
    )
    assert got_alpha is not None and got_alpha.episode == "rust"
    assert got_beta is not None and got_beta.episode == "go"
    # same key again is an update, not a third row
    await episode_repo.upsert([beta.model_copy(update={"episode": "go v2"})])
    assert await episode_repo.count_where(eq("owner_id", "u1")) == 2


async def test_find_topk_relevant_in_cluster_hydrates_own_scope() -> None:
    """A24: the by-id hydration after dense search stays inside the caller's space."""
    from everos.infra.persistence.index import (
        AgentSkill,
        agent_skill_repo,
        eq,
        primary_key,
    )

    def skill(project: str, description: str) -> AgentSkill:
        vector = [0.0] * 1024
        vector[3] = 1.0
        return AgentSkill(
            id="agent_run_the_linter",
            owner_id="agent",
            owner_type="agent",
            app_id="test_app",
            project_id=project,
            name="run_the_linter",
            description=description,
            description_tokens=description,
            content=description,
            content_tokens=description,
            confidence=0.9,
            maturity_score=1.0,
            source_case_ids=[],
            cluster_id="cl_1",
            md_path=(
                f"test_app/{project}/agents/agent/skills/skill_run_the_linter/SKILL.md"
            ),
            content_sha256=project,
            vector=vector,
        )

    repo = agent_skill_repo._repo()  # type: ignore[attr-defined]
    await repo.ensure_collection()
    await agent_skill_repo.upsert(
        [skill("alpha", "alpha flavour"), skill("beta", "beta flavour")]
    )
    assert await agent_skill_repo.count_where(eq("owner_id", "agent")) == 2
    query = [0.0] * 1024
    query[3] = 1.0
    hits = await repo.find_topk_relevant_in_cluster(
        owner_id="agent", cluster_id="cl_1", query_vector=query, top_k=5
    )
    assert {h.project_id for h in hits} == {"alpha", "beta"}
    for hit in hits:
        same = await agent_skill_repo.find_one_where(
            primary_key("test_app", hit.project_id, hit.id)
        )
        assert same is not None and same.description == hit.description


async def test_embedding_model_attestation_round_trips_on_the_collection() -> None:
    """The model that built an index is recorded with the index, not beside it.

    A local marker file would describe a remote cluster from one machine's
    disk: several machines pointing at the same Milvus would each keep their
    own answer and none of them would be wrong about their own file. Keeping
    it as a collection property means there is one answer, and dropping the
    collection takes it along.

    Read back through a *new* client so the assertion cannot be satisfied by
    something the writing client is holding in memory.
    """
    from everos.infra.persistence import milvus
    from everos.infra.persistence.milvus import milvus_manager

    backend = milvus.index_backend
    assert await backend.read_embedding_model() is None, (
        "a collection nobody attested must read as unattested, not as an error"
    )

    await backend.write_embedding_model("fake-768")
    await milvus_manager.dispose_connection()
    assert await backend.read_embedding_model() == "fake-768"

    await backend.write_embedding_model("fake-1536")
    await milvus_manager.dispose_connection()
    assert await backend.read_embedding_model() == "fake-1536", (
        "re-attesting must replace the previous value, not append to it"
    )


async def test_changed_model_is_refused_on_the_milvus_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is backend-neutral: same check, same error, other store."""
    import everos.component.embedding.accessor as embedding_accessor
    from everos.component.embedding import EmbeddingCapability
    from everos.core.errors import EmbeddingProvenanceError
    from everos.infra.persistence import milvus
    from everos.infra.persistence.index import _provenance

    await milvus.index_backend.write_embedding_model("nomic-embed-text")

    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "key")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://127.0.0.1:1/v1")
    load_settings.cache_clear()

    # Stand in for the capability and the probe: this case is about the model
    # comparison, and neither a real provider nor a reachable service is
    # needed to decide that two names differ.
    class _Provider:
        dim = 1024

        async def embed(self, text: str) -> list[float]:  # pragma: no cover
            raise AssertionError("the probe is stubbed here")

        async def embed_query(self, text: str) -> list[float]:  # pragma: no cover
            raise AssertionError("the probe is stubbed here")

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("the probe is stubbed here")

    monkeypatch.setattr(
        embedding_accessor, "_capability", EmbeddingCapability(provider=_Provider())
    )

    async def _probe() -> int | None:
        return 1024

    monkeypatch.setattr(_provenance, "probe_embedding_dimension", _probe)

    with pytest.raises(EmbeddingProvenanceError) as excinfo:
        await _provenance.verify_embedding_provenance(milvus.index_backend)
    message = str(excinfo.value)
    assert "nomic-embed-text" in message
    assert "Qwen/Qwen3-Embedding-4B" in message
    assert "cascade rebuild" in message

    assert await milvus.index_backend.read_embedding_model() == "nomic-embed-text", (
        "a refusal must not quietly adopt the new model"
    )
