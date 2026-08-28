"""Unit coverage for the remote Milvus derived-index adapter."""

from __future__ import annotations

import datetime as dt

import pytest
from pymilvus import DataType

from everos.config import load_settings
from everos.infra.persistence.index import (
    Episode,
    IndexRepository,
    episode_repo,
    foresight_repo,
    is_null,
    user_profile_repo,
)
from everos.infra.persistence.index.schema import schema_for
from everos.infra.persistence.milvus import repository
from everos.infra.persistence.milvus.milvus_manager import (
    MilvusConfigurationError,
    MilvusSchemaMismatchError,
    _resolve_uri,
)
from everos.infra.persistence.milvus.predicate import render_predicate
from everos.infra.persistence.milvus.repos import ALL_REPOS
from everos.infra.persistence.milvus.repository import (
    MilvusRepoBase,
    MilvusValueLimitError,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVEROS_INDEX__BACKEND", "milvus")
    monkeypatch.setenv("EVEROS_MILVUS__URI", "http://milvus.example:19530")
    monkeypatch.setenv("EVEROS_MILVUS__COLLECTION_PREFIX", "unit_test")
    load_settings.cache_clear()
    MilvusRepoBase._reset_locks_for_tests()
    yield
    MilvusRepoBase._reset_locks_for_tests()
    load_settings.cache_clear()


def _describe_response(milvus_repo, *, always_params: bool = False):  # type: ignore[no-untyped-def]
    """What Milvus reports for the collection this repo would create.

    Built from pymilvus' own ``CollectionSchema.to_dict()`` rather than a
    hand-written literal, so the field shape comes from the library instead of
    from our assumptions. ``describe_collection`` always emits ``params``
    while ``to_dict`` omits it when empty — ``always_params`` covers that one
    known difference between the two serializations.
    """
    fields = [
        dict(f) for f in milvus_repo._build_collection_schema().to_dict()["fields"]
    ]
    if always_params:
        for field in fields:
            field.setdefault("params", {})
    return {"fields": fields}


def _patch_describe(monkeypatch, response):  # type: ignore[no-untyped-def]
    class _FakeClient:
        def describe_collection(self, name: str):  # type: ignore[no-untyped-def]
            return response

    async def _fake_get_client():  # type: ignore[no-untyped-def]
        return _FakeClient()

    monkeypatch.setattr(repository, "get_client", _fake_get_client)


def _episode(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "id": "u1_ep1",
        "entry_id": "ep1",
        "owner_id": "u1",
        "owner_type": "user",
        "session_id": "abc=",
        "timestamp": dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        "parent_id": "mc1",
        "sender_ids": ["user"],
        "episode": "red apple memory",
        "episode_tokens": "red apple memory",
        "md_path": "default_app/default_project/users/u1/episodes/day.md",
        "content_sha256": "a",
        "vector": [1.0] + [0.0] * 1023,
        "subject_vector": [0.0, 1.0] + [0.0] * 1022,
    }
    values.update(overrides)
    return Episode(**values)


def test_remote_uri_is_required_and_local_paths_are_rejected() -> None:
    settings = load_settings().milvus.model_copy(update={"uri": ""})
    with pytest.raises(MilvusConfigurationError, match="requires"):
        _resolve_uri(settings)

    settings = settings.model_copy(update={"uri": "/tmp/milvus.db"})
    with pytest.raises(MilvusConfigurationError, match="remote http"):
        _resolve_uri(settings)

    settings = settings.model_copy(update={"uri": "file:///tmp/milvus.db"})
    with pytest.raises(MilvusConfigurationError, match=r"http\(s\)"):
        _resolve_uri(settings)


def test_neutral_schema_tracks_every_model_field_and_dense_vector() -> None:
    schema = schema_for(Episode)
    assert {field.name for field in schema.fields} == set(Episode.model_fields)
    assert [field.name for field in schema.vector_fields] == [
        "vector",
        "subject_vector",
    ]
    assert all(isinstance(repo, IndexRepository) for repo in ALL_REPOS)


def test_null_vector_predicate_targets_presence_marker() -> None:
    assert (
        render_predicate(is_null("vector"), vector_fields={"vector"})
        == "vector__present == false"
    )


def test_record_conversion_stores_every_dense_vector_with_presence() -> None:
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    record = milvus_repo._to_milvus_record(_episode())
    assert len(record["vector"]) == 1024
    assert len(record["subject_vector"]) == 1024
    assert record["vector__present"] is True
    assert record["subject_vector__present"] is True

    missing = milvus_repo._to_milvus_record(_episode(subject_vector=None))
    assert missing["subject_vector__present"] is False
    assert missing["subject_vector"] == [0.0] * 1024


def test_datetime_round_trip_is_exact_below_the_ms_heuristic() -> None:
    """Epoch ms are written unconditionally, so they must be read the same way.

    ``from_timestamp`` treats anything under 1e12 as *seconds*, so a
    pre-2001-09-09 instant stored as ms comes back in the year 30000 — or
    raises ``ValueError: year ... is out of range`` on the way out. The
    physical datetime column has to bypass that heuristic entirely.
    """
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    for moment in (
        dt.datetime(1970, 1, 2, tzinfo=dt.UTC),
        dt.datetime(1999, 1, 1, tzinfo=dt.UTC),
        dt.datetime(2001, 9, 8, tzinfo=dt.UTC),
        dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
    ):
        stored = milvus_repo._to_milvus_record(_episode(timestamp=moment))
        assert "timestamp_ms" in stored
        restored = milvus_repo._restore_row(stored)
        assert restored["timestamp"] == moment, f"round-trip broke at {moment}"


def test_update_fetch_preserves_dummy_vector_for_scalar_only_tables() -> None:
    milvus_repo = user_profile_repo._repo()  # type: ignore[attr-defined]
    assert "_everos_dummy_vector" in milvus_repo._output_fields(include_vectors=True)
    assert "_everos_dummy_vector" not in milvus_repo._output_fields(
        include_vectors=False
    )


def test_record_conversion_reports_varchar_array_and_vector_limits() -> None:
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    with pytest.raises(MilvusValueLimitError, match=r"episode is .* UTF-8 bytes"):
        milvus_repo._to_milvus_record(_episode(episode="x" * 65_536))
    with pytest.raises(MilvusValueLimitError, match="sender_ids has 257 items"):
        milvus_repo._to_milvus_record(_episode(sender_ids=["u"] * 257))
    with pytest.raises(MilvusValueLimitError, match="dimension 2"):
        milvus_repo._validate_vector(
            milvus_repo.index_schema.field("vector"),
            [1.0, 0.0],
        )


def test_server_score_normalization() -> None:
    assert repository._cosine_distance_from_milvus(0.75) == pytest.approx(0.25)
    assert repository._bm25_score_from_distance(1.5) == pytest.approx(1.5)


async def test_collection_metadata_is_cached_after_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]

    class _FakeClient:
        has_calls = 0
        describe_calls = 0

        def has_collection(self, name: str) -> bool:
            self.has_calls += 1
            return True

        def describe_collection(self, name: str):  # type: ignore[no-untyped-def]
            self.describe_calls += 1
            return _describe_response(milvus_repo)

    client = _FakeClient()

    async def _fake_get_client():  # type: ignore[no-untyped-def]
        return client

    monkeypatch.setattr(repository, "get_client", _fake_get_client)
    await milvus_repo.ensure_collection()
    await milvus_repo.ensure_collection()
    assert client.has_calls == 1
    assert client.describe_calls == 1


# ── Physical schema verification ────────────────────────────────────────


@pytest.mark.parametrize("always_params", [False, True])
async def test_verify_accepts_every_collection_it_would_create(
    monkeypatch: pytest.MonkeyPatch, always_params: bool
) -> None:
    """The false-positive guard: our own collections must always pass.

    A checker that rejects a healthy deployment is far worse than the
    name-only check it replaces, so every table is verified against the exact
    schema this adapter would have created for it.
    """
    for repo in ALL_REPOS:
        _patch_describe(
            monkeypatch, _describe_response(repo, always_params=always_params)
        )
        await repo.verify_collection()


async def test_verify_rejects_datatype_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same field name, wrong physical type — the case a name check waves through."""
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    response = _describe_response(milvus_repo)
    for field in response["fields"]:
        if field["name"] == "timestamp_ms":
            field["type"] = DataType.VARCHAR
    _patch_describe(monkeypatch, response)

    with pytest.raises(MilvusSchemaMismatchError, match="timestamp_ms: datatype"):
        await milvus_repo.verify_collection()


async def test_verify_rejects_vector_dimension_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    response = _describe_response(milvus_repo)
    for field in response["fields"]:
        if field["name"] == "vector":
            field["params"] = {"dim": 1}
    _patch_describe(monkeypatch, response)

    with pytest.raises(
        MilvusSchemaMismatchError, match=r"vector: dim 1 != expected 1024"
    ):
        await milvus_repo.verify_collection()


async def test_verify_rejects_primary_and_nullable_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``describe_collection`` omits both flags when false, so absence is exact."""
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    response = _describe_response(milvus_repo)
    for field in response["fields"]:
        if field["name"] == "id":
            field.pop("is_primary", None)
        if field["name"] == "deprecated_by":
            field.pop("nullable", None)
    _patch_describe(monkeypatch, response)

    with pytest.raises(MilvusSchemaMismatchError) as excinfo:
        await milvus_repo.verify_collection()
    assert "id: is_primary" in str(excinfo.value)
    assert "deprecated_by: nullable" in str(excinfo.value)


async def test_verify_rejects_missing_and_stale_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    response = _describe_response(milvus_repo)
    response["fields"] = [f for f in response["fields"] if f["name"] != "subject"]
    response["fields"].append({"name": "left_over", "type": DataType.VARCHAR})
    _patch_describe(monkeypatch, response)

    with pytest.raises(MilvusSchemaMismatchError) as excinfo:
        await milvus_repo.verify_collection()
    assert "'subject'" in str(excinfo.value)
    assert "'left_over'" in str(excinfo.value)


async def test_verify_treats_limit_drift_as_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong VARCHAR length fails loudly on write, so it must not block startup.

    Blocking here would turn a server-side normalization we cannot predict into
    an outage; the datatype and dimension checks carry the load instead.
    """
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    response = _describe_response(milvus_repo)
    for field in response["fields"]:
        if field["name"] == "subject":
            field["params"] = {"max_length": 4096}
    _patch_describe(monkeypatch, response)

    warnings: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        repository.logger,
        "warning",
        lambda event, **kw: warnings.append((event, kw)),
    )

    await milvus_repo.verify_collection()

    # Tolerated, but never silently: the drift has to reach the operator.
    assert [event for event, _ in warnings] == ["milvus_collection_limit_drift"]
    assert warnings[0][1]["details"] == ["subject: max_length 4096 != declared 65535"]


# Written out by hand on purpose. _describe_response() builds its fake server
# reply from _build_collection_schema(), which shares _physical_fields() with
# the verifier — so those tests can only prove that creation and verification
# agree with *each other*, never that either matches what we intend. This
# literal is the independent statement of intent: a slip in _physical_for
# (a nullable flipped, a dim dropped, an analyzer lost) changes both sides at
# once and is invisible everywhere except here.
#
# Only a run against a real Milvus can prove the declaration matches what the
# server actually stores; see tests/integration/test_milvus_remote.py.
_EPISODE_PHYSICAL = (
    ("created_at_ms", DataType.INT64),
    ("updated_at_ms", DataType.INT64),
    ("id", DataType.VARCHAR, "is_primary", 512),
    ("entry_id", DataType.VARCHAR, "", 65535),
    ("owner_id", DataType.VARCHAR, "", 65535),
    ("owner_type", DataType.VARCHAR, "", 65535),
    ("app_id", DataType.VARCHAR, "", 65535),
    ("project_id", DataType.VARCHAR, "", 65535),
    ("session_id", DataType.VARCHAR, "nullable", 65535),
    ("timestamp_ms", DataType.INT64),
    ("parent_type", DataType.VARCHAR, "", 65535),
    ("parent_id", DataType.VARCHAR, "", 65535),
    ("sender_ids", DataType.ARRAY, "", 512),
    ("subject", DataType.VARCHAR, "nullable", 65535),
    ("summary", DataType.VARCHAR, "nullable", 65535),
    ("episode", DataType.VARCHAR, "", 65535),
    ("episode_tokens", DataType.VARCHAR, "enable_analyzer", 65535),
    ("md_path", DataType.VARCHAR, "", 65535),
    ("content_sha256", DataType.VARCHAR, "", 65535),
    ("deprecated_by", DataType.VARCHAR, "nullable", 65535),
    ("vector", DataType.FLOAT_VECTOR, "", None),
    ("vector__present", DataType.BOOL),
    ("subject_vector", DataType.FLOAT_VECTOR, "", None),
    ("subject_vector__present", DataType.BOOL),
    ("episode_tokens__sparse", DataType.SPARSE_FLOAT_VECTOR),
)


def _flag(field) -> str:  # type: ignore[no-untyped-def]
    for candidate in ("is_primary", "nullable", "enable_analyzer"):
        if getattr(field, candidate):
            return candidate
    return ""


def test_episode_physical_layout_matches_the_declared_snapshot() -> None:
    """Pin the physical column list against a hand-written expectation."""
    milvus_repo = episode_repo._repo()  # type: ignore[attr-defined]
    actual = milvus_repo._physical_fields()

    assert [f.name for f in actual] == [row[0] for row in _EPISODE_PHYSICAL]
    for field, row in zip(actual, _EPISODE_PHYSICAL, strict=True):
        assert field.datatype == row[1], field.name
        if len(row) == 2:
            assert not _flag(field), field.name
            continue
        assert _flag(field) == row[2], field.name
        assert field.max_length == row[3], field.name

    sender_ids = next(f for f in actual if f.name == "sender_ids")
    assert sender_ids.element_type is DataType.VARCHAR
    assert sender_ids.max_capacity == 256
    for name in ("vector", "subject_vector"):
        assert next(f for f in actual if f.name == name).dim == 1024


def test_vectorless_table_declares_the_dummy_vector_column() -> None:
    """Milvus needs at least one vector column, so scalar-only tables fake one."""
    milvus_repo = user_profile_repo._repo()  # type: ignore[attr-defined]
    actual = {f.name: f for f in milvus_repo._physical_fields()}

    assert "_everos_dummy_vector" in actual
    assert actual["_everos_dummy_vector"].datatype is DataType.FLOAT_VECTOR
    assert actual["_everos_dummy_vector"].dim == 2
    assert not [
        f for f in actual.values() if f.datatype is DataType.SPARSE_FLOAT_VECTOR
    ]
    assert actual["id"].is_primary is True
    assert actual["summary"].nullable is False


def test_nullable_bm25_input_is_declared_not_null() -> None:
    """``foresight.evidence_tokens`` is the only nullable BM25 column.

    A BM25 input feeds an analyzer, so it is stored as "" rather than null —
    which means the physical column must be declared NOT NULL even though the
    logical field is optional. Episode and user_profile cannot cover this
    interaction, so it is asserted where it actually occurs.
    """
    logical = schema_for(foresight_repo.schema)
    assert logical.field("evidence_tokens").nullable is True
    assert "evidence_tokens" in logical.bm25_fields

    physical = {f.name: f for f in foresight_repo._repo()._physical_fields()}  # type: ignore[attr-defined]
    assert physical["evidence_tokens"].nullable is False
    assert physical["evidence_tokens"].enable_analyzer is True
    # "evidence" is the same optional text without the analyzer attached, so
    # it stays nullable — the NOT NULL above is caused by BM25, nothing else.
    assert physical["evidence"].nullable is True
