"""Unit coverage for the remote Milvus derived-index adapter."""

from __future__ import annotations

import datetime as dt

import pytest

from everos.config import load_settings
from everos.infra.persistence.index import (
    Episode,
    IndexRepository,
    episode_repo,
    is_null,
    user_profile_repo,
)
from everos.infra.persistence.index.schema import schema_for
from everos.infra.persistence.milvus import repository
from everos.infra.persistence.milvus.milvus_manager import (
    MilvusConfigurationError,
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
            return {
                "fields": [
                    {"name": field} for field in milvus_repo._stored_field_names()
                ]
            }

    client = _FakeClient()

    async def _fake_get_client():  # type: ignore[no-untyped-def]
        return client

    monkeypatch.setattr(repository, "get_client", _fake_get_client)
    await milvus_repo.ensure_collection()
    await milvus_repo.ensure_collection()
    assert client.has_calls == 1
    assert client.describe_calls == 1
