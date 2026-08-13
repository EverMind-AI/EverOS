"""Real-LanceDB tests for ``ProfileRecaller`` — KV-by-owner fetch.

Profile recall has no query / no ranking: ``fetch(owner_id)`` returns
the at-most-one row keyed by ``id = owner_id``. These tests exercise
the LanceDB path (no stubs) and the JSON unpacking that turns the
``*_json`` columns back into the DTO's ``profile_data`` mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from everos.core.persistence.lancedb.row_id import user_profile_storage_id
from everos.infra.persistence.lancedb import (
    UserProfile,
    lancedb_manager,
    user_profile_repo,
)
from everos.memory.search.recall.profile import ProfileRecaller


def _profile_row(
    *,
    owner_id: str,
    app_id: str = "default",
    project_id: str = "default",
    summary: str = "summary text",
    explicit_info: list | None = None,
    implicit_traits: list | None = None,
    profile_timestamp_ms: int = 1_700_000_000_000,
) -> UserProfile:
    return UserProfile(
        id=user_profile_storage_id(
            app_id=app_id, project_id=project_id, owner_id=owner_id
        ),
        owner_id=owner_id,
        owner_type="user",
        app_id=app_id,
        project_id=project_id,
        summary=summary,
        explicit_info_json=json.dumps(explicit_info or [], ensure_ascii=False),
        implicit_traits_json=json.dumps(implicit_traits or [], ensure_ascii=False),
        profile_timestamp_ms=profile_timestamp_ms,
        md_path=f"users/{owner_id}/user.md",
        content_sha256="x" * 64,
    )


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


async def test_fetch_returns_dto_when_row_exists() -> None:
    await user_profile_repo.upsert(
        [
            _profile_row(
                owner_id="u_alice",
                summary="Alice likes long hikes.",
                explicit_info=[{"fact": "lives in tokyo"}],
                implicit_traits=[{"trait": "introverted"}],
                profile_timestamp_ms=1_700_000_001_000,
            )
        ]
    )

    items = await ProfileRecaller().fetch(
        "u_alice", app_id="default", project_id="default"
    )
    assert len(items) == 1
    item = items[0]
    assert item.id == "u_alice"
    assert item.user_id == "u_alice"
    assert item.score is None
    # JSON columns are decoded back to live Python on the way out.
    assert item.profile_data["summary"] == "Alice likes long hikes."
    assert item.profile_data["explicit_info"] == [{"fact": "lives in tokyo"}]
    assert item.profile_data["implicit_traits"] == [{"trait": "introverted"}]
    assert item.profile_data["profile_timestamp_ms"] == 1_700_000_001_000


async def test_fetch_returns_empty_when_row_missing() -> None:
    items = await ProfileRecaller().fetch(
        "u_cold_start", app_id="default", project_id="default"
    )
    assert items == []


async def test_fetch_returns_empty_for_blank_owner() -> None:
    """Blank ``owner_id`` short-circuits — never hit LanceDB with an
    empty-string PK (which would otherwise return any row whose id was
    persisted as the empty string)."""
    items = await ProfileRecaller().fetch("", app_id="default", project_id="default")
    assert items == []


async def test_fetch_isolates_by_owner() -> None:
    await user_profile_repo.upsert(
        [
            _profile_row(owner_id="u_alice", summary="Alice"),
            _profile_row(owner_id="u_bob", summary="Bob"),
        ]
    )
    bob_items = await ProfileRecaller().fetch(
        "u_bob", app_id="default", project_id="default"
    )
    assert len(bob_items) == 1
    assert bob_items[0].profile_data["summary"] == "Bob"


async def test_fetch_tolerates_malformed_json_columns() -> None:
    """A column with corrupted JSON should not blow up the recall path —
    the bucket falls back to ``[]`` and the rest of the DTO survives."""
    await user_profile_repo.upsert(
        [
            UserProfile(
                id=user_profile_storage_id(
                    app_id="default", project_id="default", owner_id="u_broken"
                ),
                owner_id="u_broken",
                owner_type="user",
                summary="ok",
                explicit_info_json="{not valid json",
                implicit_traits_json="[]",
                profile_timestamp_ms=0,
                md_path="users/u_broken/user.md",
                content_sha256="y" * 64,
            )
        ]
    )

    items = await ProfileRecaller().fetch(
        "u_broken", app_id="default", project_id="default"
    )
    assert len(items) == 1
    assert items[0].profile_data["explicit_info"] == []
    assert items[0].profile_data["implicit_traits"] == []
    assert items[0].profile_data["summary"] == "ok"


async def test_fetch_isolates_same_owner_across_projects() -> None:
    await user_profile_repo.upsert(
        [
            _profile_row(
                owner_id="u_same",
                app_id="app_a",
                project_id="project_a",
                summary="Project A",
            ),
            _profile_row(
                owner_id="u_same",
                app_id="app_a",
                project_id="project_b",
                summary="Project B",
            ),
        ]
    )

    items = await ProfileRecaller().fetch(
        "u_same", app_id="app_a", project_id="project_b"
    )

    assert len(items) == 1
    assert items[0].id == "u_same"
    assert items[0].project_id == "project_b"
    assert items[0].profile_data["summary"] == "Project B"
