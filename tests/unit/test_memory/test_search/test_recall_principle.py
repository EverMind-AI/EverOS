"""Real-LanceDB tests for ``PrincipleRecaller`` — KV-by-owner fetch.

Principle recall has no query / no ranking: ``fetch(owner_id, app_id,
project_id)`` returns every row for that scope. These tests exercise
the LanceDB path (no stubs) and the owner / app / project isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.persistence.lancedb import (
    Principle,
    lancedb_manager,
    principle_repo,
)
from everos.memory.search.recall.principle import PrincipleRecaller


def _principle_row(
    *,
    owner_id: str,
    principle_id: str,
    title: str = "Prefer Rust",
    statement: str = "Device Runtime is implemented in Rust.",
    source_entry_ids: list[str] | None = None,
    timestamp_ms: int = 1_700_000_000_000,
    app_id: str = "default",
    project_id: str = "default",
) -> Principle:
    return Principle(
        id=f"{owner_id}_{principle_id}",
        principle_id=principle_id,
        owner_id=owner_id,
        owner_type="user",
        app_id=app_id,
        project_id=project_id,
        title=title,
        statement=statement,
        source_entry_ids=source_entry_ids or ["dc_20260101_0001"],
        timestamp_ms=timestamp_ms,
        md_path=f"users/{owner_id}/principles.md",
        content_sha256="x" * 64,
    )


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


async def test_fetch_returns_all_rows_for_owner() -> None:
    await principle_repo.upsert(
        [
            _principle_row(
                owner_id="u_alice",
                principle_id="pr_aaaaaaaaaaaa",
                title="Prefer Rust",
            ),
            _principle_row(
                owner_id="u_alice",
                principle_id="pr_bbbbbbbbbbbb",
                title="Keep Go control plane",
                statement="Leave the control plane in Go.",
                source_entry_ids=["dc_20260101_0002"],
            ),
        ]
    )

    items = await PrincipleRecaller().fetch("u_alice")
    titles = {item.title for item in items}
    assert titles == {"Prefer Rust", "Keep Go control plane"}
    rust = next(i for i in items if i.title == "Prefer Rust")
    assert rust.id == "u_alice_pr_aaaaaaaaaaaa"
    assert rust.user_id == "u_alice"
    assert rust.score is None
    assert rust.source_entry_ids == ["dc_20260101_0001"]
    assert rust.statement == "Device Runtime is implemented in Rust."


async def test_fetch_returns_empty_when_row_missing() -> None:
    items = await PrincipleRecaller().fetch("u_cold_start")
    assert items == []


async def test_fetch_returns_empty_for_blank_owner() -> None:
    items = await PrincipleRecaller().fetch("")
    assert items == []


async def test_fetch_isolates_by_owner() -> None:
    await principle_repo.upsert(
        [
            _principle_row(
                owner_id="u_alice",
                principle_id="pr_aaaaaaaaaaaa",
                title="Alice",
            ),
            _principle_row(
                owner_id="u_bob",
                principle_id="pr_bbbbbbbbbbbb",
                title="Bob",
            ),
        ]
    )
    bob_items = await PrincipleRecaller().fetch("u_bob")
    assert len(bob_items) == 1
    assert bob_items[0].title == "Bob"


async def test_fetch_isolates_by_app_project() -> None:
    await principle_repo.upsert(
        [
            _principle_row(
                owner_id="u_alice",
                principle_id="pr_aaaaaaaaaaaa",
                title="Default space",
            ),
            _principle_row(
                owner_id="u_alice",
                principle_id="pr_cccccccccccc",
                title="Other app",
                app_id="other",
            ),
        ]
    )
    items = await PrincipleRecaller().fetch(
        "u_alice", app_id="default", project_id="default"
    )
    assert [i.title for i in items] == ["Default space"]
    other = await PrincipleRecaller().fetch(
        "u_alice", app_id="other", project_id="default"
    )
    assert [i.title for i in other] == ["Other app"]
