"""Optional empty metadata survives the writer -> parser -> real index path."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from everos.component.tokenizer import build_tokenizer
from everos.component.utils.datetime import from_iso_format
from everos.config import LanceDBSettings
from everos.core.persistence import LanceRepoBase, MemoryRoot
from everos.core.persistence.lancedb import open_lancedb_connection
from everos.infra.persistence.index import Episode, eq
from everos.infra.persistence.lancedb import LanceIndexContract
from everos.infra.persistence.markdown import EpisodeWriter
from everos.memory.cascade.handlers import HandlerDeps
from everos.memory.cascade.handlers.episode import EpisodeHandler


class _EpisodeRepo(LanceIndexContract[Episode], LanceRepoBase[Episode]):
    schema = Episode


async def test_empty_session_does_not_hide_timestamp_from_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    connection = await open_lancedb_connection(root.lancedb_dir, LanceDBSettings())
    table = await connection.create_table(Episode.TABLE_NAME, schema=Episode)
    repo = _EpisodeRepo(table=table)
    monkeypatch.setattr(EpisodeHandler, "index_repo", repo)
    writer = EpisodeWriter(root)
    bucket = dt.date(2026, 4, 22)
    timestamp = "2026-04-22T10:03:11Z"
    entry_id = await writer.append_entry(
        "u1",
        date=bucket,
        inline={
            "owner_id": "u1",
            "session_id": None,
            "timestamp": timestamp,
            "parent_type": "memcell",
            "parent_id": "mc_1",
        },
        sections={"Content": "A memory without a session."},
    )
    md_path = writer.path_for("u1", bucket).relative_to(root.root).as_posix()
    handler = EpisodeHandler(HandlerDeps(memory_root=root, tokenizer=build_tokenizer()))

    outcome = await handler.handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    row = await repo.find_one_where(eq("entry_id", entry_id.format()))
    assert row is not None
    assert row.timestamp == from_iso_format(timestamp)
    assert row.parent_id == "mc_1"
    assert row.episode == "A memory without a session."
