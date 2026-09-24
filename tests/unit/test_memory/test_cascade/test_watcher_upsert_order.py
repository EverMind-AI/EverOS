"""Upserts for one path commit in delivery order.

Windows synthesises a ``created`` for every file under a freshly created
parent directory, so one ``mkdir -p`` + write hands the handler the same path
four or five times. Each upsert awaits the database; run concurrently, the
last committer wins, and on the Windows soak box a stale duplicate carrying
the first write's mtime overwrote the row of an atomic save (1 run in 4).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from watchdog.events import FileCreatedEvent

from everos.core.persistence import MemoryRoot
from everos.memory.cascade import watcher as watcher_mod
from everos.memory.cascade.watcher import _Handler

_M1 = 1_700_000_000
_M2 = 1_700_000_060


class _SlowFirstRepo:
    """The first upsert commits 50 ms late; the row is whatever committed last."""

    def __init__(self) -> None:
        self.calls = 0
        self.committed: list[float] = []

    async def upsert(
        self, md_path: str, *, kind: str, change_type: str, mtime: float
    ) -> int:
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(0.05)
        self.committed.append(mtime)
        return self.calls


async def test_duplicate_events_commit_in_delivery_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _SlowFirstRepo()
    monkeypatch.setattr(watcher_mod, "md_change_state_repo", repo)
    md = tmp_path.joinpath(
        "default_app",
        "default_project",
        "users",
        "u1",
        "episodes",
        "episode-2026-01-01.md",
    )
    md.parent.mkdir(parents=True)
    md.write_text("v1", encoding="utf-8")
    handler = _Handler(MemoryRoot(tmp_path), asyncio.get_running_loop())

    os.utime(md, (_M1, _M1))
    handler.on_created(FileCreatedEvent(str(md)))  # the synthetic duplicate
    os.utime(md, (_M2, _M2))
    handler.on_created(FileCreatedEvent(str(md)))  # the save that must win
    await asyncio.sleep(0.2)

    assert repo.committed == [_M1, _M2], (
        "the stale duplicate committed after the newer event; the row now "
        "carries the old mtime"
    )
