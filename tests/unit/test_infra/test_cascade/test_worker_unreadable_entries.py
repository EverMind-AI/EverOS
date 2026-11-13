"""The worker records how many entries a projection skipped as unreadable.

A file whose projection skipped an entry a hand edit broke is still done --
the rest of it is indexed -- so the worker marks the row done and hands the
handler's ``unreadable`` count to the ledger, where ``cascade status`` and
``/health`` read it. A projection that skipped nothing marks the row done
exactly as before: ``change_type`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from everos.infra.cascade._projections import Handler
from everos.infra.cascade._worker import CascadeWorker
from everos.infra.cascade.models import HandlerOutcome


@dataclass
class _Row:
    md_path: str
    kind: str = "episode"
    change_type: str = "added"
    retry_count: int = 0
    lsn: int = 1
    mtime: float = 0.0
    status: str = "processing"


class _Ledger:
    def __init__(self, rows: list[_Row]) -> None:
        self.pending = list(rows)
        self.rows = {row.md_path: row for row in rows}
        self.done: dict[str, dict[str, object]] = {}

    async def read_one(self, md_path: str) -> _Row | None:
        return self.rows.get(md_path)

    async def claim_pending_batch(self, limit: int) -> list[_Row]:
        claimed, self.pending = self.pending[:limit], self.pending[limit:]
        return claimed

    async def mark_done(self, md_path: str, **fields: object) -> None:
        self.done[md_path] = fields

    async def mark_failed(self, md_path: str, **fields: object) -> None:
        raise AssertionError(f"{md_path} failed: {fields}")


class _Handler(Handler):
    def __init__(self, unreadable: dict[str, int]) -> None:
        self._unreadable = unreadable

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path,
            kind="episode",
            upserted=2,
            deleted=0,
            skipped=0,
            unreadable=self._unreadable.get(md_path, 0),
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        raise AssertionError("no deletion here")


async def test_the_unreadable_count_reaches_the_ledger_and_nothing_else_changes() -> (
    None
):
    ledger = _Ledger([_Row("broken.md"), _Row("clean.md")])
    worker = CascadeWorker(
        {"episode": _Handler({"broken.md": 1})},
        retry_backoff_seconds=0,
        ledger=ledger,
    )

    assert await worker.drain_once() == 2

    assert ledger.done == {
        "broken.md": {"change_type": "added", "unreadable_entries": 1},
        "clean.md": {"change_type": "added"},
    }
